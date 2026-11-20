"""Accelerated Florence-2 / Magi v3 inference primitives (SDPA, GPU n-gram, compact beam search)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAGI_DETECTION_PROMPT = "Find all panels, texts, characters, and speech-bubble tails in the image."
MAGI_MAX_INPUT_TOKENS = 1024
MAGI_MAX_NEW_TOKENS = 1024
MAGI_NUM_BEAMS = 3
MAGI_NO_REPEAT_NGRAM_SIZE = 3
MAGI_ESSENTIAL_TEXT_THRESHOLD = 0.8

_CATEGORY_BY_PREFIX: dict[str, str] = {
    "<pa": "panels",
    "<te": "texts",
    "<ch": "characters",
    "<ta": "tails",
}

try:
    from transformers.generation.logits_process import LogitsProcessor as _BaseLogitsProcessor
except ImportError:  # pragma: no cover - optional magi dependency not installed

    class _BaseLogitsProcessor:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""


def _is_concrete_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_config_attr(lm: Any, name: str, default: Any = None) -> Any:
    for container in (getattr(lm, "generation_config", None), getattr(lm, "config", None)):
        if container is None:
            continue
        value = getattr(container, name, None)
        if value is not None:
            return value
    return default


def _is_florence_kv_cache(past_key_values: Any) -> bool:
    return (
        isinstance(past_key_values, tuple)
        and len(past_key_values) > 0
        and all(isinstance(layer, tuple) and len(layer) == 4 for layer in past_key_values)
    )


class FastNoRepeatNGramLogitsProcessor(_BaseLogitsProcessor):
    """Exact GPU/CPU-vectorized drop-in replacement for HF NoRepeatNGramLogitsProcessor."""

    def __init__(self, ngram_size: int) -> None:
        if not _is_concrete_int(ngram_size) or ngram_size <= 0:
            raise ValueError(f"ngram_size must be a positive integer, got {ngram_size!r}")
        self.ngram_size = ngram_size
        self._inf_cache: dict[tuple[Any, Any], tuple[Any, Any]] = {}

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        cur_len = input_ids.shape[-1]
        n = self.ngram_size
        if cur_len < n:
            return scores
        if n == 1:
            scores.scatter_(1, input_ids, float("-inf"))
            return scores

        prefix = input_ids[:, -(n - 1) :]
        windows = input_ids[:, :-1].unfold(1, n - 1, 1)
        matches = (windows == prefix.unsqueeze(1)).all(dim=-1)
        next_tokens = input_ids[:, n - 1 :]

        if scores.device.type == "mps":
            b_idx, pos_idx = matches.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                scores[b_idx, next_tokens[b_idx, pos_idx]] = float("-inf")
            return scores

        key = (scores.device, scores.dtype)
        inf_pair = self._inf_cache.get(key)
        if inf_pair is None:
            inf_pair = (
                torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype),
                torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype),
            )
            self._inf_cache[key] = inf_pair
        neg_inf, pos_inf = inf_pair

        try:
            src = torch.where(matches, neg_inf, pos_inf)
            scores.scatter_reduce_(1, next_tokens, src, reduce="amin", include_self=True)
        except (NotImplementedError, RuntimeError):
            b_idx, pos_idx = matches.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                scores[b_idx, next_tokens[b_idx, pos_idx]] = float("-inf")
        return scores


def supports_fast_florence2(model: Any, processor: Any) -> bool:
    """Return True when ``model`` and ``processor`` expose the Florence-2 Magi v3 contract."""
    if processor is None or not callable(processor):
        return False
    if not callable(getattr(processor, "postprocess_output", None)):
        return False
    if getattr(processor, "tokenizer", None) is None:
        return False

    for method_name in (
        "_encode_image",
        "get_input_embeddings",
        "_merge_input_ids_with_image_features",
        "get_essential_text_logits",
    ):
        if not callable(getattr(model, method_name, None)):
            return False

    lm = getattr(model, "language_model", None)
    if lm is None or not callable(getattr(lm, "get_encoder", None)):
        return False

    lm_cfg = getattr(lm, "config", None)
    if lm_cfg is None:
        return False

    start_id = getattr(lm_cfg, "decoder_start_token_id", None)
    pad_id = getattr(lm_cfg, "pad_token_id", None)
    eos_id = getattr(lm_cfg, "eos_token_id", None)
    if not (_is_concrete_int(start_id) and _is_concrete_int(pad_id)):
        return False
    if isinstance(eos_id, (list, tuple)):
        return len(eos_id) > 0 and all(_is_concrete_int(item) for item in eos_id)
    return _is_concrete_int(eos_id)


def _fallback_lm_generate(
    lm: Any,
    encoder_outputs: Any,
    attention_mask: Any,
    *,
    num_beams: int,
    max_new_tokens: int,
    ngram_processor: FastNoRepeatNGramLogitsProcessor | None,
) -> Any:
    from transformers.generation.logits_process import LogitsProcessorList

    gen_kwargs: dict[str, Any] = {
        "input_ids": None,
        "encoder_outputs": encoder_outputs,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": num_beams,
    }
    if ngram_processor is not None:
        gen_kwargs["no_repeat_ngram_size"] = 0
        gen_kwargs["logits_processor"] = LogitsProcessorList([ngram_processor])
    return lm.generate(**gen_kwargs)


def compact_beam_search(
    lm: Any,
    encoder_outputs: Any,
    attention_mask: Any,
    *,
    num_beams: int = MAGI_NUM_BEAMS,
    max_new_tokens: int = MAGI_MAX_NEW_TOKENS,
    ngram_processor: FastNoRepeatNGramLogitsProcessor | None = None,
    enable_compaction: bool = True,
) -> Any:
    """Run beam search with one batched D2H/H2D sync per step and dynamic finished-page compaction."""
    import torch
    from transformers.generation.beam_search import BeamHypotheses
    from transformers.modeling_outputs import BaseModelOutput

    lm_cfg = lm.config
    device = encoder_outputs.last_hidden_state.device
    orig_batch_size = int(encoder_outputs.last_hidden_state.shape[0])
    if orig_batch_size == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)

    decoder_start_token_id = int(lm_cfg.decoder_start_token_id)
    pad_token_id = int(lm_cfg.pad_token_id)
    raw_eos = lm_cfg.eos_token_id
    eos_ids = [int(x) for x in raw_eos] if isinstance(raw_eos, (list, tuple)) else [int(raw_eos)]
    eos_id_set = set(eos_ids)
    primary_eos_id = eos_ids[0]

    forced_bos = _read_config_attr(lm, "forced_bos_token_id")
    forced_eos = _read_config_attr(lm, "forced_eos_token_id")
    length_penalty = float(_read_config_attr(lm, "length_penalty", 1.0))
    early_stopping = _read_config_attr(lm, "early_stopping", True)
    max_length = 1 + max_new_tokens

    enc_hidden = encoder_outputs.last_hidden_state.repeat_interleave(num_beams, dim=0)
    enc_mask = attention_mask.repeat_interleave(num_beams, dim=0)
    input_ids = torch.full(
        (orig_batch_size * num_beams, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    beam_scores = torch.zeros((orig_batch_size, num_beams), dtype=torch.float32, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    hyps = [
        BeamHypotheses(num_beams, length_penalty, early_stopping, max_length=max_length)
        for _ in range(orig_batch_size)
    ]
    done = [False] * orig_batch_size
    active_orig_indices = list(range(orig_batch_size))
    past_key_values: Any = None
    cur_len = 1
    n_tokens_to_keep = max(2, 1 + len(eos_id_set)) * num_beams

    while active_orig_indices and cur_len < max_length:
        cur_bs = len(active_orig_indices)
        dec_in = input_ids[:, -1:] if past_key_values is not None else input_ids
        outputs = lm(
            input_ids=None,
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            attention_mask=enc_mask,
            decoder_input_ids=dec_in,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        if past_key_values is not None and not _is_florence_kv_cache(past_key_values):
            del outputs
            return _fallback_lm_generate(
                lm,
                encoder_outputs,
                attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                ngram_processor=ngram_processor,
            )

        next_token_logits = outputs.logits[:, -1, :].float()
        del outputs

        next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        if ngram_processor is not None:
            next_token_scores = ngram_processor(input_ids, next_token_scores)
        if forced_bos is not None and cur_len == 1:
            next_token_scores.fill_(float("-inf"))
            next_token_scores[:, int(forced_bos)] = 0.0
        if forced_eos is not None and cur_len == max_length - 1:
            next_token_scores.fill_(float("-inf"))
            next_token_scores[:, int(forced_eos)] = 0.0

        next_token_scores = next_token_scores + beam_scores[:, None]
        vocab_size = int(next_token_scores.shape[-1])
        next_token_scores = next_token_scores.view(cur_bs, num_beams * vocab_size)

        topk_scores, topk_tokens = torch.topk(
            next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
        )
        topk_beam_indices = torch.div(topk_tokens, vocab_size, rounding_mode="floor")
        topk_token_ids = topk_tokens % vocab_size

        # Single batched D2H sync per decode step for all active pages & beams.
        scores_cpu = topk_scores.cpu().tolist()
        tokens_cpu = topk_token_ids.cpu().tolist()
        indices_cpu = topk_beam_indices.cpu().tolist()

        next_scores_list: list[float] = []
        next_tokens_list: list[int] = []
        next_beam_idx_list: list[int] = []
        surviving_positions: list[int] = []
        next_len = cur_len + 1

        for pos, orig_idx in enumerate(active_orig_indices):
            if done[orig_idx]:
                surviving_positions.append(pos)
                next_scores_list.extend([0.0] * num_beams)
                next_tokens_list.extend([pad_token_id] * num_beams)
                next_beam_idx_list.extend([0] * num_beams)
                continue

            hyp = hyps[orig_idx]
            row_scores = scores_cpu[pos]
            row_tokens = tokens_cpu[pos]
            row_indices = indices_cpu[pos]

            picked_scores: list[float] = []
            picked_tokens: list[int] = []
            picked_indices: list[int] = []
            for rank in range(n_tokens_to_keep):
                tok = int(row_tokens[rank])
                score = float(row_scores[rank])
                b_idx = pos * num_beams + int(row_indices[rank])
                if tok in eos_id_set:
                    if rank >= num_beams:
                        continue
                    hyp.add(input_ids[b_idx].clone(), score, generated_len=next_len - 1)
                else:
                    picked_scores.append(score)
                    picked_tokens.append(tok)
                    picked_indices.append(b_idx)
                    if len(picked_tokens) == num_beams:
                        break

            if len(picked_tokens) < num_beams:
                raise ValueError(
                    f"At most {num_beams} tokens in {row_tokens} can be eos_token_id={eos_ids}."
                )

            if hyp.is_done(float(row_scores[0]), next_len, decoder_prompt_len=1):
                done[orig_idx] = True
                if not enable_compaction:
                    surviving_positions.append(pos)
                    next_scores_list.extend(picked_scores)
                    next_tokens_list.extend(picked_tokens)
                    next_beam_idx_list.extend(picked_indices)
            else:
                surviving_positions.append(pos)
                next_scores_list.extend(picked_scores)
                next_tokens_list.extend(picked_tokens)
                next_beam_idx_list.extend(picked_indices)

        cur_len = next_len
        if all(done) or not surviving_positions:
            break

        compacted = enable_compaction and len(surviving_positions) < cur_bs
        beam_scores = torch.tensor(next_scores_list, dtype=torch.float32, device=device)
        beam_next_tokens = torch.tensor(next_tokens_list, dtype=torch.long, device=device)
        beam_idx = torch.tensor(next_beam_idx_list, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        if compacted:
            active_orig_indices = [active_orig_indices[p] for p in surviving_positions]
            keep_beam_rows = torch.tensor(
                [p * num_beams + b for p in surviving_positions for b in range(num_beams)],
                dtype=torch.long,
                device=device,
            )
            enc_hidden = enc_hidden.index_select(0, keep_beam_rows)
            enc_mask = enc_mask.index_select(0, keep_beam_rows)
            if past_key_values is not None:
                past_key_values = tuple(
                    (
                        layer[0].index_select(0, beam_idx),
                        layer[1].index_select(0, beam_idx),
                        layer[2].index_select(0, keep_beam_rows),
                        layer[3].index_select(0, keep_beam_rows),
                    )
                    for layer in past_key_values
                )
        elif past_key_values is not None:
            past_key_values = tuple(
                (
                    layer[0].index_select(0, beam_idx),
                    layer[1].index_select(0, beam_idx),
                    layer[2],
                    layer[3],
                )
                for layer in past_key_values
            )

    if active_orig_indices:
        open_scores_cpu = beam_scores.cpu().tolist()
        for pos, orig_idx in enumerate(active_orig_indices):
            if not done[orig_idx]:
                hyp = hyps[orig_idx]
                for b in range(num_beams):
                    b_idx = pos * num_beams + b
                    seq = input_ids[b_idx]
                    hyp.add(seq, float(open_scores_cpu[b_idx]), generated_len=int(seq.shape[-1]) - 1)

    best = [max(hyps[i].beams, key=lambda item: item[0])[1] for i in range(orig_batch_size)]
    sent_lens = [len(hypo) for hypo in best]
    max_out_len = min(max(sent_lens) + 1, max_length)
    decoded = torch.full((orig_batch_size, max_out_len), pad_token_id, dtype=torch.long, device=device)
    for i, hypo in enumerate(best):
        length = len(hypo)
        decoded[i, :length] = hypo
        if length < max_out_len:
            decoded[i, length] = primary_eos_id
    return decoded


@dataclass(frozen=True)
class PreparedFlorenceBatch:
    """CPU-prepared Florence-2 inputs ready for non-blocking device transfer."""

    input_ids: Any
    pixel_values: Any
    rgb_images: list[np.ndarray]


class Florence2FastRunner:
    """Three-stage CPU/GPU runner reusing encoder_outputs and omitting unused affinity heads."""

    def __init__(self, model: Any, processor: Any, torch_module: Any) -> None:
        self._model = model
        self._processor = processor
        self._torch = torch_module
        self._dtype = getattr(model, "dtype", None)
        lm = model.language_model
        raw_ngram = _read_config_attr(lm, "no_repeat_ngram_size", MAGI_NO_REPEAT_NGRAM_SIZE)
        ngram_size = int(raw_ngram) if _is_concrete_int(raw_ngram) and raw_ngram > 0 else 0
        self._ngram_processor = (
            FastNoRepeatNGramLogitsProcessor(ngram_size) if ngram_size > 0 else None
        )

    def prepare_cpu(self, rgb_images: list[np.ndarray]) -> PreparedFlorenceBatch:
        """Tokenize and image-preprocess a chunk on CPU without touching the GPU."""
        count = len(rgb_images)
        batch_inputs = self._processor(
            batch_input_text=[MAGI_DETECTION_PROMPT] * count,
            batch_input_list_of_list_of_bboxes=[[]] * count,
            batch_images=rgb_images,
            padding=True,
            truncation=True,
            max_input_length_including_image_tokens=MAGI_MAX_INPUT_TOKENS,
            max_output_length=MAGI_MAX_NEW_TOKENS,
            return_tensors="pt",
            dtype=self._dtype,
            device=None,
        )
        return PreparedFlorenceBatch(
            input_ids=batch_inputs["input_ids"],
            pixel_values=batch_inputs["pixel_values"],
            rgb_images=rgb_images,
        )

    @staticmethod
    def slice_prepared(
        batch: PreparedFlorenceBatch, start: int, end: int
    ) -> PreparedFlorenceBatch:
        """Slice a CPU-prepared batch for recursive CUDA OOM halving without re-tokenizing."""
        return PreparedFlorenceBatch(
            input_ids=batch.input_ids[start:end],
            pixel_values=batch.pixel_values[start:end],
            rgb_images=batch.rgb_images[start:end],
        )

    def infer_gpu(
        self,
        batch: PreparedFlorenceBatch,
        *,
        need_essential_text: bool,
    ) -> tuple[Any, list[list[bool]] | None]:
        """Run vision + encoder once, compacting beam search, and optional essential-text head."""
        from torch.nn.utils.rnn import pad_sequence

        model = self._model
        lm = model.language_model
        tokenizer = self._processor.tokenizer
        device = getattr(model, "device", None)

        with self._torch.inference_mode():
            input_ids = (
                batch.input_ids.to(device, non_blocking=True)
                if device is not None
                else batch.input_ids
            )
            pixel_values = (
                batch.pixel_values.to(device, non_blocking=True)
                if device is not None
                else batch.pixel_values
            )
            inputs_embeds = model.get_input_embeddings()(input_ids)
            image_features = model._encode_image(pixel_values)
            inputs_embeds, attention_mask = model._merge_input_ids_with_image_features(
                image_features, inputs_embeds
            )
            encoder_outputs = lm.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

            generated_ids = compact_beam_search(
                lm,
                encoder_outputs,
                attention_mask,
                num_beams=MAGI_NUM_BEAMS,
                max_new_tokens=MAGI_MAX_NEW_TOKENS,
                ngram_processor=self._ngram_processor,
                enable_compaction=True,
            )

            essential_masks: list[list[bool]] | None = None
            if need_essential_text:
                bos_id = tokenizer.bos_token_id
                cleaned: list[Any] = []
                for gid in generated_ids:
                    bos_hits = (gid == bos_id).nonzero(as_tuple=False)
                    idx_bos = int(bos_hits[-1, 0].item()) if bos_hits.numel() > 0 else 0
                    cleaned.append(gid[idx_bos:])
                cleaned_ids = pad_sequence(
                    cleaned, batch_first=True, padding_value=tokenizer.pad_token_id
                )
                lm_out = lm(
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=cleaned_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                ess_logits = model.get_essential_text_logits(
                    lm_out.decoder_hidden_states[-1],
                    cleaned_ids,
                    tokenizer,
                    apply_sigmoid=True,
                )
                essential_masks = [
                    (ess_logits[idx] > MAGI_ESSENTIAL_TEXT_THRESHOLD).tolist()
                    for idx in range(len(batch.rgb_images))
                ]

            return generated_ids.cpu(), essential_masks

    def decode_cpu(
        self,
        generated_ids_cpu: Any,
        rgb_images: list[np.ndarray],
        essential_masks: list[list[bool]] | None,
    ) -> list[dict[str, Any]]:
        """Decode generated token IDs into Magi panel/text dictionaries on CPU."""
        # processing_florence2.postprocess_output mutates generated_ids in place via masked_fill_.
        with self._torch.inference_mode():
            generated_texts, list_of_bboxes_batch, batch_indices = self._processor.postprocess_output(
                generated_ids_cpu, rgb_images
            )

        results: list[dict[str, Any]] = []
        for idx, (generated_text, indices, list_of_bboxes) in enumerate(
            zip(generated_texts, batch_indices, list_of_bboxes_batch)
        ):
            item: dict[str, Any] = {
                "panels": [],
                "texts": [],
                "characters": [],
                "tails": [],
            }
            for (_start, pos), bboxes in zip(indices, list_of_bboxes):
                category = _CATEGORY_BY_PREFIX.get(generated_text[pos : pos + 3])
                if category is not None:
                    item[category].extend(bboxes)
            if essential_masks is not None and idx < len(essential_masks):
                item["is_essential_text"] = essential_masks[idx]
            results.append(item)
        return results
