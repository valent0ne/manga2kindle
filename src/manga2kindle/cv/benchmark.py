"""PopManga, CoMix, and local JSON/COCO benchmark harness for panel detectors."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from manga2kindle.cv.backends.base import PanelDetectorBackend
from manga2kindle.cv.boxes import Rect, compute_containment, compute_iou
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.models import DetectorKind, PanelBox, ReadingDirection

logger = logging.getLogger(__name__)

DEFAULT_POPMANGA_DATASET_ID = "ragavsachdeva/popmanga_test"
DEFAULT_COMIX_DATASET_ID = "emanuelevivoli/comix-v0_1-pages"

POPMANGA_LICENSE_NOTICE = (
    "PopManga benchmark dataset: subject to upstream academic/non-commercial "
    "research terms and attribution requirements."
)
COMIX_LICENSE_NOTICE = (
    "CoMix benchmark dataset: subject to upstream academic/non-commercial "
    "research terms and attribution requirements."
)


class BenchmarkDatasetKind(str, Enum):
    """Supported benchmark dataset sources."""

    POPMANGA = "popmanga"
    COMIX = "comix"
    LOCAL = "local"


@dataclass(frozen=True)
class BenchmarkSample:
    """Single benchmark page with its ground-truth panels in 1..N reading order."""

    page_id: str
    image_path: Path
    ground_truth: list[PanelBox]


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Aggregated detection, reading-order, and resource metrics for a backend."""

    backend: DetectorKind
    pages_evaluated: int
    precision_50: float
    recall_50: float
    precision_75: float
    recall_75: float
    mean_iou: float
    nested_duplicate_rate: float
    exact_count_match: float
    pairwise_order_acc: float
    kendall_tau: float
    ms_per_page: float
    peak_ram_mb: float
    peak_vram_mb: float


def get_dataset_license_notice(kind: BenchmarkDatasetKind) -> str | None:
    """Return the license/attribution notice for a benchmark dataset, if applicable."""
    if kind == BenchmarkDatasetKind.POPMANGA:
        return POPMANGA_LICENSE_NOTICE
    if kind == BenchmarkDatasetKind.COMIX:
        return COMIX_LICENSE_NOTICE
    return None


def default_direction_for_dataset(kind: BenchmarkDatasetKind) -> ReadingDirection:
    """Return the default reading direction for a benchmark dataset."""
    if kind == BenchmarkDatasetKind.COMIX:
        return ReadingDirection.LTR
    return ReadingDirection.RTL


def _to_rect(box: PanelBox) -> Rect:
    return (box.x, box.y, box.w, box.h)


def _greedy_match(
    preds: Sequence[PanelBox],
    gt: Sequence[PanelBox],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Perform 1-to-1 greedy IoU matching sorted by IoU descending."""
    if not preds or not gt:
        return []

    candidates: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(preds):
        p_rect = _to_rect(pred)
        for g_idx, target in enumerate(gt):
            iou = compute_iou(p_rect, _to_rect(target))
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_preds: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, p_idx, g_idx in candidates:
        if p_idx in used_preds or g_idx in used_gt:
            continue
        used_preds.add(p_idx)
        used_gt.add(g_idx)
        matches.append((p_idx, g_idx, iou))

    return matches


def _count_nested_duplicates(
    preds: Sequence[PanelBox],
    containment_threshold: float,
) -> int:
    """Count predicted boxes mostly contained inside a larger (or earlier equal) box."""
    if len(preds) < 2:
        return 0

    areas = [p.w * p.h for p in preds]
    rects = [_to_rect(p) for p in preds]
    duplicates = 0

    for i, rect_i in enumerate(rects):
        area_i = areas[i]
        for j, rect_j in enumerate(rects):
            if i == j:
                continue
            area_j = areas[j]
            if area_j < area_i or (area_j == area_i and j > i):
                continue
            if compute_containment(rect_i, rect_j) >= containment_threshold:
                duplicates += 1
                break

    return duplicates


def _page_reading_order_scores(
    matches: Sequence[tuple[int, int, float]],
    preds: Sequence[PanelBox],
    gt: Sequence[PanelBox],
) -> tuple[float, float] | None:
    """Compute (pairwise_order_acc, kendall_tau) on matched pairs when n >= 2."""
    if len(matches) < 2:
        return None

    ordered_pairs = sorted(
        matches,
        key=lambda m: (
            gt[m[1]].ordinal if gt[m[1]].ordinal > 0 else m[1] + 1,
            m[1],
        ),
    )
    pred_ranks = [
        preds[p_idx].ordinal if preds[p_idx].ordinal > 0 else p_idx + 1
        for p_idx, _, _ in ordered_pairs
    ]

    n = len(pred_ranks)
    total_pairs = n * (n - 1) / 2.0
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if pred_ranks[i] < pred_ranks[j]:
                concordant += 1
            elif pred_ranks[i] > pred_ranks[j]:
                discordant += 1

    pairwise_acc = concordant / total_pairs
    kendall_tau = (concordant - discordant) / total_pairs
    return pairwise_acc, kendall_tau


def evaluate_predictions(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[list[PanelBox]],
    *,
    backend: DetectorKind,
    ms_per_page: float = 0.0,
    peak_ram_mb: float = 0.0,
    peak_vram_mb: float = 0.0,
    containment_threshold: float = 0.85,
) -> BenchmarkMetrics:
    """Evaluate predicted panels and reading order against ground-truth samples."""
    if len(samples) != len(predictions):
        raise ValueError(
            f"samples length ({len(samples)}) must match predictions length ({len(predictions)})"
        )

    if not samples:
        return BenchmarkMetrics(
            backend=backend,
            pages_evaluated=0,
            precision_50=0.0,
            recall_50=0.0,
            precision_75=0.0,
            recall_75=0.0,
            mean_iou=0.0,
            nested_duplicate_rate=0.0,
            exact_count_match=0.0,
            pairwise_order_acc=0.0,
            kendall_tau=0.0,
            ms_per_page=ms_per_page,
            peak_ram_mb=peak_ram_mb,
            peak_vram_mb=peak_vram_mb,
        )

    tp_50 = 0
    tp_75 = 0
    total_preds = 0
    total_gt = 0
    matched_ious: list[float] = []
    nested_dups = 0
    exact_counts = 0
    pairwise_scores: list[float] = []
    tau_scores: list[float] = []

    for sample, preds in zip(samples, predictions):
        gt = sample.ground_truth
        total_preds += len(preds)
        total_gt += len(gt)

        if len(preds) == len(gt):
            exact_counts += 1

        nested_dups += _count_nested_duplicates(preds, containment_threshold)

        matches_50 = _greedy_match(preds, gt, 0.50)
        matches_75 = _greedy_match(preds, gt, 0.75)

        tp_50 += len(matches_50)
        tp_75 += len(matches_75)
        matched_ious.extend(iou for _, _, iou in matches_50)

        order_result = _page_reading_order_scores(matches_50, preds, gt)
        if order_result is not None:
            pairwise_scores.append(order_result[0])
            tau_scores.append(order_result[1])

    precision_50 = tp_50 / total_preds if total_preds > 0 else 0.0
    recall_50 = tp_50 / total_gt if total_gt > 0 else 0.0
    precision_75 = tp_75 / total_preds if total_preds > 0 else 0.0
    recall_75 = tp_75 / total_gt if total_gt > 0 else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    nested_dup_rate = nested_dups / total_preds if total_preds > 0 else 0.0
    exact_count_match = exact_counts / len(samples)
    pairwise_order_acc = (
        sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    )
    kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0.0

    return BenchmarkMetrics(
        backend=backend,
        pages_evaluated=len(samples),
        precision_50=precision_50,
        recall_50=recall_50,
        precision_75=precision_75,
        recall_75=recall_75,
        mean_iou=mean_iou,
        nested_duplicate_rate=nested_dup_rate,
        exact_count_match=exact_count_match,
        pairwise_order_acc=pairwise_order_acc,
        kendall_tau=kendall_tau,
        ms_per_page=ms_per_page,
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
    )


def _coerce_panel_box(raw: Any, default_ordinal: int) -> PanelBox:
    """Convert a dict or [x, y, w, h] sequence into a 1-based PanelBox."""
    if isinstance(raw, PanelBox):
        return raw
    if isinstance(raw, dict):
        if "bbox" in raw:
            bbox = raw["bbox"]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            x, y, w, h = raw["x"], raw["y"], raw["w"], raw["h"]
        raw_ord = raw.get("ordinal", raw.get("order", raw.get("reading_order", default_ordinal)))
        ordinal = int(raw_ord) if raw_ord is not None and int(raw_ord) > 0 else default_ordinal
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x, y, w, h = raw[0], raw[1], raw[2], raw[3]
        ordinal = default_ordinal
    else:
        raise ValueError(f"Unsupported panel annotation format: {raw!r}")

    return PanelBox(
        x=max(0, int(round(float(x)))),
        y=max(0, int(round(float(y)))),
        w=max(1, int(round(float(w)))),
        h=max(1, int(round(float(h)))),
        ordinal=ordinal,
    )


def _normalize_ground_truth(raw_panels: Sequence[Any]) -> list[PanelBox]:
    boxes = [_coerce_panel_box(item, idx) for idx, item in enumerate(raw_panels, start=1)]
    boxes.sort(key=lambda b: b.ordinal)
    return [box.model_copy(update={"ordinal": idx}) for idx, box in enumerate(boxes, start=1)]


def _resolve_page_image(pages_dir: Path, file_name: str) -> Path:
    candidate = pages_dir / file_name
    if candidate.is_file():
        return candidate
    fallback = pages_dir / Path(file_name).name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Benchmark page image not found: {candidate}")


def _annotation_sort_key(ann: dict[str, Any]) -> tuple[int, int]:
    """Return (ordinal, id) sort key for COCO annotations, handling explicit None values."""
    raw_order = None
    for key in ("ordinal", "order", "reading_order"):
        val = ann.get(key)
        if val is not None:
            raw_order = val
            break
    order_val = int(raw_order) if raw_order is not None else 10**9
    raw_id = ann.get("id")
    id_val = int(raw_id) if raw_id is not None else 0
    return (order_val, id_val)


def _extract_entry_file_name(entry: Any, idx: int) -> str:
    """Extract required image file_name/image from a local benchmark JSON entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"Expected dict entry at index {idx}, got {type(entry).__name__}")
    raw_name = entry.get("file_name") or entry.get("image")
    if not raw_name:
        raise ValueError(f"Benchmark entry at index {idx} is missing 'file_name' or 'image'")
    return str(raw_name)


def _load_coco_samples(pages_dir: Path, payload: dict[str, Any]) -> list[BenchmarkSample]:
    panel_category_ids: set[int] | None = None
    categories = payload.get("categories")
    if isinstance(categories, list) and categories:
        matched = {
            int(cat["id"])
            for cat in categories
            if isinstance(cat, dict)
            and "id" in cat
            and str(cat.get("name", "")).strip().lower() in {"panel", "panels", "frame", "frames"}
        }
        if matched:
            panel_category_ids = matched

    anns_by_image: dict[Any, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or "image_id" not in ann or "bbox" not in ann:
            continue
        if panel_category_ids is not None:
            cat_id = ann.get("category_id")
            if cat_id is None or int(cat_id) not in panel_category_ids:
                continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    samples: list[BenchmarkSample] = []
    for idx, image_entry in enumerate(payload.get("images", [])):
        if not isinstance(image_entry, dict) or "file_name" not in image_entry:
            continue
        img_id = image_entry.get("id", idx)
        file_name = str(image_entry["file_name"])
        image_path = _resolve_page_image(pages_dir, file_name)
        raw_anns = anns_by_image.get(img_id, [])
        raw_anns.sort(key=_annotation_sort_key)
        samples.append(
            BenchmarkSample(
                page_id=str(img_id),
                image_path=image_path,
                ground_truth=_normalize_ground_truth(raw_anns),
            )
        )
    return samples


def load_local_dataset(
    pages_dir: Path,
    truth_file: Path,
    *,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    """Load benchmark samples from a local directory and custom JSON or COCO JSON truth file."""
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Benchmark pages directory does not exist: {pages_dir}")
    if not truth_file.is_file():
        raise FileNotFoundError(f"Benchmark truth file does not exist: {truth_file}")

    payload = json.loads(truth_file.read_text(encoding="utf-8"))
    samples: list[BenchmarkSample] = []

    if isinstance(payload, dict) and "images" in payload and "annotations" in payload:
        samples = _load_coco_samples(pages_dir, payload)
    elif (isinstance(payload, dict) and isinstance(payload.get("pages"), list)) or isinstance(payload, list):
        entries = payload["pages"] if isinstance(payload, dict) else payload
        for idx, entry in enumerate(entries):
            file_name = _extract_entry_file_name(entry, idx)
            page_id = str(entry.get("page_id", Path(file_name).stem or idx))
            image_path = _resolve_page_image(pages_dir, file_name)
            panels = _normalize_ground_truth(entry.get("panels", []))
            samples.append(BenchmarkSample(page_id=page_id, image_path=image_path, ground_truth=panels))
    elif isinstance(payload, dict):
        for file_name, raw_panels in payload.items():
            image_path = _resolve_page_image(pages_dir, str(file_name))
            panels = _normalize_ground_truth(raw_panels)
            samples.append(
                BenchmarkSample(
                    page_id=Path(str(file_name)).stem,
                    image_path=image_path,
                    ground_truth=panels,
                )
            )
    else:
        raise ValueError(f"Unrecognized benchmark truth JSON structure in {truth_file}")

    if limit is not None:
        return samples[:limit]
    return samples


def _extract_hf_panels(row: dict[str, Any]) -> list[PanelBox]:
    raw_panels = row.get("panels") or row.get("bboxes") or row.get("boxes") or []
    return _normalize_ground_truth(raw_panels)


def _materialize_hf_split(
    dataset_id: str,
    split: str,
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load Hugging Face benchmark splits. "
            "Install the benchmark extra with: pip install manga2kindle[benchmark]"
        ) from exc

    ds = load_dataset(
        dataset_id,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    img_dir = (cache_dir or Path(tempfile.gettempdir()) / "manga2kindle_benchmarks") / (
        dataset_id.replace("/", "_") + f"_{split}"
    )
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: list[BenchmarkSample] = []
    for idx, row in enumerate(ds):
        if limit is not None and len(samples) >= limit:
            break
        page_id = str(row.get("id", row.get("page_id", f"page_{idx:05d}")))
        image = row.get("image")
        if isinstance(image, Path):
            image_path = image
        elif isinstance(image, str):
            image_path = Path(image)
        elif image is not None and hasattr(image, "save"):
            safe_id = page_id.replace("/", "_").replace("\\", "_")
            image_path = img_dir / f"{safe_id}.png"
            if not image_path.exists():
                image.save(image_path, format="PNG")
        else:
            raise ValueError(f"Dataset row '{page_id}' is missing a valid image field.")

        samples.append(
            BenchmarkSample(
                page_id=page_id,
                image_path=image_path,
                ground_truth=_extract_hf_panels(row),
            )
        )
    return samples


def load_popmanga_split(
    split: str = "test",
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
    dataset_id: str = DEFAULT_POPMANGA_DATASET_ID,
) -> list[BenchmarkSample]:
    """Load PopManga benchmark samples from Hugging Face datasets / local cache."""
    logger.info(POPMANGA_LICENSE_NOTICE)
    return _materialize_hf_split(dataset_id, split, cache_dir=cache_dir, limit=limit)


def load_comix_split(
    split: str = "test",
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
    dataset_id: str = DEFAULT_COMIX_DATASET_ID,
) -> list[BenchmarkSample]:
    """Load CoMix benchmark samples from Hugging Face datasets / local cache."""
    logger.info(COMIX_LICENSE_NOTICE)
    return _materialize_hf_split(dataset_id, split, cache_dir=cache_dir, limit=limit)


def _current_rss_mb() -> float:
    try:
        import psutil
    except ImportError:
        return 0.0
    process = psutil.Process()
    return process.memory_info().rss / (1024.0 * 1024.0)


def _cuda_module_if_available() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is not None and cuda_mod.is_available():
        return cuda_mod
    return None


def run_benchmark(
    samples: Sequence[BenchmarkSample],
    backend: PanelDetectorBackend,
    *,
    direction: ReadingDirection = ReadingDirection.RTL,
    warmup: int = 1,
    containment_threshold: float = 0.85,
) -> BenchmarkMetrics:
    """Benchmark a single backend on raw page images without full-page fallback."""
    if not samples:
        return evaluate_predictions([], [], backend=backend.kind)

    sorter = PanelSorter(direction=direction)
    cuda = _cuda_module_if_available()

    warmup_count = max(0, min(warmup, len(samples)))
    for idx in range(warmup_count):
        backend.detect(samples[idx].image_path)

    if cuda is not None:
        cuda.reset_peak_memory_stats()
        cuda.synchronize()

    peak_ram_mb = _current_rss_mb()
    predictions: list[list[PanelBox]] = []
    total_elapsed_s = 0.0

    for sample in samples:
        if cuda is not None:
            cuda.synchronize()
        start = time.perf_counter()
        result = backend.detect(sample.image_path)
        if cuda is not None:
            cuda.synchronize()
        total_elapsed_s += time.perf_counter() - start

        peak_ram_mb = max(peak_ram_mb, _current_rss_mb())

        if result.ordered:
            ordered_panels = [
                panel.model_copy(update={"ordinal": idx}) if panel.ordinal <= 0 else panel
                for idx, panel in enumerate(result.panels, start=1)
            ]
        else:
            ordered_panels = sorter.sort(
                result.panels,
                page_width=result.page_width if result.page_width > 0 else None,
            )
        predictions.append(ordered_panels)

    peak_vram_mb = (
        cuda.max_memory_allocated() / (1024.0 * 1024.0) if cuda is not None else 0.0
    )
    ms_per_page = (total_elapsed_s * 1000.0) / len(samples)

    return evaluate_predictions(
        samples,
        predictions,
        backend=backend.kind,
        ms_per_page=ms_per_page,
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
        containment_threshold=containment_threshold,
    )
