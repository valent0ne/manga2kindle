"""OS-native model cache hub and SHA-256-verified downloader for ONNX panel detectors."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_APP_NAME = "manga2kindle"
_DOWNLOAD_CHUNK_BYTES = 65536
_DOWNLOAD_TIMEOUT_SECONDS = 120.0

DEFAULT_ONNX_MODEL_KEY = "manga109-yolo-s"
MANGA109_YOLO_REPO_ID = "deepghs/manga109_yolo"
MANGA109_YOLO_REVISION = "6b5730a9f1f34581b87468f2c609e3269a9faccf"
MANGA109_FRAME_CLASS_ID = 2  # labels.json: ["body", "face", "frame", "text"]

DEFAULT_MAGI_MODEL_KEY = "magiv3"
MAGIV3_REPO_ID = "ragavsachdeva/magiv3"
MAGIV3_REVISION = "49c73a225122d53adbaa26d53868be81a57706e2"
MAGIV2_REPO_ID = "ragavsachdeva/magiv2"
MAGIV2_REVISION = "fbc890fec52977142e8ee00bfe26e9458b65517c"

_MANGA109_LICENSE_NOTICE = (
    "Weights from deepghs/manga109_yolo (Hugging Face), trained on the "
    "Manga109-s dataset (hal-utokyo/Manga109-s). Subject to Manga109-s "
    "academic/non-commercial dataset terms and upstream model repository license."
)

_MAGI_LICENSE_NOTICE = (
    "Weights from ragavsachdeva/magiv3 / magiv2 (Hugging Face). Subject to "
    "non-commercial / personal / research / not-for-profit use only per the "
    "upstream model card; commercial use requires a separate license from the authors."
)


@dataclass(frozen=True)
class ModelSpec:
    """Pinned metadata and provenance for a downloadable ONNX panel detector."""

    key: str
    repo_id: str
    revision: str
    filename: str
    sha256: str
    panel_class_id: int
    source_url: str
    license_notice: str
    input_size: int = 640


def _hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "manga109-yolo-s": ModelSpec(
        key="manga109-yolo-s",
        repo_id=MANGA109_YOLO_REPO_ID,
        revision=MANGA109_YOLO_REVISION,
        filename="v2023.12.07_s_yv11/model.onnx",
        sha256="e8665cb3468aa8651c3bf3c42f5727def68df1bd6b708f5e9e19ccd6357594f8",
        panel_class_id=MANGA109_FRAME_CLASS_ID,
        source_url=_hf_resolve_url(
            MANGA109_YOLO_REPO_ID,
            MANGA109_YOLO_REVISION,
            "v2023.12.07_s_yv11/model.onnx",
        ),
        license_notice=_MANGA109_LICENSE_NOTICE,
        input_size=640,
    ),
    "manga109-yolo-n": ModelSpec(
        key="manga109-yolo-n",
        repo_id=MANGA109_YOLO_REPO_ID,
        revision=MANGA109_YOLO_REVISION,
        filename="v2023.12.07_n_yv11/model.onnx",
        sha256="25c78caf504806bd35adc8630d492a7e67f7b4d035b450c69e9a4241b516ed7c",
        panel_class_id=MANGA109_FRAME_CLASS_ID,
        source_url=_hf_resolve_url(
            MANGA109_YOLO_REPO_ID,
            MANGA109_YOLO_REVISION,
            "v2023.12.07_n_yv11/model.onnx",
        ),
        license_notice=_MANGA109_LICENSE_NOTICE,
        input_size=640,
    ),
}

_MODEL_ALIASES: dict[str, str] = {
    "default": DEFAULT_ONNX_MODEL_KEY,
    "yolo": DEFAULT_ONNX_MODEL_KEY,
    "manga109-yolo": DEFAULT_ONNX_MODEL_KEY,
    "manga109_yolo": DEFAULT_ONNX_MODEL_KEY,
    MANGA109_YOLO_REPO_ID: DEFAULT_ONNX_MODEL_KEY,
}


def get_models_cache_dir() -> Path:
    """Return the OS-native cache directory for downloaded ONNX models."""
    try:
        import platformdirs
    except ImportError as exc:
        raise RuntimeError(
            "platformdirs is required for model cache resolution. "
            "Install the ONNX extra with: pip install manga2kindle[onnx]"
        ) from exc

    return Path(platformdirs.user_cache_dir(_APP_NAME, appauthor=False)) / "models"


def list_registered_models() -> list[ModelSpec]:
    """Return all registered ONNX model specifications."""
    return list(_MODEL_REGISTRY.values())


def get_model_spec(model_key: str) -> ModelSpec:
    """Look up a registered ModelSpec by key or known alias."""
    normalized = model_key.strip()
    canonical = _MODEL_ALIASES.get(normalized, normalized)
    spec = _MODEL_REGISTRY.get(canonical)
    if spec is None:
        supported = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_key}'. Registered models: {supported}")
    return spec


def get_cached_model_path(model_key: str, cache_dir: Path | None = None) -> Path:
    """Return the local cache Path for a registered model key."""
    spec = get_model_spec(model_key)
    base_dir = cache_dir if cache_dir is not None else get_models_cache_dir()
    return base_dir / spec.key / Path(spec.filename).name


def is_model_cached(model_key: str, cache_dir: Path | None = None) -> bool:
    """Return True if the registered model is already present in the local cache."""
    cached_path = get_cached_model_path(model_key, cache_dir=cache_dir)
    return cached_path.is_file() and cached_path.stat().st_size > 0


def compute_sha256(file_path: Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(file_path: Path, expected_sha256: str) -> str:
    """Verify file SHA-256 against expected hex digest; raise ValueError on mismatch."""
    actual = compute_sha256(file_path)
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {file_path}: expected {expected}, got {actual}"
        )
    return actual


def fetch_model(
    model_key: str = DEFAULT_ONNX_MODEL_KEY,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and SHA-256-verify a registered ONNX model into the local cache."""
    spec = get_model_spec(model_key)
    target_path = get_cached_model_path(spec.key, cache_dir=cache_dir)

    if target_path.is_file() and not force:
        if spec.sha256:
            verify_sha256(target_path, spec.sha256)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.part")

    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(spec.source_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
            with tmp_path.open("wb") as out_file:
                while chunk := resp.read(_DOWNLOAD_CHUNK_BYTES):
                    digest.update(chunk)
                    out_file.write(chunk)

        if spec.sha256:
            actual = digest.hexdigest()
            expected = spec.sha256.strip().lower()
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {spec.key} from {spec.source_url}: "
                    f"expected {expected}, got {actual}"
                )

        try:
            tmp_path.replace(target_path)
        except OSError:
            shutil.move(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return target_path


def resolve_model_spec(raw: str) -> tuple[Path | None, str | None]:
    """Classify a model specifier as either a local filesystem Path or a model key/ID.

    Returns ``(local_path, None)`` for filesystem paths / ``.onnx`` files, or
    ``(None, model_key)`` for registered keys, aliases, or Hugging Face repo IDs.
    """
    value = raw.strip()
    if not value:
        raise ValueError("Model specifier must not be empty")

    is_windows_drive = len(value) >= 2 and value[1] == ":" and value[0].isalpha()
    is_unc_or_relative = value.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~"))
    has_onnx_suffix = value.lower().endswith(".onnx")

    if is_windows_drive or is_unc_or_relative or has_onnx_suffix:
        return (Path(value).expanduser(), None)

    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        return (candidate_path, None)

    canonical = _MODEL_ALIASES.get(value, value)
    return (None, canonical)


@dataclass(frozen=True)
class MagiModelSpec:
    """Pinned metadata and provenance for a Hugging Face Magi model repository."""

    key: str
    repo_id: str
    revision: str
    license_notice: str
    auto_model_class: str = "AutoModelForCausalLM"
    uses_processor: bool = True


_MAGI_REGISTRY: dict[str, MagiModelSpec] = {
    "magiv3": MagiModelSpec(
        key="magiv3",
        repo_id=MAGIV3_REPO_ID,
        revision=MAGIV3_REVISION,
        license_notice=_MAGI_LICENSE_NOTICE,
        auto_model_class="AutoModelForCausalLM",
        uses_processor=True,
    ),
    "magiv2": MagiModelSpec(
        key="magiv2",
        repo_id=MAGIV2_REPO_ID,
        revision=MAGIV2_REVISION,
        license_notice=_MAGI_LICENSE_NOTICE,
        auto_model_class="AutoModel",
        uses_processor=False,
    ),
}

_MAGI_ALIASES: dict[str, str] = {
    "magi": DEFAULT_MAGI_MODEL_KEY,
    "magiv3": "magiv3",
    MAGIV3_REPO_ID: "magiv3",
    "magiv2": "magiv2",
    MAGIV2_REPO_ID: "magiv2",
}


def list_registered_magi_models() -> list[MagiModelSpec]:
    """Return all registered Magi model specifications."""
    return list(_MAGI_REGISTRY.values())


def get_magi_model_spec(model_key: str) -> MagiModelSpec:
    """Look up a registered MagiModelSpec by key or known alias."""
    normalized = model_key.strip()
    canonical = _MAGI_ALIASES.get(normalized, normalized)
    spec = _MAGI_REGISTRY.get(canonical)
    if spec is None:
        supported = ", ".join(_MAGI_REGISTRY.keys())
        raise ValueError(f"Unknown Magi model '{model_key}'. Registered models: {supported}")
    return spec


def get_cached_magi_dir(
    model_key: str = DEFAULT_MAGI_MODEL_KEY,
    cache_dir: Path | None = None,
) -> Path:
    """Return the local cache directory Path for a registered Magi model."""
    spec = get_magi_model_spec(model_key)
    base_dir = cache_dir if cache_dir is not None else get_models_cache_dir()
    return base_dir / spec.key


def is_magi_cached(
    model_key: str = DEFAULT_MAGI_MODEL_KEY,
    cache_dir: Path | None = None,
) -> bool:
    """Return True if the registered Magi snapshot directory is present locally."""
    target_dir = get_cached_magi_dir(model_key, cache_dir=cache_dir)
    return target_dir.is_dir() and (target_dir / "config.json").is_file()


def fetch_magi_model(
    model_key: str = DEFAULT_MAGI_MODEL_KEY,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download a pinned Magi model snapshot via huggingface_hub into the local cache."""
    spec = get_magi_model_spec(model_key)
    target_dir = get_cached_magi_dir(spec.key, cache_dir=cache_dir)

    if not force and is_magi_cached(spec.key, cache_dir=cache_dir):
        return target_dir

    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download Magi weights. "
            "Install the Magi extra with: pip install manga2kindle[magi]"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(target_dir),
        force_download=force,
    )
    return target_dir


def resolve_magi_model_spec(raw: str) -> tuple[Path | None, str | None]:
    """Classify a Magi model specifier as a local directory Path or registered model key."""
    value = raw.strip()
    if not value:
        raise ValueError("Magi model specifier must not be empty")

    is_windows_drive = len(value) >= 2 and value[1] == ":" and value[0].isalpha()
    is_unc_or_relative = value.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~"))

    if is_windows_drive or is_unc_or_relative:
        return (Path(value).expanduser(), None)

    candidate_path = Path(value).expanduser()
    if candidate_path.is_dir():
        return (candidate_path, None)

    canonical = _MAGI_ALIASES.get(value, value)
    return (None, canonical)
