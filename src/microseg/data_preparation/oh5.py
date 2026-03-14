"""Extraction of training-ready image/mask pairs from raw `.oh5` files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

DEFAULT_IMAGE_DATASET_CANDIDATES = (
    "/images",
    "/image",
    "/Image",
    "/Data/images",
    "/Data/image",
    "/Data/Image",
    "/ScanData/images",
    "/ScanData/image",
    "/ScanData/Image",
)

DEFAULT_PHASE_DATASET_CANDIDATES = (
    "/phaseId",
    "/phase_id",
    "/PhaseId",
    "/PhaseID",
    "/Data/phaseId",
    "/Data/phase_id",
    "/Data/PhaseId",
    "/ScanData/phaseId",
    "/ScanData/phase_id",
    "/ScanData/PhaseId",
)


@dataclass(frozen=True)
class Oh5ExtractionConfig:
    """Configuration for `.oh5` phase-ID dataset extraction."""

    input_dir: str
    output_dir: str
    recursive: bool = True
    glob_pattern: str = "*.oh5"
    image_dataset: str = ""
    phase_dataset: str = ""
    image_dataset_candidates: tuple[str, ...] = DEFAULT_IMAGE_DATASET_CANDIDATES
    phase_dataset_candidates: tuple[str, ...] = DEFAULT_PHASE_DATASET_CANDIDATES
    image_percentile_low: float = 1.0
    image_percentile_high: float = 99.0
    foreground_phase_ids: tuple[int, ...] = field(default_factory=tuple)
    phase_id_to_class_index: dict[str, int] = field(default_factory=dict)
    unknown_phase_action: Literal["background", "error"] = "background"
    report_name: str = "oh5_extract_report.json"


@dataclass(frozen=True)
class Oh5ExtractedSample:
    """Per-file extraction record."""

    input_path: str
    sample_name: str
    image_dataset_path: str
    phase_dataset_path: str
    image_path: str
    mask_path: str
    image_shape: list[int]
    phase_shape: list[int]
    unique_phase_ids: list[int]


@dataclass(frozen=True)
class Oh5ExtractionResult:
    """Summary for `.oh5` extraction."""

    schema_version: str
    input_dir: str
    output_dir: str
    report_path: str
    sample_count: int
    images_dir: str
    masks_dir: str
    samples: list[Oh5ExtractedSample]


def _require_h5py() -> Any:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - exercised in runtime environments without h5py
        raise RuntimeError(
            "h5py is required for `.oh5` extraction. Install it with `pip install h5py`."
        ) from exc
    return h5py


def _iter_h5_datasets(handle: Any) -> dict[str, Any]:
    datasets: dict[str, Any] = {}

    def _visit(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            datasets["/" + name.strip("/")] = obj

    handle.visititems(_visit)
    return datasets


def _normalize_dataset_key(value: str) -> str:
    return "/" + str(value).strip().strip("/").lower()


def _resolve_dataset(
    datasets: dict[str, Any],
    *,
    explicit_path: str,
    candidates: tuple[str, ...],
    label: str,
) -> tuple[str, Any]:
    normalized = {_normalize_dataset_key(path): (path, obj) for path, obj in datasets.items()}
    if str(explicit_path).strip():
        key = _normalize_dataset_key(explicit_path)
        if key not in normalized:
            available = ", ".join(sorted(datasets.keys())[:12])
            raise KeyError(f"{label} dataset {explicit_path!r} not found. Available datasets: {available}")
        return normalized[key]

    for candidate in candidates:
        key = _normalize_dataset_key(candidate)
        if key in normalized:
            return normalized[key]
    for key, payload in normalized.items():
        if any(key.endswith(_normalize_dataset_key(candidate)) for candidate in candidates):
            return payload
    available = ", ".join(sorted(datasets.keys())[:12])
    raise KeyError(f"could not auto-resolve {label} dataset. Available datasets: {available}")


def _normalize_uint8_channel(array: np.ndarray, *, low_pct: float, high_pct: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if not np.isfinite(lo):
        lo = float(np.min(values))
    if not np.isfinite(hi):
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _image_to_rgb_uint8(array: np.ndarray, *, low_pct: float, high_pct: float) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 2:
        channel = _normalize_uint8_channel(image, low_pct=low_pct, high_pct=high_pct)
        return np.stack([channel, channel, channel], axis=2)
    if image.ndim != 3:
        raise ValueError(f"unsupported image dataset rank: {image.ndim}")

    if image.shape[-1] in {1, 3, 4}:
        channels_last = image
    elif image.shape[0] in {1, 3, 4}:
        channels_last = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"unsupported image dataset shape: {list(image.shape)}")

    if channels_last.shape[-1] == 1:
        channel = _normalize_uint8_channel(channels_last[..., 0], low_pct=low_pct, high_pct=high_pct)
        return np.stack([channel, channel, channel], axis=2)
    rgb = channels_last[..., :3]
    out = np.empty(rgb.shape, dtype=np.uint8)
    for index in range(3):
        out[..., index] = _normalize_uint8_channel(rgb[..., index], low_pct=low_pct, high_pct=high_pct)
    return out


def _phase_array_to_mask(array: np.ndarray, config: Oh5ExtractionConfig) -> np.ndarray:
    phase = np.asarray(array)
    if phase.ndim == 3 and phase.shape[-1] == 1:
        phase = phase[..., 0]
    if phase.ndim != 2:
        raise ValueError(f"unsupported phase dataset shape: {list(phase.shape)}")

    if config.phase_id_to_class_index:
        mask = np.zeros(phase.shape, dtype=np.uint8)
        mapped_values = {int(raw): int(mapped) for raw, mapped in config.phase_id_to_class_index.items()}
        for raw_value, mapped_value in mapped_values.items():
            if mapped_value < 0 or mapped_value > 255:
                raise ValueError("phase_id_to_class_index values must be in [0, 255] for PNG export")
            mask[phase == raw_value] = np.uint8(mapped_value)
        if config.unknown_phase_action == "error":
            known = np.array(sorted(mapped_values.keys()), dtype=phase.dtype)
            unknown = sorted(int(v) for v in np.unique(phase[~np.isin(phase, known)]).tolist())
            if unknown:
                raise ValueError(f"unmapped phase ids found: {unknown[:12]}")
        return mask

    if config.foreground_phase_ids:
        foreground = np.isin(phase, np.asarray(config.foreground_phase_ids, dtype=phase.dtype))
        return (foreground.astype(np.uint8) * 255).astype(np.uint8)

    max_value = int(np.max(phase)) if phase.size else 0
    if max_value > 255:
        raise ValueError(
            "phase dataset contains values >255. Provide `foreground_phase_ids` or `phase_id_to_class_index`."
        )
    return phase.astype(np.uint8)


def extract_oh5_dataset(config: Oh5ExtractionConfig) -> Oh5ExtractionResult:
    """Extract RGB images and indexed/binary masks from raw `.oh5` files."""

    h5py = _require_h5py()
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    pattern = config.glob_pattern
    files = sorted(input_dir.rglob(pattern) if config.recursive else input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} found under {input_dir}")

    samples: list[Oh5ExtractedSample] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            datasets = _iter_h5_datasets(handle)
            image_dataset_path, image_dataset = _resolve_dataset(
                datasets,
                explicit_path=config.image_dataset,
                candidates=config.image_dataset_candidates,
                label="image",
            )
            phase_dataset_path, phase_dataset = _resolve_dataset(
                datasets,
                explicit_path=config.phase_dataset,
                candidates=config.phase_dataset_candidates,
                label="phase",
            )
            image_array = np.asarray(image_dataset)
            phase_array = np.asarray(phase_dataset)
            image = _image_to_rgb_uint8(
                image_array,
                low_pct=float(config.image_percentile_low),
                high_pct=float(config.image_percentile_high),
            )
            mask = _phase_array_to_mask(phase_array, config)
            unique_phase_ids = sorted(int(v) for v in np.unique(phase_array).tolist())

        sample_name = path.stem
        image_path = images_dir / f"{sample_name}.png"
        mask_path = masks_dir / f"{sample_name}.png"
        Image.fromarray(image, mode="RGB").save(image_path)
        Image.fromarray(mask).save(mask_path)

        samples.append(
            Oh5ExtractedSample(
                input_path=str(path.resolve()),
                sample_name=sample_name,
                image_dataset_path=str(image_dataset_path),
                phase_dataset_path=str(phase_dataset_path),
                image_path=str(image_path.resolve()),
                mask_path=str(mask_path.resolve()),
                image_shape=list(image.shape),
                phase_shape=list(phase_array.shape),
                unique_phase_ids=unique_phase_ids[:64],
            )
        )

    report_path = output_dir / config.report_name
    payload = {
        "schema_version": "microseg.oh5_extract.v1",
        "config": asdict(config),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "sample_count": len(samples),
        "images_dir": str(images_dir.resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "samples": [asdict(sample) for sample in samples],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return Oh5ExtractionResult(
        schema_version="microseg.oh5_extract.v1",
        input_dir=str(input_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        report_path=str(report_path.resolve()),
        sample_count=len(samples),
        images_dir=str(images_dir.resolve()),
        masks_dir=str(masks_dir.resolve()),
        samples=samples,
    )
