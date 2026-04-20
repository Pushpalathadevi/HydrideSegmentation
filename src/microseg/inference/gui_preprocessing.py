"""GUI-facing preprocessing for ML inference paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ManualContrastAdjustment:
    """Explicit manual contrast override for one inference run."""

    black_percentile: float = 1.0
    white_percentile: float = 99.0
    gamma: float = 1.0


@dataclass(frozen=True)
class GuiInferencePreprocessConfig:
    """Configurable ML-only preprocessing controls used by the desktop GUI."""

    target_long_side: int = 512
    auto_contrast_enabled: bool = True
    contrast_mode: str = "histogram_stretch"
    manual_adjustment: ManualContrastAdjustment | None = None


@dataclass(frozen=True)
class GuiInferencePreprocessResult:
    """Prepared arrays and provenance metadata for one ML inference input."""

    original_image: np.ndarray
    resized_image_before_contrast: np.ndarray
    resized_image: np.ndarray
    processed_image: np.ndarray
    model_ready_image: np.ndarray
    original_size: tuple[int, int]
    preprocessed_size: tuple[int, int]
    original_channel_count: int
    output_channel_count: int
    channel_duplicated: bool
    resize_scale: float
    metadata: dict[str, Any]


def coerce_gui_inference_preprocess_config(payload: GuiInferencePreprocessConfig | dict[str, Any] | None) -> GuiInferencePreprocessConfig:
    """Normalize raw GUI preprocessing payloads into a typed config."""

    if isinstance(payload, GuiInferencePreprocessConfig):
        return payload
    if not isinstance(payload, dict):
        return GuiInferencePreprocessConfig()
    manual_payload = payload.get("manual_adjustment")
    manual: ManualContrastAdjustment | None = None
    if isinstance(manual_payload, ManualContrastAdjustment):
        manual = manual_payload
    elif isinstance(manual_payload, dict):
        manual = ManualContrastAdjustment(
            black_percentile=float(manual_payload.get("black_percentile", 1.0)),
            white_percentile=float(manual_payload.get("white_percentile", 99.0)),
            gamma=float(manual_payload.get("gamma", 1.0)),
        )
    return GuiInferencePreprocessConfig(
        target_long_side=max(8, int(payload.get("target_long_side", 512))),
        auto_contrast_enabled=bool(payload.get("auto_contrast_enabled", True)),
        contrast_mode=str(payload.get("contrast_mode", "histogram_stretch")).strip().lower() or "histogram_stretch",
        manual_adjustment=manual,
    )


def inspect_image_metadata(image_path: str | Path) -> dict[str, Any]:
    """Read lightweight size/channel metadata without applying preprocessing."""

    image = Image.open(image_path)
    arr = np.asarray(image)
    original_channel_count = 1 if arr.ndim == 2 else int(arr.shape[2]) if arr.ndim == 3 else 0
    height, width = image.height, image.width
    return {
        "width": int(width),
        "height": int(height),
        "original_channel_count": int(original_channel_count),
    }


def load_original_inference_image(image_path: str | Path) -> tuple[np.ndarray, int]:
    """Load an image while preserving whether it was single-channel or multi-channel."""

    return _load_original_image(image_path)


def prepare_gui_inference_input(
    image_path: str | Path,
    config: GuiInferencePreprocessConfig | dict[str, Any] | None = None,
) -> GuiInferencePreprocessResult:
    """Load, resize, contrast-adjust, and channel-normalize one ML inference input."""

    cfg = coerce_gui_inference_preprocess_config(config)
    original_image, original_channel_count = load_original_inference_image(image_path)
    resized_image_before_contrast, resize_scale = _resize_preserve_aspect(original_image, cfg.target_long_side)

    contrast_metadata: dict[str, Any] = {
        "enabled": False,
        "mode": "disabled",
        "parameters": {},
    }
    processed = resized_image_before_contrast
    if cfg.manual_adjustment is not None:
        processed = _apply_manual_contrast(resized_image_before_contrast, cfg.manual_adjustment)
        contrast_metadata = {
            "enabled": True,
            "mode": "manual",
            "parameters": asdict(cfg.manual_adjustment),
        }
    elif cfg.auto_contrast_enabled:
        processed = _apply_auto_contrast(resized_image_before_contrast)
        contrast_metadata = {
            "enabled": True,
            "mode": cfg.contrast_mode,
            "parameters": {"low_percentile": 1.0, "high_percentile": 99.0},
        }

    model_ready_image, duplicated = _ensure_three_channels(processed)
    original_size = (int(original_image.shape[1]), int(original_image.shape[0]))
    preprocessed_size = (int(processed.shape[1]), int(processed.shape[0]))
    metadata = {
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "preprocessed_size": {"width": preprocessed_size[0], "height": preprocessed_size[1]},
        "original_channel_count": int(original_channel_count),
        "preprocessed_channel_count": int(1 if processed.ndim == 2 else processed.shape[2]),
        "model_input_channel_count": int(model_ready_image.shape[2]),
        "channel_duplicated": bool(duplicated),
        "resize": {
            "policy": "preserve_aspect_ratio_long_side",
            "target_long_side": int(cfg.target_long_side),
            "scale": float(resize_scale),
            "resized_size_before_contrast": {
                "width": int(resized_image_before_contrast.shape[1]),
                "height": int(resized_image_before_contrast.shape[0]),
            },
        },
        "contrast": contrast_metadata,
    }
    return GuiInferencePreprocessResult(
        original_image=original_image,
        resized_image_before_contrast=resized_image_before_contrast,
        resized_image=processed,
        processed_image=processed,
        model_ready_image=model_ready_image,
        original_size=original_size,
        preprocessed_size=preprocessed_size,
        original_channel_count=int(original_channel_count),
        output_channel_count=int(model_ready_image.shape[2]),
        channel_duplicated=bool(duplicated),
        resize_scale=float(resize_scale),
        metadata=metadata,
    )


def rescale_mask_to_original(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Resize predicted mask back to the source image size with nearest-neighbor."""

    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    restored = pil_mask.resize((int(original_size[0]), int(original_size[1])), Image.Resampling.NEAREST)
    return np.asarray(restored, dtype=np.uint8)


def rescale_image_to_original(image: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Resize a display image back to the source size with bicubic interpolation."""

    mode = "L" if image.ndim == 2 else "RGB"
    restored = Image.fromarray(image.astype(np.uint8), mode=mode).resize(
        (int(original_size[0]), int(original_size[1])),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(restored, dtype=np.uint8)


def _load_original_image(image_path: str | Path) -> tuple[np.ndarray, int]:
    image = Image.open(image_path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8), 1
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.uint8), 1
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return rgb, int(arr.shape[2])
    raise ValueError(f"unsupported image shape for inference preprocessing: {arr.shape!r}")


def _resize_preserve_aspect(image: np.ndarray, target_long_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    long_side = max(height, width)
    scale = float(target_long_side) / float(max(1, long_side))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    mode = "L" if image.ndim == 2 else "RGB"
    resized = Image.fromarray(image, mode=mode).resize((new_width, new_height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8), scale


def _apply_auto_contrast(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return _stretch_channel(image, low_percentile=1.0, high_percentile=99.0, gamma=1.0)
    channels = [
        _stretch_channel(image[:, :, idx], low_percentile=1.0, high_percentile=99.0, gamma=1.0)
        for idx in range(image.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.uint8)


def _apply_manual_contrast(image: np.ndarray, adjustment: ManualContrastAdjustment) -> np.ndarray:
    low = float(np.clip(adjustment.black_percentile, 0.0, 49.0))
    high = float(np.clip(adjustment.white_percentile, low + 0.1, 100.0))
    gamma = float(max(0.1, adjustment.gamma))
    if image.ndim == 2:
        return _stretch_channel(image, low_percentile=low, high_percentile=high, gamma=gamma)
    channels = [
        _stretch_channel(image[:, :, idx], low_percentile=low, high_percentile=high, gamma=gamma)
        for idx in range(image.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.uint8)


def _stretch_channel(channel: np.ndarray, *, low_percentile: float, high_percentile: float, gamma: float) -> np.ndarray:
    low = float(np.percentile(channel, low_percentile))
    high = float(np.percentile(channel, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return channel.astype(np.uint8, copy=True)
    scaled = np.clip((channel.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        scaled = np.power(scaled, 1.0 / gamma)
    return np.round(scaled * 255.0).astype(np.uint8)


def _ensure_three_channels(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8), True
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=True), False
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2).astype(np.uint8), True
    raise ValueError(f"expected grayscale or RGB image, got shape {image.shape!r}")
