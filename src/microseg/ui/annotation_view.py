"""Annotation-layer composition helpers for correction UIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.microseg.corrections.classes import DEFAULT_CLASS_MAP, SegmentationClassMap, to_index_mask
from src.microseg.utils import to_rgb


@dataclass
class AnnotationLayerSettings:
    """Display settings for layered correction visualization."""

    show_predicted: bool = True
    show_corrected: bool = True
    show_difference: bool = True
    predicted_alpha: float = 0.35
    corrected_alpha: float = 0.45
    difference_alpha: float = 0.70


def _blend_color(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    if alpha <= 0:
        return out
    alpha = float(np.clip(alpha, 0.0, 1.0))
    idx = mask > 0
    if not np.any(idx):
        return out
    fg = np.array(color, dtype=np.float32)
    out[idx] = (1.0 - alpha) * out[idx] + alpha * fg
    return out


def compose_annotation_view(
    base_image: np.ndarray,
    predicted_mask: np.ndarray,
    corrected_mask: np.ndarray,
    settings: AnnotationLayerSettings,
    class_map: SegmentationClassMap = DEFAULT_CLASS_MAP,
) -> np.ndarray:
    """Compose base image and annotation layers into one RGB view."""

    base = to_rgb(base_image).astype(np.float32)
    pred = to_index_mask(predicted_mask)
    corr = to_index_mask(corrected_mask)

    out = base
    if settings.show_predicted:
        for cls in class_map.classes:
            if cls.index == 0:
                continue
            out = _blend_color(out, pred == cls.index, color=cls.color_rgb, alpha=settings.predicted_alpha)

    if settings.show_corrected:
        for cls in class_map.classes:
            if cls.index == 0:
                continue
            out = _blend_color(out, corr == cls.index, color=cls.color_rgb, alpha=settings.corrected_alpha)

    if settings.show_difference:
        added = (corr > 0) & (corr != pred)
        removed = (pred > 0) & (corr == 0)
        out = _blend_color(out, added, color=(0, 255, 60), alpha=settings.difference_alpha)
        out = _blend_color(out, removed, color=(180, 0, 255), alpha=settings.difference_alpha)

    return np.clip(out, 0, 255).astype(np.uint8)
