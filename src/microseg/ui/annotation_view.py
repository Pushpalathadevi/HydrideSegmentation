"""Annotation-layer composition helpers for correction UIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
) -> np.ndarray:
    """Compose base image and annotation layers into one RGB view."""

    base = to_rgb(base_image).astype(np.float32)
    pred = (predicted_mask > 0).astype(np.uint8)
    corr = (corrected_mask > 0).astype(np.uint8)

    out = base
    if settings.show_predicted:
        out = _blend_color(out, pred, color=(255, 180, 0), alpha=settings.predicted_alpha)

    if settings.show_corrected:
        out = _blend_color(out, corr, color=(255, 0, 0), alpha=settings.corrected_alpha)

    if settings.show_difference:
        added = (corr == 1) & (pred == 0)
        removed = (pred == 1) & (corr == 0)
        out = _blend_color(out, added, color=(0, 255, 60), alpha=settings.difference_alpha)
        out = _blend_color(out, removed, color=(180, 0, 255), alpha=settings.difference_alpha)

    return np.clip(out, 0, 255).astype(np.uint8)
