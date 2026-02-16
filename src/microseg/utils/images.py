"""Image utility functions shared across microseg pipelines."""

from __future__ import annotations

import numpy as np


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB image regardless of grayscale/RGB input."""

    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    """Overlay positive mask pixels on top of input image."""

    overlay = to_rgb(image).copy()
    overlay[mask > 0] = color
    return overlay
