"""Conventional image segmentation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skimage import exposure, filters, morphology, util


@dataclass
class ConventionalParams:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    adaptive_window: int = 31
    adaptive_offset: int = 2
    morph_kernel: int = 3
    morph_iters: int = 1


def segment(image: np.ndarray, params: ConventionalParams) -> tuple[np.ndarray, np.ndarray]:
    """Run CLAHE → adaptive threshold → morphology."""
    img_float = image / 255.0
    clahe = exposure.equalize_adapthist(
        img_float, clip_limit=params.clahe_clip_limit, kernel_size=params.clahe_tile_grid
    )
    clahe_img = util.img_as_ubyte(clahe)
    thresh = filters.threshold_local(
        clahe_img, params.adaptive_window, offset=params.adaptive_offset
    )
    mask = (clahe_img < thresh).astype(np.uint8)
    selem = morphology.square(params.morph_kernel)
    mask = mask.astype(bool)
    for _ in range(params.morph_iters):
        mask = morphology.binary_closing(mask, selem)
    mask = (mask.astype(np.uint8)) * 255

    edges = morphology.binary_dilation(mask > 0) ^ (mask > 0)
    overlay = np.stack([image] * 3, axis=-1)
    overlay[edges] = [255, 0, 0]
    return mask, overlay
