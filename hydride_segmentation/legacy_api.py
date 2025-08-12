"""Public API for hydride image segmentation and analysis."""
from __future__ import annotations

from typing import Union, Dict
import tempfile
import os

import numpy as np
from PIL import Image

from .segmentation_mask_creation import run_model as _run_conv_model
from .core.analysis import orientation_analysis

# Default parameters for the conventional segmentation backend
DEFAULT_CONVENTIONAL_PARAMS = {
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "adaptive": {"block_size": 13, "C": 20},
    "morph": {"kernel_size": [5, 5], "iterations": 0},
    "area_threshold": 150,
    "crop": False,
    "crop_percent": 0,
}


def _to_ndarray(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Convert ``image`` to an ``np.ndarray``."""
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError("image must be a numpy array or PIL.Image")


def _overlay_image(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Return overlay of ``mask`` on ``image`` as a PIL image."""
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image.copy()
    rgb[mask > 0] = [255, 0, 0]
    return Image.fromarray(rgb)


def _run_conventional(image: np.ndarray, params: dict) -> np.ndarray:
    """Run conventional segmentation on ``image`` array."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(image if image.ndim == 2 else image).convert("L").save(tmp.name)
        path = tmp.name
    try:
        _, mask = _run_conv_model(path, params)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return mask


def segment_hydride_image(
    image: Union[np.ndarray, Image.Image],
    mode: str = "ml",
    **kwargs,
) -> Dict[str, Union[Image.Image, float]]:
    """Perform segmentation and orientation analysis on a hydride image.

    Parameters
    ----------
    image:
        Input image as a ``numpy`` array or ``PIL.Image``.
    mode:
        ``"ml"`` to use the machine learning model or ``"conv"`` for the
        conventional image processing pipeline.
    **kwargs:
        Optional parameters for the conventional segmentation backend.

    Returns
    -------
    dict
        Dictionary with PIL images for the original input, mask, overlay,
        orientation map, size distribution and angle distribution along with
        the hydride area fraction as a float.
    """
    arr = _to_ndarray(image)

    if mode.lower() == "conv":
        params = DEFAULT_CONVENTIONAL_PARAMS.copy()
        params.update(kwargs)
        mask = _run_conventional(arr if arr.ndim == 2 else np.array(Image.fromarray(arr).convert("L")), params)
        original_arr = arr if arr.ndim != 2 else arr
    else:
        from .ml_api import run_inference_from_image
        mask = run_inference_from_image(arr)
        original_arr = arr

    original_img = Image.fromarray(original_arr if original_arr.ndim != 2 else original_arr)
    mask_img = Image.fromarray(mask)
    overlay_img = _overlay_image(original_arr if original_arr.ndim != 2 else original_arr, mask)

    orient_img, size_img, angle_img = orientation_analysis(mask)
    fraction = float(np.count_nonzero(mask) / mask.size)

    return {
        "original": original_img,
        "mask": mask_img,
        "overlay": overlay_img,
        "orientation_map": orient_img,
        "distribution_plot": size_img,
        "angle_distribution": angle_img,
        "hydride_area_fraction": fraction,
    }

