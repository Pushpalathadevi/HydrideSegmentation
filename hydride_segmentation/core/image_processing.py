"""Helper functions for GUI image processing."""

from __future__ import annotations

import importlib
from typing import Tuple

import numpy as np
from PIL import Image

# Map human readable names to backend modules
MODEL_BACKENDS = {
    "Conventional Model": "segmentation_mask_creation",
    "ML Model": "inference",
}


def run_segmentation(params: dict, model_name: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Run segmentation via the chosen backend and return PIL images.

    Parameters
    ----------
    params: dict
        Parameters dictionary produced by the GUI.
    model_name: str
        Key from ``MODEL_BACKENDS`` specifying which backend to use.
    """
    module_name = MODEL_BACKENDS.get(model_name, "inference")
    backend = importlib.import_module(f"hydride_segmentation.{module_name}")
    image, mask = backend.run_model(params["image_path"], params)

    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image

    input_img = Image.fromarray(rgb)
    mask_img = Image.fromarray(mask)
    overlay_np = rgb.copy()
    overlay_np[mask > 0] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay_np)
    return input_img, mask_img, overlay_img
