"""Helper functions for GUI image processing."""

from __future__ import annotations

from typing import Tuple

from PIL import Image

from hydride_segmentation.microseg_adapter import (
    get_gui_model_options,
    is_conventional_model,
    run_pipeline_from_gui,
)

MODEL_OPTIONS = get_gui_model_options()


def model_uses_manual_params(model_name: str) -> bool:
    """Return whether selected model should expose conventional parameters."""
    return is_conventional_model(model_name)


def run_segmentation_with_result(
    params: dict, model_name: str, include_analysis: bool = False
) -> Tuple[Image.Image, Image.Image, Image.Image, object]:
    """Run segmentation via the chosen backend and return PIL images.

    Parameters
    ----------
    params: dict
        Parameters dictionary produced by the GUI.
    model_name: str
        Registry-backed display name selected in the GUI.
    """
    result = run_pipeline_from_gui(
        image_path=params["image_path"],
        model_name=model_name,
        params=params,
        include_analysis=include_analysis,
    )
    input_img = Image.fromarray(result.image if result.image.ndim != 2 else result.image)
    mask_img = Image.fromarray(result.mask)
    from src.microseg.utils import mask_overlay
    overlay_img = Image.fromarray(mask_overlay(result.image, result.mask))
    return input_img, mask_img, overlay_img, result


def run_segmentation(params: dict, model_name: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Compatibility wrapper returning the historical three-image tuple."""
    input_img, mask_img, overlay_img, _ = run_segmentation_with_result(
        params=params,
        model_name=model_name,
        include_analysis=False,
    )
    return input_img, mask_img, overlay_img
