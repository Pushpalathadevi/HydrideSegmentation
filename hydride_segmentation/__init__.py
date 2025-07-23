"""Hydride segmentation package."""

from .core.analysis import orientation_analysis, combined_figure
from .core.gui_app import HydrideSegmentationGUI
from .segmentation_mask_creation import HydrideSegmentation, run_model


def run_ml_model(*args, **kwargs):
    """Lazy import wrapper for the ML-based segmentation."""
    from .inference import run_model as _run_model
    return _run_model(*args, **kwargs)

__all__ = [
    "orientation_analysis",
    "combined_figure",
    "HydrideSegmentationGUI",
    "HydrideSegmentation",
    "run_model",
    "run_ml_model",
]
