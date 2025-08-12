"""Hydride segmentation package."""

"""Hydride segmentation package."""

from .core.analysis import orientation_analysis, combined_figure
from .core.gui_app import HydrideSegmentationGUI
from .api import create_blueprint


def run_model(*args, **kwargs):
    """Lazy import wrapper for the conventional segmentation."""
    from .segmentation_mask_creation import run_model as _run_model
    return _run_model(*args, **kwargs)


def HydrideSegmentation(*args, **kwargs):
    """Lazy import wrapper returning ``HydrideSegmentation`` class instance."""
    from .segmentation_mask_creation import HydrideSegmentation as _HS
    return _HS(*args, **kwargs)


def run_ml_model(*args, **kwargs):
    """Lazy import wrapper for the ML-based segmentation."""
    from .inference import run_model as _run_model
    return _run_model(*args, **kwargs)


def segment_hydride_image(*args, **kwargs):
    """Wrapper for legacy segmentation function."""
    from .legacy_api import segment_hydride_image as _seg
    return _seg(*args, **kwargs)


__all__ = [
    "orientation_analysis",
    "combined_figure",
    "HydrideSegmentationGUI",
    "HydrideSegmentation",
    "run_model",
    "run_ml_model",
    "segment_hydride_image",
    "create_blueprint",
]
