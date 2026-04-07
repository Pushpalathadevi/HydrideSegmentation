"""Hydride segmentation package."""

from .core.analysis import orientation_analysis, combined_figure
from .api import create_blueprint
from .version import __version__


def __getattr__(name):
    """Lazy-load optional GUI symbols to avoid hard dependency at import time."""
    if name == "HydrideSegmentationGUI":
        from .core.gui_app import HydrideSegmentationGUI as _GUI
        return _GUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def run_microseg_pipeline(*args, **kwargs):
    """Wrapper for phase-1 microseg orchestration adapter."""
    from .microseg_adapter import run_pipeline as _run
    return _run(*args, **kwargs)


def launch_qt_gui(*args, **kwargs):
    """Wrapper for Qt GUI launcher."""
    from .qt_gui import launch_qt_gui as _launch
    return _launch(*args, **kwargs)


__all__ = [
    "orientation_analysis",
    "combined_figure",
    "HydrideSegmentationGUI",
    "HydrideSegmentation",
    "run_model",
    "run_ml_model",
    "segment_hydride_image",
    "run_microseg_pipeline",
    "launch_qt_gui",
    "create_blueprint",
    "__version__",
]
