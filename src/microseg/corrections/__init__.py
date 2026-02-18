"""Correction session and export utilities.

This module intentionally avoids importing workflow-heavy modules at import time.
Use lazy attribute resolution for optional runtime components.
"""

from __future__ import annotations

from .classes import (
    DEFAULT_CLASS_MAP,
    SegmentationClass,
    SegmentationClassMap,
    binary_remapped_foreground_values,
    colorize_index_mask,
    class_map_to_colormap,
    load_class_map,
    normalize_binary_index_mask,
    resolve_class_map,
    to_index_mask,
)

__all__ = [
    "DEFAULT_CLASS_MAP",
    "CorrectionDatasetPackager",
    "CorrectionExporter",
    "CorrectionSession",
    "SCHEMA_VERSION",
    "SegmentationClass",
    "SegmentationClassMap",
    "binary_remapped_foreground_values",
    "colorize_index_mask",
    "class_map_to_colormap",
    "load_class_map",
    "normalize_binary_index_mask",
    "resolve_class_map",
    "to_index_mask",
]


def __getattr__(name: str):
    if name in {"CorrectionDatasetPackager", "CorrectionExporter", "SCHEMA_VERSION"}:
        from .exporter import CorrectionDatasetPackager, CorrectionExporter, SCHEMA_VERSION

        mapping = {
            "CorrectionDatasetPackager": CorrectionDatasetPackager,
            "CorrectionExporter": CorrectionExporter,
            "SCHEMA_VERSION": SCHEMA_VERSION,
        }
        return mapping[name]
    if name == "CorrectionSession":
        from .session import CorrectionSession

        return CorrectionSession
    raise AttributeError(name)
