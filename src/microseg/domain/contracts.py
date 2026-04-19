"""Core data contracts for segmentation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentationRequest:
    """Input contract for a segmentation run."""

    image_path: str
    model_id: str
    params: dict[str, Any] = field(default_factory=dict)
    include_analysis: bool = True


@dataclass
class SegmentationOutput:
    """Raw predictor output arrays."""

    image: np.ndarray
    mask: np.ndarray
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementReport:
    """Computed metrics and optional analysis images (base64 PNG strings)."""

    metrics: dict[str, float | int] = field(default_factory=dict)
    analysis_images_b64: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline response object with manifest metadata."""

    ok: bool
    model_id: str
    image: np.ndarray
    mask: np.ndarray
    metrics: dict[str, float | int] = field(default_factory=dict)
    images_b64: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Registry metadata for a model backend."""

    model_id: str
    display_name: str
    feature_family: str
    description: str = ""
    details: str = ""


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()
