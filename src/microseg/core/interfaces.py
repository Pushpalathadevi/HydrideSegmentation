"""Protocol interfaces for model-agnostic segmentation components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from src.microseg.domain import MeasurementReport, SegmentationOutput


class Predictor(Protocol):
    """Predictor interface used by orchestration pipelines."""

    model_id: str

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        """Run segmentation and return image/mask arrays."""

    def predict_array(
        self,
        image: np.ndarray,
        params: dict | None = None,
        *,
        source_name: str = "in-memory image",
        progress_hook: Callable[[str, int, str], None] | None = None,
    ) -> SegmentationOutput:
        """Run segmentation without writing the source image to disk."""


class Analyzer(Protocol):
    """Analyzer interface that computes metrics and analysis artifacts."""

    def analyze(self, image, mask) -> MeasurementReport:
        """Analyze a segmentation result."""
