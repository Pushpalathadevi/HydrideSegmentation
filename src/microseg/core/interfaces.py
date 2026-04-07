"""Protocol interfaces for model-agnostic segmentation components."""

from __future__ import annotations

from typing import Protocol

from src.microseg.domain import MeasurementReport, SegmentationOutput


class Predictor(Protocol):
    """Predictor interface used by orchestration pipelines."""

    model_id: str

    def predict(self, image_path: str, params: dict | None = None) -> SegmentationOutput:
        """Run segmentation and return image/mask arrays."""


class Analyzer(Protocol):
    """Analyzer interface that computes metrics and analysis artifacts."""

    def analyze(self, image, mask) -> MeasurementReport:
        """Analyze a segmentation result."""
