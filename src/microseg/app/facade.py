"""Application-level facade for invoking segmentation pipelines."""

from __future__ import annotations

from src.microseg.domain import SegmentationRequest
from src.microseg.pipelines import SegmentationPipeline


def run_request(request: SegmentationRequest):
    """Execute a segmentation request using default registry wiring."""

    pipeline = SegmentationPipeline.with_hydride_defaults()
    return pipeline.run(request)
