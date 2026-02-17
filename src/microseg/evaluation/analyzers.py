"""Analysis adapters for segmentation outputs."""

from __future__ import annotations

import numpy as np

from src.microseg.domain import MeasurementReport
from src.microseg.evaluation.hydride_metrics import analyze_mask, compute_metrics


class HydrideAnalyzer:
    """Analyzer wrapping hydride metric and plot utilities."""

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> MeasurementReport:
        _ = image
        metrics = compute_metrics(mask)
        analysis_imgs = analyze_mask(mask)
        return MeasurementReport(metrics=metrics, analysis_images_b64=analysis_imgs)
