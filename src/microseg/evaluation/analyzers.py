"""Analysis adapters for segmentation outputs."""

from __future__ import annotations

import numpy as np

from hydride_segmentation.core.analysis import analyze_mask, compute_metrics

from src.microseg.domain import MeasurementReport


class HydrideAnalyzer:
    """Analyzer wrapping the current hydride metric and plot utilities."""

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> MeasurementReport:
        _ = image  # Reserved for future image-aware analyses.
        metrics = compute_metrics(mask)
        analysis_imgs = analyze_mask(mask)
        return MeasurementReport(metrics=metrics, analysis_images_b64=analysis_imgs)
