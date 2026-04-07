"""Analysis adapters for segmentation outputs."""

from __future__ import annotations

import numpy as np

from src.microseg.domain import MeasurementReport
from src.microseg.evaluation.hydride_metrics import analyze_mask, compute_metrics
from src.microseg.evaluation.hydride_statistics import compute_hydride_statistics


class HydrideAnalyzer:
    """Analyzer wrapping hydride metric and plot utilities."""

    def analyze(self, image: np.ndarray, mask: np.ndarray) -> MeasurementReport:
        _ = image
        stats = compute_hydride_statistics(mask)
        metrics = dict(stats.scalar_metrics)
        # Keep compatibility with phase-1 metric keys expected in tests/docs.
        base_metrics = compute_metrics(mask)
        metrics["mask_area_fraction"] = float(base_metrics["mask_area_fraction"])
        metrics["hydride_count"] = int(base_metrics["hydride_count"])
        analysis_imgs = analyze_mask(mask)
        return MeasurementReport(metrics=metrics, analysis_images_b64=analysis_imgs)
