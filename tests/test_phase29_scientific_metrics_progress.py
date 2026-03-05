"""Phase 29 tests for scientific-metric progress callbacks."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.microseg.evaluation.hydride_metrics import scientific_distance_metrics


def test_phase29_scientific_metrics_progress_hook_emits_events() -> None:
    gt = np.zeros((16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    pred[9:13, 9:13] = 1

    events: list[tuple[str, dict[str, Any]]] = []

    def hook(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))

    result = scientific_distance_metrics(gt, pred, progress_hook=hook, progress_prefix="unit_case")

    assert "hydride_size_wasserstein" in result
    names = [name for name, _payload in events]
    assert "scientific_metrics_start" in names
    assert "component_scan_start" in names
    assert "component_scan_end" in names
    assert "scientific_metrics_reduce_start" in names
    assert "scientific_metrics_end" in names


def test_phase29_scientific_metrics_progress_hook_errors_are_non_fatal() -> None:
    gt = np.zeros((8, 8), dtype=np.uint8)
    pred = np.zeros((8, 8), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    pred[5:7, 5:7] = 1

    def failing_hook(_event: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("progress callback failure")

    result = scientific_distance_metrics(gt, pred, progress_hook=failing_hook)
    assert "hydride_orientation_wasserstein" in result
