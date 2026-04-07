"""Tests for annotation layer composition utilities."""

from __future__ import annotations

import numpy as np

from src.microseg.ui import AnnotationLayerSettings, compose_annotation_view


def test_annotation_compose_changes_pixels_for_enabled_layers() -> None:
    base = np.zeros((20, 20), dtype=np.uint8)
    pred = np.zeros((20, 20), dtype=np.uint8)
    corr = np.zeros((20, 20), dtype=np.uint8)
    pred[5:10, 5:10] = 255
    corr[8:13, 8:13] = 255

    settings = AnnotationLayerSettings(
        show_predicted=True,
        show_corrected=True,
        show_difference=True,
        predicted_alpha=0.5,
        corrected_alpha=0.6,
        difference_alpha=0.7,
    )
    out = compose_annotation_view(base, pred, corr, settings)

    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8
    assert int(out.sum()) > 0


def test_annotation_compose_base_only_when_layers_disabled() -> None:
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    base[..., 1] = 100
    pred = np.ones((10, 10), dtype=np.uint8) * 255
    corr = np.zeros((10, 10), dtype=np.uint8)

    settings = AnnotationLayerSettings(
        show_predicted=False,
        show_corrected=False,
        show_difference=False,
        predicted_alpha=1.0,
        corrected_alpha=1.0,
        difference_alpha=1.0,
    )
    out = compose_annotation_view(base, pred, corr, settings)

    assert np.array_equal(out, base)
