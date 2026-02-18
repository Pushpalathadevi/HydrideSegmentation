"""Tests for optional binary mask auto-normalization."""

from __future__ import annotations

import numpy as np

from src.microseg.corrections.classes import binary_remapped_foreground_values, normalize_binary_index_mask


def test_binary_mask_normalization_two_value_zero_background() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[:, 3:] = 7

    preserved = normalize_binary_index_mask(src, mode="off")
    normalized = normalize_binary_index_mask(src, mode="two_value_zero_background")

    assert set(np.unique(preserved).tolist()) == {0, 7}
    assert set(np.unique(normalized).tolist()) == {0, 1}


def test_binary_mask_normalization_nonzero_foreground() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[:, 2:4] = 78
    src[:, 4:6] = 80
    src[:, 6:] = 255

    normalized = normalize_binary_index_mask(src, mode="nonzero_foreground")
    remapped = binary_remapped_foreground_values(src, mode="nonzero_foreground")

    assert set(np.unique(normalized).tolist()) == {0, 1}
    assert remapped == (78, 80, 255)
