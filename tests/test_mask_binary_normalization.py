"""Tests for optional binary mask auto-normalization."""

from __future__ import annotations

import numpy as np

from src.microseg.corrections.classes import normalize_binary_index_mask


def test_binary_mask_normalization_two_value_zero_background() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[:, 3:] = 7

    preserved = normalize_binary_index_mask(src, mode="off")
    normalized = normalize_binary_index_mask(src, mode="two_value_zero_background")

    assert set(np.unique(preserved).tolist()) == {0, 7}
    assert set(np.unique(normalized).tolist()) == {0, 1}
