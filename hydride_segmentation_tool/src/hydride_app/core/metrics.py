"""Metrics for hydride segmentation."""
from PIL import Image
import numpy as np


def area_fraction(mask: Image.Image) -> float:
    """Compute the area fraction of hydrides in ``mask``.

    Args:
        mask: Binary mask image.

    Returns:
        Fraction of pixels that are non-zero.
    """
    arr = np.array(mask.convert("L"))
    return float(np.count_nonzero(arr) / arr.size)


__all__ = ["area_fraction"]

