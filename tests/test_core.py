import numpy as np
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from hydride_segmentation.core.conventional import ConventionalParams, segment
from hydride_segmentation.core.analysis import compute_metrics


def _synthetic_image() -> np.ndarray:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:90, 10:30] = 255
    img[20:80, 60:80] = 255
    return img


def test_conventional_segmentation_and_metrics():
    img = _synthetic_image()
    params = ConventionalParams()
    mask, overlay = segment(img, params)
    metrics = compute_metrics(mask)
    assert metrics["hydride_count"] == 2
    assert 0 < metrics["mask_area_fraction"] < 1
    assert overlay.shape[2] == 3
