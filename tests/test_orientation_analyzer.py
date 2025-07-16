import numpy as np
from hydride_segmentation.hydride_orientation_analyzer import HydrideOrientationAnalyzer


def test_orientation_from_coords_diagonal():
    coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    angle = HydrideOrientationAnalyzer._orientation_from_coords(coords)
    assert 40 < angle < 50


def test_orientation_from_coords_vertical():
    coords = np.array([[0, 0], [1, 0], [2, 0]])
    angle = HydrideOrientationAnalyzer._orientation_from_coords(coords)
    assert angle == 0 or angle == 90 or abs(angle) < 1

