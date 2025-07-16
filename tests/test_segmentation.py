import cv2
import numpy as np
from hydride_segmentation.segmentation_mask_creation import run_model

TEST_IMG = 'test_data/3PB_SRT_data_generation_1817_OD_side1_8.png'

# Default params for faster test
PARAMS = {
    'clahe': {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
    'adaptive': {'block_size': 13, 'C': 20},
    'morph': {'kernel_size': [5, 5], 'iterations': 0},
    'area_threshold': 50,
    'crop': False,
    'crop_percent': 0,
}

def test_run_model_basic():
    image, mask = run_model(TEST_IMG, PARAMS)
    assert isinstance(image, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert image.shape == mask.shape
    assert mask.dtype == np.uint8
    assert mask.shape[0] > 0 and mask.shape[1] > 0


def test_run_model_crop():
    params = PARAMS.copy()
    params['crop'] = True
    params['crop_percent'] = 10
    orig = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
    image, _ = run_model(TEST_IMG, params)
    expected_h = orig.shape[0] - int(orig.shape[0] * 0.1)
    assert image.shape[0] == expected_h

