import numpy as np
import cv2
from hydride_segmentation.prepare_dataset import DatasetPreparer


def test_mask_raw_ok():
    assert DatasetPreparer._mask_raw_ok(np.array([[0, 1]], dtype=np.uint8))
    assert DatasetPreparer._mask_raw_ok(np.array([[0, 255]], dtype=np.uint8))
    assert DatasetPreparer._mask_raw_ok(np.array([[0, 0]], dtype=np.uint8))


def test_read_mask_binary(tmp_path):
    mask_path = tmp_path / 'mask.png'
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[1:3, 1:3] = [255, 0, 0]
    cv2.imwrite(str(mask_path), rgb)
    bin_mask = DatasetPreparer._read_mask_binary(mask_path)
    assert bin_mask.dtype == np.uint8
    assert set(np.unique(bin_mask)) <= {0, 1}
    assert bin_mask.shape == (5, 5)

