import numpy as np
from PIL import Image

from hydride_segmentation.api import segment_hydride_image

TEST_IMG = 'test_data/syntheticHydrides.png'


def test_segment_hydride_image_conv():
    img = Image.open(TEST_IMG)
    result = segment_hydride_image(img, mode='conv')
    expected_keys = {
        'original',
        'mask',
        'overlay',
        'orientation_map',
        'distribution_plot',
        'angle_distribution',
        'hydride_area_fraction',
    }
    assert expected_keys.issubset(result.keys())
    for key in expected_keys - {'hydride_area_fraction'}:
        assert isinstance(result[key], Image.Image)
    frac = result['hydride_area_fraction']
    assert isinstance(frac, float)
    assert 0.0 <= frac <= 1.0

