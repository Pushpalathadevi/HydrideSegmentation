from PIL import Image

from hydride_app.core.metrics import area_fraction
from hydride_app.core.segmentation import segment_hydrides


def test_area_fraction() -> None:
    img = Image.new("L", (8, 8), color=255)
    mask = segment_hydrides(img)
    fraction = area_fraction(mask)
    assert 0.0 < fraction <= 1.0

