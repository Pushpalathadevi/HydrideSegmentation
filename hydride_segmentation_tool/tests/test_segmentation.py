from PIL import Image

from hydride_app.core.segmentation import segment_hydrides


def test_segment_hydrides() -> None:
    img = Image.new("RGB", (10, 10), color="black")
    result = segment_hydrides(img)
    assert result.size == (img.height, img.width)

