from pathlib import Path
from PIL import Image

from hydride_app.core.image_io import load_image, save_image


def test_load_and_save_image(tmp_path: Path) -> None:
    img_path = tmp_path / "input.png"
    img = Image.new("RGB", (8, 8), color="white")
    img.save(img_path)

    loaded = load_image(img_path)
    test_path = tmp_path / "output.png"
    save_image(loaded, test_path)
    roundtrip = load_image(test_path)
    assert roundtrip.size == img.size

