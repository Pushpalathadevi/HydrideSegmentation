import numpy as np
from PIL import Image
from hydride_segmentation.core.analysis import orientation_analysis, combined_figure


def test_orientation_analysis_returns_images():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[30:40, 30:45] = 1
    orient_img, size_img, angle_img = orientation_analysis(mask)
    for img in (orient_img, size_img, angle_img):
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0


def test_combined_figure(tmp_path):
    base_img = Image.new('L', (20, 20), color=0)
    mask_img = Image.new('L', (20, 20), color=0)
    overlay = Image.new('RGB', (20, 20), color='red')
    dummy = Image.new('RGB', (10, 10), color='blue')
    fig = combined_figure(base_img, mask_img, overlay, dummy, dummy, dummy,
                          save_path=str(tmp_path / 'out.png'))
    assert hasattr(fig, 'savefig')
    assert (tmp_path / 'out.png').exists()

