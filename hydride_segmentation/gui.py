"""Entry point for the Hydride Segmentation GUI and debug mode."""
import argparse
import os
import numpy as np
from PIL import Image

from hydride_segmentation.core.gui_app import HydrideSegmentationGUI
from hydride_segmentation.core.analysis import orientation_analysis, combined_figure
from hydride_segmentation.segmentation_mask_creation import run_model

DEBUG_IMAGE = "test_data/3PB_SRT_data_generation_1817_OD_side1_8.png"
DEBUG_OUTPUT = "test_data/results/3PB_SRT_data_generation_1817_OD_side1_8_combined.png"


def run_debug() -> None:
    """Run segmentation and orientation analysis without launching the GUI."""
    params = {
        'clahe': {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
        'adaptive': {'block_size': 13, 'C': 40},
        'morph': {'kernel_size': [5, 5], 'iterations': 0},
        'area_threshold': 95,
        'crop': False,
        'crop_percent': 10
    }
    image, mask = run_model(DEBUG_IMAGE, params)
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image
    input_img = Image.fromarray(rgb)
    mask_img = Image.fromarray(mask)
    overlay_np = rgb.copy()
    overlay_np[mask > 0] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay_np)
    orient, size_plot, angle_plot = orientation_analysis(mask)
    os.makedirs(os.path.dirname(DEBUG_OUTPUT), exist_ok=True)
    combined_figure(
        input_img,
        mask_img,
        overlay_img,
        orient,
        size_plot,
        angle_plot,
        save_path=DEBUG_OUTPUT,
    )
    print(f"Debug output saved to {DEBUG_OUTPUT}")


def main() -> None:
    """Parse command line arguments and launch GUI or debug run."""
    parser = argparse.ArgumentParser(description="Hydride Segmentation GUI")
    parser.add_argument("--debug", action="store_true", help="Run in headless debug mode")
    args = parser.parse_args()

    if args.debug:
        run_debug()
    else:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
        HydrideSegmentationGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
