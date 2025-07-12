"""Analyse hydride plate orientations from a binary mask image."""

import argparse
import os
import random
import logging
import importlib

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, measure, morphology
from scipy.ndimage import binary_fill_holes

# Import segmentationMaskCreation if shouldRunSegmentation is enabled
try:
    segmentationMaskCreation = importlib.import_module("segmentationMaskCreation")
except ModuleNotFoundError:
    segmentationMaskCreation = None

class HydrideOrientationAnalyzer:
    """Compute morphological orientation of each hydride in an image."""

    def __init__(self, image_path: str, *, method: str = "default", debug: bool = False, shouldRunSegmentation: bool = False):
        self.image_path = image_path
        self.method = method
        self.debug = debug
        self.shouldRunSegmentation = shouldRunSegmentation
        self.orientations = []
        self.labels = None
        self.image = None
        self.logger = logging.getLogger("HydrideOrientationAnalyzer")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def run_segmentation(self) -> np.ndarray:
        """Run conventional segmentation and return mask as numpy array."""
        if segmentationMaskCreation is None:
            self.logger.error("segmentationMaskCreation module not found.")
            raise ImportError("segmentationMaskCreation module not found.")
        self.logger.info("Running conventional segmentation...")
        # Define default params or load from config
        params = {
            'clahe': {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
            'adaptive': {'block_size': 13, 'C': 20},
            'morph': {'kernel_size': [5, 5], 'iterations': 0},
            'area_threshold': 150,
            'crop': False,
            'crop_percent': 0
        }
        _, mask = segmentationMaskCreation.run_model(self.image_path, params)
        self.logger.info(f"Segmentation mask generated (numpy array) with shape: {mask.shape}")
        return mask

    def load_image(self) -> np.ndarray:
        """Load `image_path` and binarise, or use segmentation mask if requested."""
        if self.shouldRunSegmentation:
            mask = self.run_segmentation()
            binary = mask > 0
            self.image = mask
            return binary
        img = io.imread(self.image_path)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        self.image = img
        binary = img > 0
        return binary

    @staticmethod
    def _orientation_from_coords(coords: np.ndarray) -> float:
        """Return angle [0,90] in degrees from coordinate array."""
        if len(coords) < 2:
            return 0.0
        cov = np.cov(coords, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vx, vy = vecs[:, np.argmax(vals)]
        angle = np.degrees(np.arctan2(vy, vx)) % 180
        if angle > 90:
            angle = 180 - angle
        return float(angle)

    def _process_region_default(self, region_mask: np.ndarray) -> float:
        filled = binary_fill_holes(region_mask)
        dilated = morphology.binary_dilation(filled, morphology.disk(1))
        skel = morphology.skeletonize(dilated)
        coords = np.column_stack(np.nonzero(skel))[:, ::-1]
        return self._orientation_from_coords(coords)

    def _process_region_alt(self, region_mask: np.ndarray) -> float:
        props = measure.regionprops(region_mask.astype(int))
        if not props:
            return 0.0
        angle = np.degrees(props[0].orientation)
        angle = abs(angle) % 180
        if angle > 90:
            angle = 180 - angle
        return float(angle)

    def analyse(self) -> None:
        """Detect hydrides and compute their orientations."""
        self.logger.debug("Loading image and binarising...")
        binary = self.load_image()
        labels = measure.label(binary)
        self.labels = labels
        self.orientations = []
        for i in range(1, labels.max() + 1):
            region_mask = labels == i
            if self.method == "alternate":
                angle = self._process_region_alt(region_mask)
            else:
                angle = self._process_region_default(region_mask)
            self.orientations.append(angle)
            self.logger.debug(f"Hydride {i:02d} orientation calculated: {angle:.1f}°")

    def plot_results(self) -> None:
        """Create color map of hydrides with optional angle annotations and color bar."""
        cmap = plt.get_cmap("coolwarm")
        rgb = np.zeros((*self.labels.shape, 3))
        for i, angle in enumerate(self.orientations, start=1):
            mask = self.labels == i
            rgb[mask] = cmap(angle / 90)[:3]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(rgb)
        ax.axis("off")
        if self.debug and self.orientations:
            ids = list(range(1, len(self.orientations) + 1))
            random.shuffle(ids)
            ids = ids[:min(20, len(ids))]
            for i in ids:
                y, x = np.mean(np.nonzero(self.labels == i), axis=1)
                self.logger.debug(f"Annotating hydride {i} at ({x:.1f}, {y:.1f}) with angle {self.orientations[i-1]:.0f}°")
                ax.text(x, y, f"{self.orientations[i-1]:.0f}°", color="white",
                        ha="center", va="center", fontsize=8)
        # Add color bar for angle
        norm = plt.Normalize(0, 90)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hydride Orientation (degrees)', fontsize=10)
        os.makedirs("tmp", exist_ok=True)
        out_path = os.path.join("tmp", "orientation_plot.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        # Save to {basename}_hydride_orientation.png
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        hydride_out = f"{base}_hydride_orientation.png"
        plt.savefig(hydride_out, dpi=150)
        self.logger.log(logging.DEBUG if self.debug else logging.INFO,
                        f"✅ Orientation plot saved → {out_path} and {hydride_out}")
        if self.debug:
            plt.show()
        plt.close(fig)

    def run(self) -> None:
        self.logger.debug("Starting analysis...")
        if self.shouldRunSegmentation:
            mask_path = self.run_segmentation()
        self.analyse()
        self.logger.debug("Plotting results...")
        self.plot_results()
        for idx, angle in enumerate(self.orientations, start=1):
            self.logger.log(logging.DEBUG if self.debug else logging.INFO,
                            f"Hydride {idx:02d}: {angle:.1f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse hydride orientations")
    parser.add_argument("--image", default="test_data/syntheticHydrides.png",
                        help="Input image path")
    parser.add_argument("--method", choices=["default", "alternate"],
                        default="default", help="Orientation algorithm")
    parser.add_argument("--debug", action="store_true",
                        help="Annotate random hydrides with angle text")
    parser.add_argument("--shouldRunSegmentation", action="store_true",
                        help="Run conventional segmentation before orientation analysis")
    args = parser.parse_args()

    # Setup logging format
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    HydrideOrientationAnalyzer(
        args.image,
        method=args.method,
        debug=args.debug,
        shouldRunSegmentation=args.shouldRunSegmentation
    ).run()


if __name__ == "__main__":
    main()
