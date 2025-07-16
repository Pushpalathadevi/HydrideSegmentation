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
    from hydride_segmentation import segmentation_mask_creation as segmentationMaskCreation
except Exception:
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
        """Create a 2x2 publication-quality plot:
        (a) Original input image,
        (b) Color map of hydrides by orientation,
        (c) Hydride size distribution,
        (d) Hydride orientation (angle) distribution.
        """
        cmap = plt.get_cmap("coolwarm")
        rgb = np.zeros((*self.labels.shape, 3))
        sizes = []
        for i, angle in enumerate(self.orientations, start=1):
            mask = self.labels == i
            rgb[mask] = cmap(angle / 90)[:3]
            sizes.append(np.sum(mask))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # (a) Original input image
        ax0 = axes[0, 0]
        # Always reload original image from disk for display
        img = io.imread(self.image_path)
        if img.ndim == 3:
            img_disp = color.rgb2gray(img)
        else:
            img_disp = img
        ax0.imshow(img_disp, cmap="gray")
        ax0.set_title("(a) Original Input Image", fontsize=16, fontweight="bold")
        ax0.axis("off")

        # (b) Color map
        ax1 = axes[0, 1]
        im = ax1.imshow(rgb)
        ax1.axis("off")
        ax1.set_title("(b) Hydride Orientation Color Map", fontsize=16, fontweight="bold")
        # Annotate random hydrides if debug
        if self.debug and self.orientations:
            ids = list(range(1, len(self.orientations) + 1))
            random.shuffle(ids)
            ids = ids[:min(20, len(ids))]
            for i in ids:
                y, x = np.mean(np.nonzero(self.labels == i), axis=1)
                ax1.text(x, y, f"{self.orientations[i-1]:.0f}°", color="white",
                         ha="center", va="center", fontsize=10, fontweight="bold")
        # Color bar
        norm = plt.Normalize(0, 90)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Hydride Orientation (degrees)', fontsize=12)

        # (c) Hydride size distribution
        ax2 = axes[1, 0]
        ax2.hist(sizes, bins=20, color="dodgerblue", edgecolor="black", alpha=0.8)
        ax2.set_title("(c) Hydride Size Distribution", fontsize=16, fontweight="bold")
        ax2.set_xlabel("Hydride Size (pixels)", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        # Annotate mean and median
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)
        ax2.axvline(mean_size, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_size:.0f}")
        ax2.axvline(median_size, color="green", linestyle=":", linewidth=2, label=f"Median: {median_size:.0f}")
        ax2.legend(fontsize=10)

        # (d) Hydride orientation distribution
        ax3 = axes[1, 1]
        ax3.hist(self.orientations, bins=18, color="orange", edgecolor="black", alpha=0.8)
        ax3.set_title("(d) Hydride Orientation Distribution", fontsize=16, fontweight="bold")
        ax3.set_xlabel("Orientation Angle (degrees)", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.grid(True, linestyle="--", alpha=0.5)
        # Annotate mean and median
        mean_angle = np.mean(self.orientations)
        median_angle = np.median(self.orientations)
        ax3.axvline(mean_angle, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_angle:.1f}°")
        ax3.axvline(median_angle, color="green", linestyle=":", linewidth=2, label=f"Median: {median_angle:.1f}°")
        ax3.legend(fontsize=10)

        plt.suptitle("Hydride Segmentation & Orientation Analysis", fontsize=22, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs("tmp", exist_ok=True)
        out_path = os.path.join("tmp", "orientation_plot.png")
        plt.savefig(out_path, dpi=200)
        # Save to {basename}_hydride_orientation.png
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        hydride_out = f"{base}_hydride_orientation.png"
        plt.savefig(hydride_out, dpi=200)
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
            self.logger.debug(f"Hydride {idx:02d}: {angle:.1f}°")


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
