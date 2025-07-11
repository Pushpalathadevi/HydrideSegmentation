"""Analyse hydride plate orientations from a binary mask image."""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, measure, morphology
from scipy.ndimage import binary_fill_holes

class HydrideOrientationAnalyzer:
    """Compute morphological orientation of each hydride in an image."""

    def __init__(self, image_path: str, *, method: str = "default", debug: bool = False):
        self.image_path = image_path
        self.method = method
        self.debug = debug
        self.orientations = []
        self.labels = None
        self.image = None

    def load_image(self) -> np.ndarray:
        """Load `image_path` and binarise."""
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

    def plot_results(self) -> None:
        """Create color map of hydrides with optional angle annotations."""
        cmap = plt.get_cmap("coolwarm")
        rgb = np.zeros((*self.labels.shape, 3))
        for i, angle in enumerate(self.orientations, start=1):
            mask = self.labels == i
            rgb[mask] = cmap(angle / 90)[:3]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb)
        ax.axis("off")
        if self.debug and self.orientations:
            ids = list(range(1, len(self.orientations) + 1))
            random.shuffle(ids)
            ids = ids[:min(20, len(ids))]
            for i in ids:
                y, x = np.mean(np.nonzero(self.labels == i), axis=1)
                ax.text(x, y, f"{self.orientations[i-1]:.0f}°", color="white",
                        ha="center", va="center", fontsize=8)
        os.makedirs("tmp", exist_ok=True)
        out_path = os.path.join("tmp", "orientation_plot.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Orientation plot saved → {out_path}")

    def run(self) -> None:
        self.analyse()
        self.plot_results()
        for idx, angle in enumerate(self.orientations, start=1):
            print(f"Hydride {idx:02d}: {angle:.1f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse hydride orientations")
    parser.add_argument("--image", default="test_data/syntheticHydrides.png",
                        help="Input image path")
    parser.add_argument("--method", choices=["default", "alternate"],
                        default="default", help="Orientation algorithm")
    parser.add_argument("--debug", action="store_true",
                        help="Annotate random hydrides with angle text")
    args = parser.parse_args()
    HydrideOrientationAnalyzer(args.image, method=args.method,
                               debug=args.debug).run()


if __name__ == "__main__":
    main()
