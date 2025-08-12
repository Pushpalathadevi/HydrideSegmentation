"""Utility functions for hydride orientation analysis and plotting."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from PIL import Image
from io import BytesIO


def orientation_analysis(mask: np.ndarray):
    """Return orientation color map, size distribution plot and orientation distribution plot as PIL images."""
    labels = measure.label(mask > 0)
    orientations = []
    sizes = []
    for i in range(1, labels.max() + 1):
        region = labels == i
        filled = binary_fill_holes(region)
        dilated = morphology.binary_dilation(filled, morphology.disk(1))
        skel = morphology.skeletonize(dilated)
        coords = np.column_stack(np.nonzero(skel))[:, ::-1]
        if len(coords) < 2:
            angle = 0.0
        else:
            cov = np.cov(coords, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            vx, vy = vecs[:, np.argmax(vals)]
            angle = np.degrees(np.arctan2(vy, vx)) % 180
            if angle > 90:
                angle = 180 - angle
        orientations.append(float(angle))
        sizes.append(np.sum(region))

    cmap = plt.get_cmap('coolwarm')
    rgb = np.zeros((*labels.shape, 3))
    for i, angle in enumerate(orientations, start=1):
        rgb[labels == i] = cmap(angle / 90)[:3]

    # Orientation map
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(rgb)
    ax1.axis('off')
    norm = plt.Normalize(0, 90)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, label='Orientation (deg)')
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    orient_img = Image.open(buf1)

    # Size distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(sizes, bins=20, color='dodgerblue', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Hydride Size (pixels)')
    ax2.set_ylabel('Count')
    ax2.set_title('Hydride Size Distribution')
    buf2 = BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    size_img = Image.open(buf2)

    # Orientation distribution
    fig3, ax3 = plt.subplots()
    ax3.hist(orientations, bins=18, color='orange', edgecolor='black', alpha=0.8)
    ax3.set_xlabel('Orientation Angle (degrees)')
    ax3.set_ylabel('Count')
    ax3.set_title('Hydride Orientation Distribution')
    buf3 = BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight')
    plt.close(fig3)
    buf3.seek(0)
    angle_img = Image.open(buf3)

    return orient_img, size_img, angle_img


def combined_figure(input_img: Image.Image, mask_img: Image.Image, overlay_img: Image.Image,
                     orient_img: Image.Image, size_img: Image.Image, angle_img: Image.Image,
                     save_path: str | None = None):
    """Create 2x3 combined figure and optionally save to file."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_img, cmap='gray')
    axes[0, 1].set_title('Predicted Mask')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(overlay_img)
    axes[0, 2].set_title('Overlay')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(orient_img)
    axes[1, 0].set_title('Hydride Orientation')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(size_img)
    axes[1, 1].set_title('Size Distribution')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(angle_img)
    axes[1, 2].set_title('Orientation Distribution')
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig

from .utils import image_to_png_base64


def compute_metrics(mask: np.ndarray) -> dict:
    labels = measure.label(mask > 0)
    area_fraction = float(np.count_nonzero(mask) / mask.size)
    hydride_count = int(labels.max())
    return {
        "mask_area_fraction": area_fraction,
        "hydride_count": hydride_count,
    }


def analyze_mask(mask: np.ndarray) -> dict:
    orient_img, size_img, angle_img = orientation_analysis(mask)
    return {
        "orientation_map_png_b64": image_to_png_base64(orient_img),
        "size_histogram_png_b64": image_to_png_base64(size_img),
        "angle_histogram_png_b64": image_to_png_base64(angle_img),
    }
