"""Hydride-oriented scientific metrics and plots for segmentation masks."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, find_objects
from scipy.stats import ks_2samp, wasserstein_distance
from skimage import measure, morphology

from src.microseg.utils import image_to_png_base64

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ProgressHook = Callable[[str, dict[str, Any]], None]

#: Border kept around each component crop so a ``disk(1)`` dilation cannot be
#: clipped by the crop edge. Two pixels leaves one pixel of slack beyond the
#: single-pixel expansion, which keeps cropped results identical to whole-image
#: results while avoiding a full-image pass per component.
_COMPONENT_CROP_PADDING = 2


def _padded_component_slice(
    component_slice: tuple[slice, slice],
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    """Grow a component bounding box by the morphology padding, clipped to the image.

    Parameters
    ----------
    component_slice:
        Row and column slices of one connected component.
    shape:
        Height and width of the full label image.

    Returns
    -------
    tuple of slice
        Padded row and column slices safe for fill, dilation, and skeletonization.
    """

    rows, cols = component_slice
    return (
        slice(
            max(0, rows.start - _COMPONENT_CROP_PADDING),
            min(shape[0], rows.stop + _COMPONENT_CROP_PADDING),
        ),
        slice(
            max(0, cols.start - _COMPONENT_CROP_PADDING),
            min(shape[1], cols.stop + _COMPONENT_CROP_PADDING),
        ),
    )


def _emit_progress(progress_hook: ProgressHook | None, event: str, **payload: Any) -> None:
    if progress_hook is None:
        return
    try:
        progress_hook(event, payload)
    except Exception:
        # Progress reporting must never interrupt metric computation.
        return


def compute_metrics(mask: np.ndarray) -> dict[str, float | int]:
    """Compute area-fraction and connected-feature count from a binary mask."""

    labels = measure.label(mask > 0)
    return {
        "mask_area_fraction": float(np.count_nonzero(mask) / mask.size),
        "hydride_count": int(labels.max()),
    }


def _component_orientations_and_sizes(
    mask: np.ndarray,
    *,
    progress_hook: ProgressHook | None = None,
    progress_prefix: str = "",
) -> tuple[list[float], list[int], np.ndarray]:
    labels = measure.label(mask > 0)
    orientations: list[float] = []
    sizes: list[int] = []
    total_components = int(labels.max())
    prefix = str(progress_prefix).strip()
    report_every = max(1, min(250, total_components))
    _emit_progress(
        progress_hook,
        "component_scan_start",
        prefix=prefix,
        total_components=total_components,
    )
    # Each component is measured inside its own padded bounding box. Operating on
    # the whole image once per component is quadratic in component count, which
    # dominates runtime on masks with many small features.
    component_slices = find_objects(labels)
    footprint = morphology.disk(1)
    for idx in range(1, total_components + 1):
        component_slice = component_slices[idx - 1] if idx - 1 < len(component_slices) else None
        if component_slice is None:
            orientations.append(0.0)
            sizes.append(0)
            continue

        window = _padded_component_slice(component_slice, labels.shape)
        region = labels[window] == idx
        filled = binary_fill_holes(region)
        dilated = morphology.dilation(filled, footprint)
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
        sizes.append(int(np.count_nonzero(region)))
        if idx == 1 or idx == total_components or idx % report_every == 0:
            _emit_progress(
                progress_hook,
                "component_scan_progress",
                prefix=prefix,
                done_components=int(idx),
                total_components=total_components,
            )
    _emit_progress(
        progress_hook,
        "component_scan_end",
        prefix=prefix,
        total_components=total_components,
    )
    return orientations, sizes, labels


def analyze_mask(mask: np.ndarray) -> dict[str, str]:
    """Return base64-encoded orientation and distribution plots."""

    orientations, sizes, labels = _component_orientations_and_sizes(mask)
    cmap = plt.get_cmap("coolwarm")
    # Colour every component in one pass through a label-indexed lookup table
    # instead of one full-image comparison per component.
    colour_table = np.zeros((len(orientations) + 1, 3), dtype=float)
    for idx, angle in enumerate(orientations, start=1):
        colour_table[idx] = cmap(angle / 90.0)[:3]
    rgb = colour_table[np.clip(labels, 0, len(orientations))]

    fig1, ax1 = plt.subplots()
    ax1.imshow(rgb)
    ax1.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, label="Orientation (deg)")
    b1 = BytesIO()
    fig1.savefig(b1, format="png", bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(sizes, bins=20, color="dodgerblue", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Hydride Size (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("Hydride Size Distribution")
    b2 = BytesIO()
    fig2.savefig(b2, format="png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(orientations, bins=18, color="orange", edgecolor="black", alpha=0.8)
    ax3.set_xlabel("Orientation Angle (degrees)")
    ax3.set_ylabel("Count")
    ax3.set_title("Hydride Orientation Distribution")
    b3 = BytesIO()
    fig3.savefig(b3, format="png", bbox_inches="tight")
    plt.close(fig3)

    b1.seek(0)
    b2.seek(0)
    b3.seek(0)
    return {
        "orientation_map_png_b64": image_to_png_base64(Image.open(b1)),
        "size_histogram_png_b64": image_to_png_base64(Image.open(b2)),
        "angle_histogram_png_b64": image_to_png_base64(Image.open(b3)),
    }


def scientific_distance_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    progress_hook: ProgressHook | None = None,
    progress_prefix: str = "",
) -> dict[str, float]:
    """Compute scientific-distribution distances for binary masks.

    Parameters
    ----------
    gt_mask:
        Ground-truth binary mask.
    pred_mask:
        Predicted binary mask.
    progress_hook:
        Optional callback used for heartbeat-style progress updates. Signature:
        ``callback(event: str, payload: dict[str, Any])``.
    progress_prefix:
        Optional context string emitted with progress callbacks.
    """
    prefix = str(progress_prefix).strip()
    _emit_progress(progress_hook, "scientific_metrics_start", prefix=prefix)

    gt_prefix = f"{prefix}:gt" if prefix else "gt"
    pr_prefix = f"{prefix}:pred" if prefix else "pred"
    gt_orient, gt_sizes, _ = _component_orientations_and_sizes(
        gt_mask,
        progress_hook=progress_hook,
        progress_prefix=gt_prefix,
    )
    pr_orient, pr_sizes, _ = _component_orientations_and_sizes(
        pred_mask,
        progress_hook=progress_hook,
        progress_prefix=pr_prefix,
    )

    def _safe_dist(a: list[float], b: list[float]) -> tuple[float, float]:
        if not a and not b:
            return 0.0, 0.0
        if not a:
            a = [0.0]
        if not b:
            b = [0.0]
        return float(wasserstein_distance(a, b)), float(ks_2samp(a, b, method="auto").statistic)

    _emit_progress(progress_hook, "scientific_metrics_reduce_start", prefix=prefix)
    size_wd, size_ks = _safe_dist([float(v) for v in gt_sizes], [float(v) for v in pr_sizes])
    orient_wd, orient_ks = _safe_dist(gt_orient, pr_orient)

    gt_basic = compute_metrics(gt_mask)
    pr_basic = compute_metrics(pred_mask)

    result = {
        "mask_area_fraction_gt": float(gt_basic["mask_area_fraction"]),
        "mask_area_fraction_pred": float(pr_basic["mask_area_fraction"]),
        "mask_area_fraction_abs_error": abs(float(gt_basic["mask_area_fraction"]) - float(pr_basic["mask_area_fraction"])),
        "hydride_count_gt": float(gt_basic["hydride_count"]),
        "hydride_count_pred": float(pr_basic["hydride_count"]),
        "hydride_count_abs_error": abs(float(gt_basic["hydride_count"]) - float(pr_basic["hydride_count"])),
        "hydride_size_wasserstein": size_wd,
        "hydride_size_ks": size_ks,
        "hydride_orientation_wasserstein": orient_wd,
        "hydride_orientation_ks": orient_ks,
    }
    _emit_progress(progress_hook, "scientific_metrics_end", prefix=prefix)
    return result
