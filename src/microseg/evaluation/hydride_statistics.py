"""Hydride statistics and visualization utilities for desktop reporting."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import log2, pi, sqrt

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class HydrideVisualizationConfig:
    """Configuration for hydride distribution plots.

    Parameters
    ----------
    orientation_bins:
        Number of bins used for orientation-distribution histograms.
    size_bins:
        Number of bins used for size-distribution histograms.
    min_feature_pixels:
        Minimum connected-component area (pixels) retained for statistics.
    orientation_cmap:
        Matplotlib colormap name used for orientation map rendering.
    size_scale:
        Histogram x-axis scale for feature size plots. Allowed values are
        ``"linear"`` and ``"log"``.
    """

    orientation_bins: int = 18
    size_bins: int = 20
    min_feature_pixels: int = 1
    orientation_cmap: str = "coolwarm"
    size_scale: str = "linear"


@dataclass
class HydrideStatisticsResult:
    """Computed hydride statistics payload."""

    scalar_metrics: dict[str, float | int]
    orientations_deg: list[float]
    sizes_px: list[int]
    feature_label_ids: list[int]
    orientation_hist_counts: list[int]
    orientation_hist_edges_deg: list[float]
    size_hist_counts: list[int]
    size_hist_edges_px: list[float]
    size_hist_counts_um2: list[int]
    size_hist_edges_um2: list[float]
    sizes_um2: list[float]
    equivalent_diameters_px: list[float]
    equivalent_diameters_um: list[float]
    microns_per_pixel: float | None
    label_map: np.ndarray


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Serialize a Matplotlib figure into an RGB array."""

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _safe_size_histogram(values: list[int], bins: int) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        edges = np.linspace(0.0, 1.0, num=max(2, bins + 1), dtype=np.float64)
        return np.zeros(max(1, bins), dtype=np.int64), edges
    vmax = max(float(max(values)), 1.0)
    counts, edges = np.histogram(values, bins=max(1, bins), range=(0.0, vmax))
    return counts.astype(np.int64), edges.astype(np.float64)


def _safe_histogram_float(values: list[float], bins: int) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        edges = np.linspace(0.0, 1.0, num=max(2, bins + 1), dtype=np.float64)
        return np.zeros(max(1, bins), dtype=np.int64), edges
    vmax = max(float(max(values)), 1.0)
    counts, edges = np.histogram(values, bins=max(1, bins), range=(0.0, vmax))
    return counts.astype(np.int64), edges.astype(np.float64)


def _scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _orientation_alignment_index(orientations_deg: list[float]) -> float:
    """Return alignment index in [0, 1] for orientation angles in degrees."""

    if not orientations_deg:
        return 0.0
    rad = np.deg2rad(np.asarray(orientations_deg, dtype=np.float64))
    # 180-degree periodicity for line orientation.
    vec = np.exp(1j * 2.0 * rad)
    return float(np.abs(np.mean(vec)))


def _orientation_entropy_bits(hist_counts: np.ndarray) -> float:
    total = int(np.sum(hist_counts))
    if total <= 0:
        return 0.0
    probs = (hist_counts / float(total)).astype(np.float64)
    return float(-np.sum([p * log2(p) for p in probs if p > 0]))


def compute_hydride_statistics(
    mask: np.ndarray,
    *,
    orientation_bins: int = 18,
    size_bins: int = 20,
    min_feature_pixels: int = 1,
    microns_per_pixel: float | None = None,
    include_extended_metrics: bool = True,
    include_histograms: bool = True,
    include_physical_metrics: bool = True,
) -> HydrideStatisticsResult:
    """Compute hydride statistics from a segmented mask.

    Parameters
    ----------
    mask:
        Segmentation mask where foreground hydrides are any non-zero values.
    orientation_bins:
        Number of bins for orientation-distribution histograms.
    size_bins:
        Number of bins for size-distribution histograms.
    min_feature_pixels:
        Minimum connected-component area retained in the analysis.
    microns_per_pixel:
        Optional spatial calibration ratio. When provided, additional physical
        metrics (um, um^2) are emitted.

    Returns
    -------
    HydrideStatisticsResult
        Scalar summaries, histogram vectors, and feature-wise measurements.
    """

    if min_feature_pixels < 1:
        min_feature_pixels = 1
    orientation_bins = max(1, int(orientation_bins))
    size_bins = max(1, int(size_bins))

    binary = np.asarray(mask) > 0
    label_map = measure.label(binary, connectivity=2)

    orientations: list[float] = []
    sizes: list[int] = []
    kept_label_ids: list[int] = []
    excluded_small = 0
    n_labels = int(label_map.max())

    for label_id in range(1, n_labels + 1):
        region = label_map == label_id
        area = int(np.count_nonzero(region))
        if area < min_feature_pixels:
            excluded_small += 1
            continue

        filled = binary_fill_holes(region)
        dilated = morphology.dilation(filled, morphology.disk(1))
        skel = morphology.skeletonize(dilated)
        coords = np.column_stack(np.nonzero(skel))[:, ::-1]
        if len(coords) < 2:
            angle = 0.0
        else:
            cov = np.cov(coords, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            vx, vy = vecs[:, int(np.argmax(vals))]
            angle = float(np.degrees(np.arctan2(vy, vx)) % 180.0)
            if angle > 90.0:
                angle = 180.0 - angle

        orientations.append(float(angle))
        sizes.append(area)
        kept_label_ids.append(label_id)

    if bool(include_histograms):
        orient_counts, orient_edges = np.histogram(
            orientations,
            bins=orientation_bins,
            range=(0.0, 90.0),
        )
        size_counts, size_edges = _safe_size_histogram(sizes, bins=size_bins)
    else:
        orient_counts = np.zeros(0, dtype=np.int64)
        orient_edges = np.zeros(0, dtype=np.float64)
        size_counts = np.zeros(0, dtype=np.int64)
        size_edges = np.zeros(0, dtype=np.float64)
    size_counts_um2 = np.zeros_like(size_counts, dtype=np.int64)
    size_edges_um2 = np.zeros_like(size_edges, dtype=np.float64)

    orientation_stats = _scalar_summary(orientations)
    size_stats = _scalar_summary([float(v) for v in sizes])

    total_pixels = int(binary.size)
    foreground_pixels = int(np.count_nonzero(binary))
    area_fraction = float(foreground_pixels / total_pixels) if total_pixels else 0.0
    eq_diam_px = [float(2.0 * sqrt(max(v, 0.0) / pi)) for v in sizes]

    scalar_metrics: dict[str, float | int] = {
        "mask_area_fraction": area_fraction,
        "hydride_area_fraction": area_fraction,
        "hydride_area_fraction_percent": area_fraction * 100.0,
        "hydride_count": int(len(sizes)),
        "hydride_total_area_pixels": int(sum(sizes)),
        "size_mean_pixels": size_stats["mean"],
        "size_median_pixels": size_stats["median"],
        "size_min_pixels": size_stats["min"],
        "size_max_pixels": size_stats["max"],
        "orientation_mean_deg": orientation_stats["mean"],
        "orientation_median_deg": orientation_stats["median"],
        "orientation_min_deg": orientation_stats["min"],
        "orientation_max_deg": orientation_stats["max"],
        "excluded_small_features": int(excluded_small),
        "min_feature_pixels": int(min_feature_pixels),
    }
    if bool(include_extended_metrics):
        density_per_megapixel = float(len(sizes) / (total_pixels / 1_000_000.0)) if total_pixels else 0.0
        eq_px_stats = _scalar_summary(eq_diam_px)
        scalar_metrics.update(
            {
                "hydride_density_per_megapixel": density_per_megapixel,
                "size_std_pixels": size_stats["std"],
                "size_p10_pixels": size_stats["p10"],
                "size_p90_pixels": size_stats["p90"],
                "equivalent_diameter_mean_px": eq_px_stats["mean"],
                "equivalent_diameter_median_px": eq_px_stats["median"],
                "equivalent_diameter_std_px": eq_px_stats["std"],
                "equivalent_diameter_min_px": eq_px_stats["min"],
                "equivalent_diameter_max_px": eq_px_stats["max"],
                "orientation_std_deg": orientation_stats["std"],
                "orientation_p10_deg": orientation_stats["p10"],
                "orientation_p90_deg": orientation_stats["p90"],
                "orientation_alignment_index": _orientation_alignment_index(orientations),
                "orientation_entropy_bits": _orientation_entropy_bits(orient_counts),
            }
        )

    sizes_um2: list[float] = []
    eq_diam_um: list[float] = []
    um_per_px = float(microns_per_pixel) if microns_per_pixel is not None else None
    if bool(include_physical_metrics) and um_per_px is not None and um_per_px > 0:
        area_scale = um_per_px * um_per_px
        sizes_um2 = [float(v * area_scale) for v in sizes]
        eq_diam_um = [float(v * um_per_px) for v in eq_diam_px]
        size_um2_stats = _scalar_summary(sizes_um2)
        eq_um_stats = _scalar_summary(eq_diam_um)
        size_counts_um2, size_edges_um2 = _safe_histogram_float(sizes_um2, bins=size_bins)
        scalar_metrics.update(
            {
                "microns_per_pixel": um_per_px,
                "hydride_total_area_um2": float(sum(sizes_um2)),
                "size_mean_um2": size_um2_stats["mean"],
                "size_median_um2": size_um2_stats["median"],
                "size_std_um2": size_um2_stats["std"],
                "size_min_um2": size_um2_stats["min"],
                "size_max_um2": size_um2_stats["max"],
                "size_p10_um2": size_um2_stats["p10"],
                "size_p90_um2": size_um2_stats["p90"],
                "equivalent_diameter_mean_um": eq_um_stats["mean"],
                "equivalent_diameter_median_um": eq_um_stats["median"],
                "equivalent_diameter_std_um": eq_um_stats["std"],
                "equivalent_diameter_min_um": eq_um_stats["min"],
                "equivalent_diameter_max_um": eq_um_stats["max"],
            }
        )

    return HydrideStatisticsResult(
        scalar_metrics=scalar_metrics,
        orientations_deg=[float(v) for v in orientations],
        sizes_px=[int(v) for v in sizes],
        feature_label_ids=[int(v) for v in kept_label_ids],
        orientation_hist_counts=[int(v) for v in orient_counts.tolist()],
        orientation_hist_edges_deg=[float(v) for v in orient_edges.tolist()],
        size_hist_counts=[int(v) for v in size_counts.tolist()],
        size_hist_edges_px=[float(v) for v in size_edges.tolist()],
        size_hist_counts_um2=[int(v) for v in size_counts_um2.tolist()],
        size_hist_edges_um2=[float(v) for v in size_edges_um2.tolist()],
        sizes_um2=[float(v) for v in sizes_um2],
        equivalent_diameters_px=[float(v) for v in eq_diam_px],
        equivalent_diameters_um=[float(v) for v in eq_diam_um],
        microns_per_pixel=um_per_px if um_per_px is not None and um_per_px > 0 else None,
        label_map=label_map.astype(np.int32),
    )


def render_hydride_visualizations(
    stats: HydrideStatisticsResult,
    config: HydrideVisualizationConfig,
    *,
    include_distribution_charts: bool = True,
) -> dict[str, np.ndarray]:
    """Render analysis visualizations for hydride statistics.

    Parameters
    ----------
    stats:
        Statistics payload from :func:`compute_hydride_statistics`.
    config:
        Rendering and histogram controls.

    Returns
    -------
    dict[str, np.ndarray]
        RGB arrays for orientation map and distribution charts.
    """

    cmap = plt.get_cmap(config.orientation_cmap)
    label_map = np.asarray(stats.label_map)
    orientation_map = np.zeros((*label_map.shape, 3), dtype=np.float32)
    for label_id, angle in zip(stats.feature_label_ids, stats.orientations_deg):
        orientation_map[label_map == int(label_id)] = cmap(float(angle) / 90.0)[:3]

    fig_map, ax_map = plt.subplots(figsize=(5.5, 5.0))
    ax_map.imshow(orientation_map)
    ax_map.set_title("Hydride Orientation Map")
    ax_map.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
    sm.set_array([])
    fig_map.colorbar(sm, ax=ax_map, fraction=0.046, pad=0.04, label="Orientation (deg)")
    map_rgb = _figure_to_rgb(fig_map)

    if not bool(include_distribution_charts):
        return {
            "orientation_map_rgb": map_rgb,
        }

    fig_size, ax_size = plt.subplots(figsize=(5.5, 4.4))
    size_values = [float(v) for v in stats.sizes_px]
    size_label = "Feature Size (pixels)"
    if stats.microns_per_pixel is not None and stats.sizes_um2:
        size_values = [float(v) for v in stats.sizes_um2]
        size_label = "Feature Area (um^2)"
    if size_values:
        if str(config.size_scale).lower() == "log":
            min_size = max(min(size_values), 1e-9)
            max_size = max(max(size_values), min_size)
            if max_size > min_size:
                bins = np.logspace(
                    np.log10(min_size),
                    np.log10(max_size),
                    num=max(2, int(config.size_bins) + 1),
                )
            else:
                bins = np.linspace(min_size, min_size + 1.0, num=max(2, int(config.size_bins) + 1))
            ax_size.hist(size_values, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
            ax_size.set_xscale("log")
        else:
            ax_size.hist(size_values, bins=max(1, int(config.size_bins)), color="#1f77b4", edgecolor="black", alpha=0.85)
    ax_size.set_xlabel(size_label)
    ax_size.set_ylabel("Count")
    ax_size.set_title("Hydride Size Distribution")
    size_rgb = _figure_to_rgb(fig_size)

    fig_orient, ax_orient = plt.subplots(figsize=(5.5, 4.4))
    if stats.orientations_deg:
        ax_orient.hist(
            stats.orientations_deg,
            bins=max(1, int(config.orientation_bins)),
            range=(0.0, 90.0),
            color="#ff8c00",
            edgecolor="black",
            alpha=0.85,
        )
    ax_orient.set_xlabel("Orientation (degrees)")
    ax_orient.set_ylabel("Count")
    ax_orient.set_title("Hydride Orientation Distribution")
    orient_rgb = _figure_to_rgb(fig_orient)

    return {
        "orientation_map_rgb": map_rgb,
        "size_distribution_rgb": size_rgb,
        "orientation_distribution_rgb": orient_rgb,
    }


def statistics_to_json(stats: HydrideStatisticsResult) -> dict[str, object]:
    """Convert statistics payload into a JSON-friendly dictionary.

    Parameters
    ----------
    stats:
        Hydride statistics result.

    Returns
    -------
    dict[str, object]
        JSON-serializable dictionary for report emission.
    """

    return {
        "scalar_metrics": dict(stats.scalar_metrics),
        "orientations_deg": [float(v) for v in stats.orientations_deg],
        "sizes_px": [int(v) for v in stats.sizes_px],
        "sizes_um2": [float(v) for v in stats.sizes_um2],
        "equivalent_diameters_px": [float(v) for v in stats.equivalent_diameters_px],
        "equivalent_diameters_um": [float(v) for v in stats.equivalent_diameters_um],
        "microns_per_pixel": None if stats.microns_per_pixel is None else float(stats.microns_per_pixel),
        "orientation_histogram": {
            "counts": [int(v) for v in stats.orientation_hist_counts],
            "bin_edges_deg": [float(v) for v in stats.orientation_hist_edges_deg],
        },
        "size_histogram": {
            "counts": [int(v) for v in stats.size_hist_counts],
            "bin_edges_px": [float(v) for v in stats.size_hist_edges_px],
        },
        "size_histogram_um2": {
            "counts": [int(v) for v in stats.size_hist_counts_um2],
            "bin_edges_um2": [float(v) for v in stats.size_hist_edges_um2],
        },
    }
