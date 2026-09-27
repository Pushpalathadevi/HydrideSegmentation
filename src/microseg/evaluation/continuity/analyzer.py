"""Hydride Connectivity Index (HCI) analysis: one deterministic core for every interface.

``analyze_continuity`` is a pure function of the mask, the configuration and
the calibration. CLI, desktop and web adapters call it unchanged, so identical
inputs give identical results.
"""

from __future__ import annotations

import hashlib
import platform
import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from .config import DIRECTIONS, FORMULATION_ID, Calibration, ContinuityAnalysisConfig
from .contracts import ContinuityAnalysisResult, ContinuityClusterResult
from .linkage import _feret, _hull_vertices, compute_linkage
from .path import path_continuity
from .skeleton_graph import skeleton_topology

ProgressHook = Callable[[str, float], None]


def binarize(mask: np.ndarray, foreground: tuple[int, ...] = (1,)) -> tuple[np.ndarray, list[str]]:
    """Convert a boolean, indexed or RGB mask to a boolean hydride mask.

    Returns the mask and any warnings about how it was interpreted.
    """

    warnings: list[str] = []
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr.copy(), warnings
    if arr.ndim == 3:
        warnings.append("colour mask: any non-black pixel treated as hydride")
        return np.any(arr > 0, axis=2), warnings
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (or RGB); got shape {arr.shape}")
    values = set(np.unique(arr).tolist())
    fg = set(int(v) for v in foreground)
    if values & fg:
        return np.isin(arr, list(fg)), warnings
    nonzero = values - {0}
    if len(nonzero) == 1:
        v = nonzero.pop()
        warnings.append(f"no pixel has a foreground class index {sorted(fg)}; the single non-zero value {v} is treated as hydride")
        return arr == v, warnings
    return np.zeros(arr.shape, bool), warnings


def clean_mask(mask: np.ndarray, min_area_px: float, max_hole_px: float) -> np.ndarray:
    """Remove 8-connected components below ``min_area_px`` and fill interior holes below ``max_hole_px``."""

    from scipy import ndimage as ndi
    from skimage.measure import label

    lab = label(mask, connectivity=2)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    out = keep[lab]
    if max_hole_px > 0:
        hlab, _ = ndi.label(~out)
        hsizes = np.bincount(hlab.ravel())
        border = np.unique(np.concatenate([hlab[0], hlab[-1], hlab[:, 0], hlab[:, -1]]))
        small = hsizes < max_hole_px
        small[border] = False
        small[0] = False
        out = out | small[hlab]
    return out


def _code_version() -> str:
    try:
        from src.microseg.version import __version__
    except Exception:  # pragma: no cover - alternative import roots
        try:
            from microseg.version import __version__  # type: ignore
        except Exception:
            __version__ = "unknown"
    return str(__version__)


def analyze_continuity(
    mask: np.ndarray,
    config: ContinuityAnalysisConfig | None = None,
    calibration: Calibration | None = None,
    *,
    mask_source: str = "predicted",
    progress: ProgressHook | None = None,
) -> ContinuityAnalysisResult:
    """Compute the Hydride Connectivity Index and its companion descriptors.

    Parameters
    ----------
    mask:
        Boolean, indexed (class indices) or RGB segmentation mask. Image rows
        are the radial direction and columns the circumferential direction.
    config:
        Analysis parameters; defaults to ``ContinuityAnalysisConfig()``.
    calibration:
        Pixel size. Without it every length is reported in pixels.
    mask_source:
        ``"predicted"`` or ``"corrected"``, recorded in the provenance.
    progress:
        Optional callback ``(stage, fraction_complete)``.

    Returns
    -------
    ContinuityAnalysisResult
        ``status`` is ``"ok"``, or ``"not_applicable"`` when no hydride
        survives clean-up. Numeric fields are then ``None``, never 0.
    """

    cfg = config or ContinuityAnalysisConfig()
    cal = calibration or Calibration()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def stage(name: str, frac: float) -> None:
        timings[name] = round(time.perf_counter() - t0, 4)
        if progress:
            progress(name, frac)

    flags: list[str] = []
    raw, warnings = binarize(mask, cfg.foreground_class_indices)
    h, w = raw.shape
    if h * w > cfg.max_pixels:
        raise ValueError(f"mask has {h * w} pixels, above max_pixels={cfg.max_pixels}")
    a = cal.scale
    unit = cal.unit
    min_px = cfg.min_feature_area_um2 / a**2 if cal.calibrated else cfg.min_feature_area_px
    hole_px = cfg.max_hole_area_um2 / a**2 if cal.calibrated else cfg.max_hole_area_px
    # Resolution floor: a feature or hole smaller than 2 × 2 pixels cannot be resolved,
    # so coarse images never keep single-pixel specks or pinholes.
    floor = float(cfg.resolution_floor_px)
    if cal.calibrated and (min_px < floor or hole_px < floor):
        flags.append("cleaning_thresholds_raised_to_resolution_floor")
    min_px, hole_px = max(min_px, floor), max(hole_px, floor)
    cleaned = clean_mask(raw, min_px, hole_px)
    stage("clean", 0.1)

    if not cal.calibrated:
        flags.append("uncalibrated_lengths_in_pixels")
    af_raw = float(raw.mean()) if raw.size else 0.0
    af_clean = float(cleaned.mean()) if cleaned.size else 0.0
    if raw.any() and (raw.sum() - np.logical_and(raw, cleaned).sum()) > 0.1 * raw.sum():
        flags.append("cleaning_removed_over_10_percent_of_area")

    parameters: dict[str, Any] = cfg.to_dict()
    parameters.update({"min_feature_area_px_effective": float(min_px), "max_hole_area_px_effective": float(hole_px), "connectivity": 8})
    provenance = {
        "code_version": _code_version(),
        "formulation_id": FORMULATION_ID,
        "input_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        "input_shape": [int(h), int(w)],
        "mask_source": mask_source,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    calibration_payload = {"pixel_size_um": cal.pixel_size_um, "source": cal.source, "unit": unit, **cal.extra}

    if not cleaned.any():
        provenance["runtime_s"] = round(time.perf_counter() - t0, 4)
        return ContinuityAnalysisResult(
            status="not_applicable",
            status_reason="no hydride component survives clean-up" if raw.any() else "mask contains no hydride",
            unit=unit,
            specimen={"HCI": {d: None for d in DIRECTIONS}, "area_fraction_raw": af_raw, "area_fraction_cleaned": af_clean},
            curve={},
            clusters=[],
            parameters=parameters,
            calibration=calibration_payload,
            provenance=provenance,
            quality_flags=flags,
            warnings=warnings,
            arrays={"raw": raw, "cleaned": cleaned},
        )

    ref_px: dict[str, float] = {}
    if cfg.reference_length_radial_um:
        ref_px["radial"] = cfg.reference_length_radial_um / a
    if cfg.reference_length_circumferential_um:
        ref_px["circumferential"] = cfg.reference_length_circumferential_um / a
    link = compute_linkage(cleaned, ref_px or None)
    comp = link.components
    stage("linkage", 0.45)

    # δ_max: explicit, or auto_bridge_fraction (default 1/5) of a reference hydride length:
    # the smallest accepted hydride (default) or a percentile of the accepted lengths.
    lengths = comp.feret_px[1:]
    smallest_len_px = float(lengths.min())
    if cfg.auto_size_statistic == "percentile":
        ref_len_px = float(np.percentile(lengths, cfg.auto_size_percentile))
    else:
        ref_len_px = smallest_len_px
    if cfg.max_bridge_distance == "auto":
        dmax_px = max(1.0, cfg.auto_bridge_fraction * ref_len_px)
        dmax_source = f"auto:{cfg.auto_size_statistic}"
        if cfg.auto_bridge_fraction * ref_len_px < 1.0:
            flags.append("auto_max_bridge_distance_floored_to_one_pixel")
    else:
        dmax_px = float(cfg.max_bridge_distance) / a
        dmax_source = "user"
    d0_px = dmax_px if cfg.report_distance == "auto" else float(cfg.report_distance) / a

    hci = {d: link.integral_mean(d, dmax_px) for d in DIRECTIONS}
    c_at_0 = {d: link.value_at(d, d0_px) for d in DIRECTIONS}
    half = {d: link.delta_half(d) for d in DIRECTIONS}

    # Cluster table at δ0.
    roots = link.roots_at(d0_px)
    groups: dict[int, list[int]] = {}
    for i in range(1, comp.n + 1):
        groups.setdefault(int(roots[i]), []).append(i)
    bridges_by_root: dict[int, list[dict[str, Any]]] = {}
    for ev in link.events:
        if ev.gap_px > d0_px:
            break
        e = ev.edge_index
        bridges_by_root.setdefault(int(roots[ev.a]), []).append(
            {
                "a": ev.a,
                "b": ev.b,
                "gap": ev.gap_px * a,
                "p_rc": link.edges.p_rc[e].tolist(),
                "q_rc": link.edges.q_rc[e].tolist(),
            }
        )
    topo = skeleton_topology(cleaned, comp.labels) if cfg.topology_enabled else None
    stage("topology", 0.7)

    clusters: list[ContinuityClusterResult] = []
    xi_num = {d: 0.0 for d in DIRECTIONS}
    ref = link.reference_px
    order = sorted(groups.values(), key=lambda m: -comp.area_px[m].sum())
    for idx, members in enumerate(order, 1):
        area_px = float(comp.area_px[members].sum())
        cov_r = len(np.unique(np.concatenate([comp.rows[m] for m in members])))
        cov_c = len(np.unique(np.concatenate([comp.cols[m] for m in members])))
        hull = _hull_vertices(np.vstack([comp.hulls[m] for m in members]))
        fer = _feret(hull)
        weight = {
            "radial": min(1.0, cov_r / ref["radial"]),
            "circumferential": min(1.0, cov_c / ref["circumferential"]),
            "isotropic": min(1.0, fer / ref["isotropic"]),
        }
        xi_num["radial"] += area_px * cov_r
        xi_num["circumferential"] += area_px * cov_c
        xi_num["isotropic"] += area_px * fer
        tsum = {"endpoints": 0, "junctions": 0, "branching_excess": 0}
        if topo is not None:
            for m in members:
                for k, v in topo.per_component.get(m, {}).items():
                    tsum[k] += v
        clusters.append(
            ContinuityClusterResult(
                cluster_id=f"K{idx:04d}",
                members=[int(m) for m in members],
                area=area_px * a * a,
                coverage_radial=cov_r * a,
                coverage_circumferential=cov_c * a,
                feret=fer * a,
                weight=weight,
                touches_border=bool(comp.touches_border[members].any()),
                bridges=bridges_by_root.get(int(roots[members[0]]), []),
                topology=tsum,
            )
        )
    xi = {d: xi_num[d] / link.total_area_px * a for d in DIRECTIONS}

    if any(c.touches_border for c in clusters[:1]):
        flags.append("largest_cluster_touches_border")
    if any(max(c.weight.values()) >= 1.0 for c in clusters):
        flags.append("cluster_extent_capped_at_reference_length")
    if len(clusters) < 5:
        flags.append("fewer_than_5_clusters")
    dt_med = None
    if cleaned.any():
        from scipy.ndimage import distance_transform_edt

        dt = distance_transform_edt(cleaned)
        if topo is not None and topo.skeleton.any():
            dt_med = float(np.median(dt[topo.skeleton]) * 2.0)
            if dt_med < 3.0:
                flags.append("hydride_thickness_below_3_px_topology_unreliable")

    path_payload: dict[str, Any] = {}
    best_paths: dict[str, np.ndarray] = {}
    if cfg.path_enabled:
        for d in ("radial", "circumferential"):
            pc = path_continuity(cleaned, d, ref[d], cfg.path_hydride_cost)
            path_payload[d] = pc.mean
            path_payload[f"{d}_best"] = pc.best
            path_payload[f"{d}_best_ligament"] = pc.ligament_px * a
            best_paths[d] = pc.best_path_rc
    stage("path", 0.9)

    grid = np.linspace(0.0, dmax_px, int(cfg.curve_points))
    events_payload = [{"delta": ev.gap_px * a, "a": ev.a, "b": ev.b} for ev in link.events if ev.gap_px <= dmax_px]
    curve = {
        "delta": (grid * a).tolist(),
        **{f"C_{d}": [link.value_at(d, g) for g in grid] for d in DIRECTIONS},
        "n_clusters": [link.clusters_at_count(g) for g in grid],
        "step_breaks": (link.breaks_px * a).tolist(),
        **{f"step_C_{d}": link.values[d].tolist() for d in DIRECTIONS},
        "merge_events_within_max_bridge_distance": events_payload,
    }

    area_unit = a * a
    field_area = h * w * area_unit
    specimen: dict[str, Any] = {
        "HCI": hci,
        "connectivity_at_report_distance": c_at_0,
        "connectivity_length": xi,
        "critical_linking_distance": {d: (None if half[d] is None else half[d] * a) for d in DIRECTIONS},
        "critical_linking_distance_reached": {d: half[d] is not None for d in DIRECTIONS},
        "max_bridge_distance": dmax_px * a,
        "max_bridge_distance_source": dmax_source,
        "report_distance": d0_px * a,
        "smallest_hydride_length": smallest_len_px * a,
        "auto_reference_hydride_length": ref_len_px * a,
        "reference_length": {d: ref[d] * a for d in DIRECTIONS},
        "path_continuity": path_payload,
        "area_fraction_raw": af_raw,
        "area_fraction_cleaned": af_clean,
        "n_components": int(comp.n),
        "n_clusters_at_report_distance": len(clusters),
        "median_hydride_thickness": None if dt_med is None else dt_med * a,
    }
    if topo is not None:
        # field_area is in µm² (calibrated) or px²; ×1e-6 converts to mm² or megapixels.
        per_area = 1.0 / (field_area * 1e-6)
        specimen["topology"] = {
            "endpoints": topo.n_endpoints,
            "junctions": topo.n_junctions,
            "junction_degree_histogram": topo.junction_degree_histogram,
            "branching_excess": topo.branching_excess,
            "loops": topo.n_loops,
            "skeleton_components": topo.n_components,
            "network_closure": topo.closure,
            "identity_residual": topo.identity_residual,
            "skeleton_length": topo.skeleton_length_px * a,
            "density_unit": "per_mm2" if cal.calibrated else "per_megapixel",
            "endpoints_density": topo.n_endpoints * per_area,
            "junctions_density": topo.n_junctions * per_area,
            "loops_density": topo.n_loops * per_area,
        }
        if topo.identity_residual != 0:
            flags.append("topology_identity_residual_nonzero")
    provenance["runtime_s"] = round(time.perf_counter() - t0, 4)
    provenance["stage_times_s"] = timings
    stage("done", 1.0)
    return ContinuityAnalysisResult(
        status="ok",
        status_reason=None,
        unit=unit,
        specimen=specimen,
        curve=curve,
        clusters=clusters,
        parameters=parameters,
        calibration=calibration_payload,
        provenance=provenance,
        quality_flags=flags,
        warnings=warnings,
        arrays={
            "raw": raw,
            "cleaned": cleaned,
            "labels": comp.labels,
            "roots": roots,
            "linkage": link,
            "topology": topo,
            "best_paths": best_paths,
        },
    )
