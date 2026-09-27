"""Research harness: score candidate HCI formulations on the synthetic benchmark.

Report keys versus specification symbols: ``IBAR_R``/``IBAR_iso`` are the v1 HCI
(integral of the connectivity function over 0..delta_max); ``HCI_R``/``HCI_C``/``HCI_iso``
are the connectivity function C_u at delta0; ``HCI_R_span`` is the rejected
bounding-span variant; ``PI_*`` is path continuity; ``topo_*`` is skeleton topology.

This is a *study* script that produces the evidence tables in
``docs/hci_specification.md``. It is not the production HCI implementation,
which is specified there and remains to be built in ``src/microseg/evaluation/continuity``.

Candidates
----------
prototype_v0
    The intern prototype executed unchanged (``HARSHAD_WORK/HCI_2_4_works_1 (1).py``)
    in an isolated subprocess: both the workbook value and ``compute_hci()``.
linkage (HCI_R, HCI_C, HCI_iso)
    Area-weighted mean normalised extent of delta-linked hydride clusters.
path continuity (PI_R, PI_C)
    Area-weighted mean of (1 - minimum matrix-ligament fraction) of the best
    field-crossing path through each hydride pixel.
topology
    Skeleton-graph endpoints, junctions, loops and network closure.

Usage::

    python scripts/hci_candidate_study.py --benchmark test_data/hci_synthetic_v1 \
        --output artifacts/hci_candidate_study
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROTOTYPE_DEFAULT = "HARSHAD_WORK/HCI_2_4_works_1 (1).py"
PIXEL_UM_DEFAULT = 0.5
MIN_AREA_UM2 = 5.0  # 20 px at 0.5 um/px, the prototype value
MAX_HOLE_UM2 = 5.0
DELTA0_UM = 2.5  # 5 px at 0.5 um/px, the prototype association distance
DELTA_GRID_UM = [round(0.25 * k, 2) for k in range(0, 41)] + [12.5, 15.0, 20.0]
DELTA_MAX_UM = 10.0


# ---------------------------------------------------------------------------
# Common preprocessing
# ---------------------------------------------------------------------------


def clean(mask: np.ndarray, pixel_um: float) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.measure import label

    min_px = MIN_AREA_UM2 / pixel_um**2
    hole_px = MAX_HOLE_UM2 / pixel_um**2
    lab = label(mask, connectivity=2)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    out = keep[lab]
    holes = ~out
    hlab, _ = ndi.label(holes)
    hsizes = np.bincount(hlab.ravel())
    border = np.unique(np.concatenate([hlab[0], hlab[-1], hlab[:, 0], hlab[:, -1]]))
    small = hsizes < hole_px
    small[border] = False
    small[0] = False
    return out | small[hlab]


# ---------------------------------------------------------------------------
# Linkage (cluster-extent) index
# ---------------------------------------------------------------------------


def component_gap_edges(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum surface gap (pixels) between Voronoi-neighbouring components.

    Single-linkage clustering only needs minimum-spanning-tree edges, which are
    always between Voronoi neighbours, so this sparse edge set is sufficient.
    """

    from scipy.ndimage import distance_transform_edt

    if labels.max() < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    near = labels[iy, ix]
    a_list, b_list, d_list = [], [], []
    h, w = labels.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys0, ys1 = slice(0, h - dy), slice(dy, h)
        xs0, xs1 = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        la, lb = near[ys0, xs0], near[ys1, xs1]
        diff = la != lb
        if not diff.any():
            continue
        d = np.hypot(iy[ys0, xs0][diff] - iy[ys1, xs1][diff], ix[ys0, xs0][diff] - ix[ys1, xs1][diff]) - 1.0
        a, b = la[diff], lb[diff]
        a_list.append(np.minimum(a, b))
        b_list.append(np.maximum(a, b))
        d_list.append(d)
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    d = np.concatenate(d_list)
    key = a.astype(np.int64) * (labels.max() + 1) + b
    order = np.lexsort((d, key))
    key, a, b, d = key[order], a[order], b[order], d[order]
    first = np.r_[True, key[1:] != key[:-1]]
    return a[first], b[first], np.maximum(d[first], 0.0)


def linkage_curve(mask: np.ndarray, deltas_px: list[float]) -> dict[str, list[float]]:
    """HCI_R, HCI_C and HCI_iso for each association distance."""

    from scipy.spatial import ConvexHull, QhullError
    from skimage.measure import label, regionprops

    h, w = mask.shape
    lab = label(mask, connectivity=2)
    n = int(lab.max())
    out = {"R": [], "C": [], "iso": [], "R_span": [], "n_clusters": []}
    if n == 0:
        for key in out:
            out[key] = [float("nan")] * len(deltas_px)
        return out
    props = regionprops(lab)
    rows_of: list[np.ndarray] = [np.zeros(0, int)] * (n + 1)
    cols_of: list[np.ndarray] = [np.zeros(0, int)] * (n + 1)
    for p in props:
        rows_of[p.label] = np.unique(p.coords[:, 0])
        cols_of[p.label] = np.unique(p.coords[:, 1])
    area = np.zeros(n + 1)
    bbox = np.zeros((n + 1, 4))
    hulls: list[np.ndarray] = [np.zeros((0, 2))] * (n + 1)
    for p in props:
        area[p.label] = p.area
        r0, c0, r1, c1 = p.bbox
        bbox[p.label] = (r0, c0, r1 - 1, c1 - 1)
        pts = p.coords.astype(float)
        try:
            hulls[p.label] = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
        except QhullError:
            hulls[p.label] = pts
    ea, eb, ed = component_gap_edges(lab)
    order = np.argsort(ed)
    ea, eb, ed = ea[order], eb[order], ed[order]
    ref_iso = float(min(h, w))
    for delta in deltas_px:
        parent = list(range(n + 1))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b, d in zip(ea, eb, ed):
            if d > delta:
                break
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        groups: dict[int, list[int]] = {}
        for i in range(1, n + 1):
            groups.setdefault(find(i), []).append(i)
        num_r = num_c = num_i = num_rs = 0.0
        for members in groups.values():
            bb = bbox[members]
            a_sum = area[members].sum()
            # Projected hydride coverage: bridged gaps add no extent.
            cov_r = len(np.unique(np.concatenate([rows_of[m] for m in members])))
            cov_c = len(np.unique(np.concatenate([cols_of[m] for m in members])))
            span_r = bb[:, 2].max() - bb[:, 0].min() + 1
            num_rs += a_sum * min(1.0, span_r / h)
            pts = np.vstack([hulls[m] for m in members])
            if len(pts) > 400:
                try:
                    pts = pts[ConvexHull(pts).vertices]
                except QhullError:
                    pass
            diff = pts[:, None, :] - pts[None, :, :]
            feret = math.sqrt(float((diff**2).sum(-1).max())) + 1.0
            num_r += a_sum * min(1.0, cov_r / h)
            num_c += a_sum * min(1.0, cov_c / w)
            num_i += a_sum * min(1.0, feret / ref_iso)
        tot = area[1:].sum()
        out["R"].append(num_r / tot)
        out["C"].append(num_c / tot)
        out["iso"].append(num_i / tot)
        out["R_span"].append(num_rs / tot)
        out["n_clusters"].append(float(len(groups)))
    return out


# ---------------------------------------------------------------------------
# Path continuity (minimum matrix ligament)
# ---------------------------------------------------------------------------


def path_continuity(mask: np.ndarray) -> dict[str, float]:
    from skimage.graph import MCP_Geometric

    if not mask.any():
        return {"PI_R": float("nan"), "PI_C": float("nan"), "PI_R_best": float("nan"), "PI_C_best": float("nan")}
    cost = (~mask).astype(float)
    h, w = mask.shape
    res = {}
    for name, starts_a, starts_b, span in (
        ("R", [(0, j) for j in range(w)], [(h - 1, j) for j in range(w)], h - 1),
        ("C", [(i, 0) for i in range(h)], [(i, w - 1) for i in range(h)], w - 1),
    ):
        da, _ = MCP_Geometric(cost).find_costs(starts_a)
        db, _ = MCP_Geometric(cost).find_costs(starts_b)
        lig = da + db
        weights = np.clip(1.0 - lig[mask] / span, 0.0, 1.0)
        res[f"PI_{name}"] = float(weights.mean())
        res[f"PI_{name}_best"] = float(np.clip(1.0 - lig.min() / span, 0.0, 1.0))
    return res


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------


def skeleton_topology(mask: np.ndarray) -> dict[str, float]:
    """Endpoints, collapsed junctions, loops and closure from the skeleton graph."""

    from scipy import ndimage as ndi
    from skimage.measure import label
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    if not skel.any():
        return {"endpoints": 0, "junctions": 0, "loops": 0, "branching_excess": 0, "closure": float("nan")}
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    deg = ndi.convolve(skel.astype(int), k, mode="constant") * skel
    endpoints = int(((deg == 1) & skel).sum())
    jmask = (deg >= 3) & skel
    jlab, n_j = ndi.label(jmask, structure=np.ones((3, 3)))
    chains = skel & ~jmask
    clab, n_c = ndi.label(chains, structure=np.ones((3, 3)))
    # Degree of each collapsed junction = number of chain pieces touching it.
    jdeg = np.zeros(n_j + 1, int)
    if n_j:
        grown = ndi.grey_dilation(jlab, footprint=np.ones((3, 3)))
        pairs = set(zip(clab[(clab > 0) & (grown > 0)], grown[(clab > 0) & (grown > 0)]))
        for _, j in pairs:
            jdeg[j] += 1
        # Adjacent junction clusters are merged by the 8-connected labelling.
    junction_degrees = jdeg[1:][jdeg[1:] >= 3]
    comps = int(label(skel, connectivity=2).max())
    # Euler characteristic of the skeleton (8-connectivity) gives loops exactly.
    from skimage.measure import euler_number

    chi = int(euler_number(skel, connectivity=2))
    loops = comps - chi
    beta = int((junction_degrees - 2).sum())
    denom = beta + 2 * comps
    return {
        "endpoints": endpoints,
        "junctions": int(len(junction_degrees)),
        "loops": int(loops),
        "branching_excess": beta,
        "closure": float(2.0 * loops / denom) if denom else float("nan"),
        "skeleton_components": comps,
    }


# ---------------------------------------------------------------------------
# Prototype baseline (isolated subprocess)
# ---------------------------------------------------------------------------


def prototype_worker(proto: Path, mask_path: Path) -> dict[str, Any]:
    import os

    import pandas as pd
    from PIL import Image

    src = proto.read_text(encoding="utf-8")
    tmp = tempfile.mkdtemp(prefix="hci_proto_")
    src = src.replace('INPUT_DIR = "', 'INPUT_DIR = r"' + tmp + '" or "').replace('OUTPUT_DIR = "', 'OUTPUT_DIR = r"' + tmp + '" or "')
    ns: dict[str, Any] = {"__name__": "hci_prototype"}
    exec(compile(src, str(proto), "exec"), ns)
    mask = np.array(Image.open(mask_path)) > 0
    img = np.dstack([mask.astype(np.uint8) * 255] * 3)
    binary = ns["preprocess_mask"](ns["rgb_to_binary"](img))
    _, props = ns["identify_hydrides"](binary)
    if len(props) == 0:
        return {"status": "empty", "workbook_hci": None, "compute_hci": None}
    _, connectors = ns["associate_hydrides"](props)
    analysis = ns["create_analysis_mask"](img.shape, props, connectors)
    skeleton = ns["skeletonize_clusters"](analysis)
    ns["save_cluster_summary"](skeleton, binary, tmp)
    sheet = pd.read_excel(os.path.join(tmp, "cluster_summary.xlsx"), sheet_name="Calculated_Values", header=None)
    workbook = None
    for _, row in sheet.iterrows():
        if str(row.iloc[0]).strip() == "HCI":
            workbook = float(row.iloc[1])
    rec = pd.read_excel(os.path.join(tmp, "cluster_summary.xlsx"), sheet_name="Recorded_Values")
    return {
        "status": "ok",
        "workbook_hci": workbook,
        "compute_hci": float(ns["compute_hci"](skeleton)),
        "n_clusters": int(len(rec)),
        "mean_primary_length_px": float(rec["Primary_Length_px"].mean()) if len(rec) else None,
        "sum_junctions": int(rec["Junctions"].sum()) if len(rec) else 0,
    }


def run_prototype(proto: Path, mask_path: Path, timeout: float) -> dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--prototype-worker", str(mask_path), "--prototype", str(proto)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": timeout}
    elapsed = time.monotonic() - start
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"status": "error", "stderr": proc.stderr[-400:], "seconds": elapsed}
    payload = json.loads(lines[-1])
    payload["seconds"] = round(elapsed, 2)
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate_sample(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    from PIL import Image

    mask = np.array(Image.open(root / entry["mask_path"])) > 0
    pixel_um = float(entry.get("parameters", {}).get("pixel_size_um", entry.get("pixel_size_um", PIXEL_UM_DEFAULT)))
    cleaned = clean(mask, pixel_um)
    deltas_px = [d / pixel_um for d in DELTA_GRID_UM]
    curve = linkage_curve(cleaned, deltas_px)
    i0 = DELTA_GRID_UM.index(DELTA0_UM)
    res: dict[str, Any] = {
        "sample_id": entry["sample_id"],
        "family": entry["family"],
        "condition": entry["condition"],
        "replicate": entry["replicate"],
        "area_fraction": entry["area_fraction"],
        "cleaned_area_fraction": float(cleaned.mean()),
        "pixel_size_um": pixel_um,
        "HCI_R": curve["R"][i0],
        "HCI_C": curve["C"][i0],
        "HCI_iso": curve["iso"][i0],
        "HCI_R_delta0": curve["R"][0],
        "clusters_at_delta0": curve["n_clusters"][i0],
        "curve_delta_um": DELTA_GRID_UM,
        "curve_HCI_R": curve["R"],
        "curve_HCI_iso": curve["iso"],
    }
    if np.isfinite(curve["R"][0]):
        grid = np.array(DELTA_GRID_UM)
        sel = grid <= DELTA_MAX_UM
        for key, name in (("R", "IBAR_R"), ("C", "IBAR_C"), ("iso", "IBAR_iso")):
            # The curve is a right-continuous step function; integrate it exactly on the grid.
            vals = np.array(curve[key])[sel]
            res[name] = float(np.sum(vals[:-1] * np.diff(grid[sel])) / DELTA_MAX_UM)
        half = [d for d, v in zip(DELTA_GRID_UM, curve["R"]) if v >= 0.5]
        res["delta_half_R_um"] = half[0] if half else None
        res["HCI_R_span"] = curve["R_span"][i0]
    else:
        res.update({"IBAR_R": float("nan"), "IBAR_C": float("nan"), "IBAR_iso": float("nan"), "delta_half_R_um": None, "HCI_R_span": float("nan")})
    res.update(path_continuity(cleaned))
    res.update({f"topo_{k}": v for k, v in skeleton_topology(cleaned).items()})
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="test_data/hci_synthetic_v1")
    parser.add_argument("--output", default="artifacts/hci_candidate_study")
    parser.add_argument("--prototype", default=PROTOTYPE_DEFAULT)
    parser.add_argument("--skip-prototype", action="store_true")
    parser.add_argument("--reuse-prototype", default=None, help="Copy prototype results from an earlier report.json")
    parser.add_argument("--prototype-timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prototype-worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.prototype_worker:
        print(json.dumps(prototype_worker(Path(args.prototype), Path(args.prototype_worker))))
        return 0

    root = Path(args.benchmark)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["samples"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    results = []
    for i, e in enumerate(entries, 1):
        results.append(evaluate_sample(e, root))
        if i % 20 == 0 or i == len(entries):
            print(f"[candidates] {i}/{len(entries)} ({100 * i / len(entries):.0f}%) {time.monotonic() - t0:.0f}s", flush=True)
    proto_meta = None
    if args.reuse_prototype:
        previous = json.loads(Path(args.reuse_prototype).read_text(encoding="utf-8"))
        by_id = {r["sample_id"]: r.get("prototype") for r in previous["results"]}
        for r in results:
            r["prototype"] = by_id.get(r["sample_id"])
        proto_meta = previous.get("prototype")
    elif not args.skip_prototype:
        proto = Path(args.prototype)
        proto_meta = {"path": proto.as_posix(), "sha256": hashlib.sha256(proto.read_bytes()).hexdigest()}
        done = 0

        def job(e: dict[str, Any]) -> dict[str, Any]:
            return run_prototype(proto, root / e["mask_path"], args.prototype_timeout)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for r, res in zip(results, pool.map(job, entries)):
                r["prototype"] = res
                done += 1
                if done % 20 == 0 or done == len(entries):
                    print(f"[prototype] {done}/{len(entries)} {time.monotonic() - t0:.0f}s", flush=True)
    report = {
        "schema_version": "microseg.hci_candidate_study.v1",
        "benchmark_schema": manifest["schema_version"],
        "benchmark_code_version": manifest.get("code_version"),
        "parameters": {
            "min_area_um2": MIN_AREA_UM2,
            "max_hole_um2": MAX_HOLE_UM2,
            "delta0_um": DELTA0_UM,
            "delta_grid_um": DELTA_GRID_UM,
            "delta_max_um": DELTA_MAX_UM,
            "directional_extent": "projected hydride coverage (distinct rows/columns of member pixels)",
            "connectivity": 8,
        },
        "prototype": proto_meta,
        "results": results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"report: {out / 'report.json'}")
    summarize(report, out)
    print(f"summary: {out / 'summary.json'}")
    return 0




# ---------------------------------------------------------------------------
# Summary: condition tables, axiom checks, evidence figure
# ---------------------------------------------------------------------------

METRICS = {
    "proto_workbook": "Prototype HCI (workbook)",
    "proto_compute": "Prototype HCI (compute_hci)",
    "HCI_R": "HCI_R(δ0)",
    "HCI_iso": "HCI_iso(δ0)",
    "HCI_R_delta0": "HCI_R(0)",
    "HCI_R_span": "HCI_R span variant",
    "IBAR_R": "Ī_R",
    "IBAR_iso": "Ī_iso",
    "PI_R": "Π_R",
    "PI_R_best": "Π_R best",
    "topo_closure": "κ (skeleton)",
}

#: Absolute tolerance for "non-decreasing" checks (declared in the specification).
MONOTONE_TOLERANCE = 0.01

ORDERED_FAMILIES = {
    "gap_sweep": ("increasing", ["HCI_R", "HCI_R_span", "IBAR_R", "HCI_iso", "PI_R", "proto_workbook"]),
    "fragmentation": ("decreasing", ["HCI_R", "IBAR_R", "HCI_iso", "PI_R", "proto_workbook"]),
    "arrangement": ("increasing", ["HCI_R", "IBAR_R", "PI_R", "proto_workbook"]),
    "orientation": ("increasing", ["HCI_R", "IBAR_R", "PI_R", "proto_workbook"]),
}


def _metric(r: dict[str, Any], key: str) -> float:
    if key.startswith("proto_"):
        p = r.get("prototype") or {}
        v = p.get("workbook_hci" if key == "proto_workbook" else "compute_hci")
        return float("nan") if v is None else float(v)
    v = r.get(key)
    return float("nan") if v is None else float(v)


def summarize(report: dict[str, Any], out: Path) -> dict[str, Any]:
    results = report["results"]
    table: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in results:
        fam = table.setdefault(r["family"], {})
        cond = fam.setdefault(r["condition"], {k: [] for k in METRICS})
        for k in METRICS:
            cond[k].append(_metric(r, k))
    stats = {
        fam: {
            cond: {k: {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None, "sd": float(np.nanstd(v)) if np.isfinite(v).any() else None, "n": int(np.isfinite(v).sum())} for k, v in vals.items()}
            for cond, vals in conds.items()
        }
        for fam, conds in table.items()
    }
    # Axiom: monotonic in condition order, per replicate, and Spearman rho over all samples.
    from scipy.stats import spearmanr

    axioms = {}
    for fam, (direction, keys) in ORDERED_FAMILIES.items():
        conds = list(table.get(fam, {}))
        order = {c: i for i, c in enumerate(conds)}
        fam_rows = [r for r in results if r["family"] == fam]
        axioms[fam] = {}
        for k in keys:
            reps: dict[int, list[tuple[int, float]]] = {}
            for r in fam_rows:
                reps.setdefault(r["replicate"], []).append((order[r["condition"]], _metric(r, k)))
            ok = 0
            for seq in reps.values():
                vals = [v for _, v in sorted(seq)]
                d = np.diff(vals)
                tol = MONOTONE_TOLERANCE
                ok += bool(np.all(d >= -tol)) if direction == "increasing" else bool(np.all(d <= tol))
            xs = [order[r["condition"]] for r in fam_rows]
            ys = [_metric(r, k) for r in fam_rows]
            rho = spearmanr(xs, ys).statistic if np.isfinite(ys).all() else float("nan")
            axioms[fam][k] = {"direction": direction, "replicates_monotone": ok, "replicates": len(reps), "spearman_rho": float(rho)}
    # Invariance: relative change against the reference sample.
    inv = {}
    ref_by_id = {r["sample_id"]: r for r in results}
    for r in results:
        if r["family"] != "invariance":
            continue
        base, kind = r["condition"].split("__")
        family = "topology" if base == "honeycomb_mesh" else "arrangement"
        ref = ref_by_id[f"{family}__{base}__r00"]
        row = {}
        for k in ("HCI_R", "IBAR_R", "HCI_iso", "PI_R", "proto_workbook"):
            a, b = _metric(ref, k), _metric(r, k)
            if kind == "transpose" and k in ("HCI_R", "IBAR_R", "PI_R"):
                a = _metric(ref, {"HCI_R": "HCI_C", "IBAR_R": "IBAR_C", "PI_R": "PI_C"}[k])
            row[k] = None if not (np.isfinite(a) and np.isfinite(b)) or a == 0 else round((b - a) / abs(a), 4)
        inv.setdefault(base, {})[kind] = row
    summary = {"schema_version": "microseg.hci_candidate_summary.v1", "condition_stats": stats, "ordering_axioms": axioms, "invariance_relative_change": inv}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _figure(stats, out / "hci_candidate_evidence.svg")
    return summary


def _figure(stats: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("gap_sweep", "Gap sweep (Af = 5 %)", "surface gap"),
        ("fragmentation", "Fragmentation (Af = 5 %)", "segments"),
        ("arrangement", "Arrangement (Af = 5 %)", ""),
        ("topology", "Topology (Af = 5 %)", ""),
        ("orientation", "Orientation (Af = 5 %)", "angle from circumferential"),
        ("boolean_model", "Boolean model (Af varies)", ""),
    ]
    series = [("IBAR_R", "HCI_R v1 (integrated, δ_max = 10 µm)", "#e76f51", "D"), ("HCI_R", "C_R(δ0 = 2.5 µm)", "#2a6f97", "o"), ("HCI_iso", "C_iso(δ0)", "#6a4c93", "s"), ("PI_R", "Π_R path continuity", "#2a9d8f", "^"), ("proto_workbook", "prototype HCI ÷ 10", "#c1121f", "x")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (fam, title, xlabel) in zip(axes.ravel(), panels):
        conds = list(stats.get(fam, {}))
        x = np.arange(len(conds))
        for key, label, color, marker in series:
            m = [stats[fam][c][key]["mean"] for c in conds]
            s = [stats[fam][c][key]["sd"] for c in conds]
            m = np.array([np.nan if v is None else v for v in m], float)
            s = np.array([0 if v is None else v for v in s], float)
            if key == "proto_workbook":
                m, s = m / 10.0, s / 10.0
            # Break the line between condition groups (e.g. isotropic vs radial Boolean model).
            groups = [c.split("_")[0] for c in conds] if fam == "boolean_model" else [""] * len(conds)
            first = True
            for g in dict.fromkeys(groups):
                idx = [i for i, gg in enumerate(groups) if gg == g]
                ax.errorbar(x[idx], m[idx], yerr=s[idx], label=label if first else None, color=color, marker=marker, capsize=3, lw=1.6, ms=5)
                first = False
        labels = [c.replace("random_", "rnd ").replace("radial_", "").replace("isotropic_", "iso ").replace("theta_", "").replace("deg", "°").replace("_", " ") for c in conds]
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Candidate indices on the constant-area-fraction synthetic benchmark (mean ± SD over replicates)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summarize_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/hci_candidate_study")
    args = parser.parse_args(sys.argv[2:])
    out = Path(args.output)
    summarize(json.loads((out / "report.json").read_text(encoding="utf-8")), out)
    print(f"summary: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        raise SystemExit(_summarize_main())
    raise SystemExit(main())
