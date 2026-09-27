"""Build every figure used in the HCI review deck from committed data.

Run from the repository root::

    python docs/presentations/hci_review/build_assets.py

Outputs go to ``docs/presentations/hci_review/assets/``. SVG diagrams from
``docs/diagrams`` are also rasterised (Microsoft Edge headless) to provide the
PNG fallbacks PowerPoint needs next to embedded SVG.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.microseg.evaluation.continuity import Calibration, ContinuityAnalysisConfig, analyze_continuity  # noqa: E402
from src.microseg.evaluation.continuity import visualization as viz  # noqa: E402
from src.microseg.evaluation.hydride_statistics import compute_hydride_statistics  # noqa: E402

OUT = Path(__file__).resolve().parent / "assets"
BENCH = ROOT / "test_data" / "hci_synthetic_v1"
MAN = {e["sample_id"]: e for e in json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))["samples"]}
FIXED = ContinuityAnalysisConfig(max_bridge_distance=10.0)
CAL = Calibration(0.5, "manifest")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)
C_R, C_C, C_I, C_P, C_PROTO = "#e76f51", "#2a9d8f", "#6a4c93", "#264653", "#c1121f"


def mask(sample_id: str) -> np.ndarray:
    return np.array(Image.open(BENCH / MAN[sample_id]["mask_path"])) > 0


def save(fig, name: str, svg: bool = False) -> None:
    fig.savefig(OUT / f"{name}.png")
    if svg:
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def show_mask(ax, m, title=None, invert=True):
    ax.imshow(~m if invert else m, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#9aa9b5")
    if title:
        ax.set_title(title)


def fn_length(m: np.ndarray) -> float:
    stats = compute_hydride_statistics(m.astype(np.uint8), include_fn_metrics=True, fn_angle_threshold_deg=45.0)
    return float(stats.scalar_metrics.get("fn_length_weighted", float("nan")))


# ---------------------------------------------------------------------------


def problem_same_af() -> None:
    ids = [("arrangement__random_radial__r00", "Random short platelets"), ("arrangement__radial_stringers__r00", "Aligned stringers (3 µm gaps)"), ("arrangement__radial_continuous__r00", "Continuous lines")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6))
    for ax, (sid, label) in zip(axes, ids):
        m = mask(sid)
        r = analyze_continuity(m, FIXED, CAL)
        show_mask(ax, m, label)
        ax.set_xlabel(f"Af = {100 * m.mean():.1f} %   Fn = {fn_length(m):.2f}\nHCI radial = {r.specimen['HCI']['radial']:.2f}", fontsize=18)
    fig.tight_layout()
    save(fig, "problem_same_af")


def prototype_confound() -> None:
    real = json.loads((ROOT / "docs/hci_evidence/real_report_images/report.json").read_text(encoding="utf-8"))
    ppm = [25, 70, 200]
    specs = ["25ppm", "70ppm", "200ppm"]
    reported = [2.5891, 3.3764, 8.0192]
    af = [100 * real["specimens"][s]["report_mask_area_fraction"] for s in specs]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    ax = axes[0]
    ax.plot(ppm, reported, "o-", color=C_PROTO, lw=3, ms=10)
    ax.set_xlabel("hydrogen content (ppm)"); ax.set_ylabel("prototype HCI (report)")
    ax.set_title("Report: HCI rises with hydrogen")
    for x, y in zip(ppm, reported):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(8, -18), fontsize=16)
    ax = axes[1]
    ax.plot(af, reported, "s-", color=C_PROTO, lw=3, ms=10)
    for x, y, p in zip(af, reported, ppm):
        ax.annotate(f"{p} ppm", (x, y), textcoords="offset points", xytext=(8, -18), fontsize=16)
    ax.set_xlabel("hydride area fraction of the same masks (%)"); ax.set_ylabel("prototype HCI (report)")
    ax.set_title("…but so does the amount of hydride")
    fig.tight_layout()
    save(fig, "prototype_confound")


def prototype_failures() -> None:
    s = json.loads((ROOT / "docs/hci_candidate_study.summary.json").read_text(encoding="utf-8"))
    cs = s["condition_stats"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    # (a) gap sweep
    ax = axes[0]
    conds = list(cs["gap_sweep"])
    x = np.arange(len(conds))
    m = [cs["gap_sweep"][c]["proto_workbook"]["mean"] for c in conds]
    e = [cs["gap_sweep"][c]["proto_workbook"]["sd"] for c in conds]
    ax.errorbar(x, m, yerr=e, color=C_PROTO, marker="o", lw=3, capsize=4)
    ax.set_xticks(x, [c.split("_")[1].replace("px", "") for c in conds])
    ax.set_xlabel("gap between segments (px)"); ax.set_title("(a) Drops when chains merge")
    ax.annotate("merged", (5, m[5]), xytext=(3.2, m[5] - 2.2), arrowprops=dict(arrowstyle="->", lw=2), fontsize=16)
    # (b) topology
    ax = axes[1]
    conds = list(cs["topology"])
    x = np.arange(len(conds))
    ax.bar(x, [cs["topology"][c]["proto_workbook"]["mean"] for c in conds], color=C_PROTO)
    ax.set_xticks(x, ["lines", "Y", "X", "trees", "rings", "mesh"])
    ax.set_title("(b) Zero for rings and a mesh")
    ax.set_ylabel("prototype HCI")
    # (c) orientation
    ax = axes[2]
    conds = list(cs["orientation"])
    ang = [int(c.split("_")[1].replace("deg", "")) for c in conds]
    ax.plot(ang, [cs["orientation"][c]["proto_workbook"]["mean"] for c in conds], "o-", color=C_PROTO, lw=3, ms=8)
    ax.set_xlabel("rotation of identical lines (°)"); ax.set_title("(c) Changes under rotation")
    ax.set_xticks([0, 30, 60, 90])
    # (d) invariance
    ax = axes[3]
    inv = s["invariance_relative_change"]
    kinds = ["transpose", "scale_0.5", "scale_2.0", "boundary_roughness"]
    proto = [100 * max(abs(inv[b][k]["proto_workbook"] or 0) for b in inv) for k in kinds]
    ax.bar(np.arange(4), proto, color=C_PROTO)
    ax.set_xticks(np.arange(4), ["transpose", "×0.5", "×2", "rough"])
    ax.set_ylabel("max |change| (%)"); ax.set_title("(d) Not invariant")
    fig.tight_layout()
    save(fig, "prototype_failures")


def benchmark_construction() -> None:
    sid = "topology__y_junctions__r00"
    geo = json.loads((BENCH / "geometry" / "topology.json").read_text(encoding="utf-8"))[sid]
    nodes = np.array(geo["nodes"])
    m = mask(sid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    ax = axes[0]
    for a, b in geo["edges"]:
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color="#1d3557", lw=2)
    deg = np.bincount(np.array(geo["edges"]).ravel(), minlength=len(nodes))
    ax.scatter(nodes[deg == 1, 0], nodes[deg == 1, 1], s=40, color="#e63946", zorder=3, label="free end (degree 1)")
    ax.scatter(nodes[deg >= 3, 0], nodes[deg >= 3, 1], s=60, color="#1d3557", marker="s", zorder=3, label="junction (degree 3)")
    ax.set_xlim(0, 511); ax.set_ylim(511, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("1. Centreline network = exact ground truth"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    show_mask(axes[1], m, "2. Rasterised capsules (4 px thick), Af = 5.0 %")
    fig.tight_layout()
    save(fig, "benchmark_construction")


FAMILY_LABELS = {
    "gap_sweep": ("gap_sweep", ["gap_40px", "gap_20px", "gap_10px", "gap_06px", "gap_03px", "gap_00px"], ["40 px", "20 px", "10 px", "6 px", "3 px", "merged"]),
    "fragmentation": ("fragmentation", ["n_008", "n_016", "n_032", "n_064", "n_128"], ["8 pieces", "16", "32", "64", "128"]),
    "arrangement": ("arrangement", ["random_circumferential", "random_isotropic", "random_radial", "radial_stringers", "radial_stepped", "radial_continuous"], ["random circ.", "random isotropic", "random radial", "stringers", "stepped", "continuous"]),
    "topology": ("topology", ["lines", "y_junctions", "crosses", "branched_trees", "rings", "honeycomb_mesh"], ["lines", "Y", "X", "trees", "rings", "mesh"]),
    "orientation": ("orientation", [f"theta_{d:02d}deg" for d in (0, 15, 30, 45, 60, 75, 90)], ["0°", "15°", "30°", "45°", "60°", "75°", "90°"]),
    "boolean_model": ("boolean_model", [f"isotropic_af{a:02d}" for a in (2, 5, 10, 15, 20)], ["Af 2 %", "5 %", "10 %", "15 %", "20 %"]),
    "degenerate": ("degenerate", ["empty", "speck_below_min_area", "single_segment", "single_spanning_radial", "single_ring", "single_mesh_cluster", "two_parallel_lines"], ["empty", "speck", "segment", "spanning", "ring", "mesh", "pair"]),
}


def family_strips() -> None:
    for name, (fam, conds, labels) in FAMILY_LABELS.items():
        n = len(conds)
        fig, axes = plt.subplots(1, n, figsize=(2.75 * n, 3.3))
        for ax, c, lab in zip(axes, conds, labels):
            show_mask(ax, mask(f"{fam}__{c}__r00"), lab)
        fig.tight_layout()
        save(fig, f"family_{name}")
    kinds = ["transpose", "flip_lr", "scale_0.5", "scale_2.0", "boundary_roughness", "speckle", "pinholes", "spurs", "breaks"]
    labels = ["transpose", "flip", "×0.5 res.", "×2 res.", "roughness", "speckle", "pinholes", "spurs", "2 px breaks"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(2.75 * len(kinds), 3.3))
    for ax, k, lab in zip(axes, kinds, labels):
        m = mask(f"invariance__radial_stringers__{k}__r00")
        h, w = m.shape
        show_mask(ax, m[: h // 2, : w // 2], lab)
    fig.tight_layout()
    save(fig, "family_invariance")


def _cluster_rgb(res, delta_px: float) -> np.ndarray:
    link = res.arrays["linkage"]
    roots = link.roots_at(delta_px)
    labels = res.arrays["labels"]
    rimg = roots[labels]
    img = np.full(labels.shape + (3,), 255, np.uint8)
    uniq = sorted({int(r) for r in np.unique(rimg[labels > 0])}, key=lambda r: -int((rimg == r).sum()))
    pal = viz._PALETTE
    lut = np.zeros((int(roots.max()) + 1, 3), np.uint8)
    for i, r in enumerate(uniq):
        lut[r] = pal[i % len(pal)]
    img[labels > 0] = lut[rimg[labels > 0]]
    for ev in link.events:
        if ev.gap_px > delta_px:
            break
        p, q = link.edges.p_rc[ev.edge_index], link.edges.q_rc[ev.edge_index]
        viz._line(img, tuple(p), tuple(q), viz.BRIDGE_RGB, 3)
    return img


def stages(sid: str, prefix: str, deltas_um: list[float], cal: Calibration, cfg: ContinuityAnalysisConfig, crop: tuple[slice, slice] | None = None, mask_array: np.ndarray | None = None) -> None:
    from scipy.ndimage import distance_transform_edt

    m = mask(sid) if mask_array is None else mask_array
    res = analyze_continuity(m, cfg, cal)
    link = res.arrays["linkage"]
    comp = link.components
    a = cal.scale
    cr = crop or (slice(None), slice(None))
    # Stage 1: mask and components
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    show_mask(axes[0], res.arrays["raw"][cr], "1. Segmented mask (input)")
    rng = np.random.default_rng(3)
    cols = np.vstack([[1, 1, 1], rng.uniform(0.15, 0.85, (comp.n, 3))])
    axes[1].imshow(cols[comp.labels[cr]], interpolation="nearest")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(f"2. Clean-up + 8-connected components (n = {comp.n})")
    fig.tight_layout(); save(fig, f"{prefix}_stage1")
    # Stage 2: nearest-component (Voronoi) map and gap edges
    _, (iy, ix) = distance_transform_edt(comp.labels == 0, return_indices=True)
    near = comp.labels[iy, ix]
    rng = np.random.default_rng(5)
    pastel = np.vstack([[1, 1, 1], 0.55 + 0.45 * rng.uniform(0, 1, (comp.n, 3))])
    vor = pastel[near]
    vor[comp.labels > 0] = [0.15, 0.15, 0.15]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.imshow(vor[cr], interpolation="nearest")
    r0 = cr[0].start or 0
    c0 = cr[1].start or 0
    for ev in link.events:
        g_um = ev.gap_px * a
        if g_um > max(deltas_um) * 1.6:
            continue
        p, q = link.edges.p_rc[ev.edge_index], link.edges.q_rc[ev.edge_index]
        ax.plot([p[1] - c0, q[1] - c0], [p[0] - r0, q[0] - r0], color="#e63946", lw=3)
        ax.text((p[1] + q[1]) / 2 - c0 + 4, (p[0] + q[0]) / 2 - r0, f"{g_um:.1f}", color="#b00020", fontsize=12, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, vor[cr].shape[1] - 1); ax.set_ylim(vor[cr].shape[0] - 1, 0)
    ax.set_title("3. Nearest-component map; red = minimum-spanning gap edges (µm)")
    fig.tight_layout(); save(fig, f"{prefix}_stage2")
    # Stage 3: clusters at increasing δ
    fig, axes = plt.subplots(1, len(deltas_um), figsize=(4.6 * len(deltas_um), 5.4))
    for ax, d in zip(axes, deltas_um):
        img = _cluster_rgb(res, d / a)
        ax.imshow(img[cr], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        c_r = link.value_at("radial", d / a)
        ax.set_title(f"δ = {d:g} µm: {link.clusters_at_count(d / a)} clusters\nC_R(δ) = {c_r:.2f}")
    fig.tight_layout(); save(fig, f"{prefix}_stage3")
    # Stage 4: coverage of the largest cluster and the connectivity curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    dsel = deltas_um[-1] / a
    roots = link.roots_at(dsel)
    labels = comp.labels
    big = max({int(r) for r in roots[1:]}, key=lambda r: comp.area_px[np.nonzero(roots == r)[0]].sum() if r else 0)
    members = [i for i in range(1, comp.n + 1) if roots[i] == big]
    sel = np.isin(labels, members)
    rows = np.unique(np.concatenate([comp.rows[i] for i in members]))
    img = np.full(labels.shape + (3,), 255, np.uint8)
    img[labels > 0] = (200, 200, 200)
    img[sel] = (31, 119, 180)
    ax.imshow(img[cr], interpolation="nearest")
    xs = (img[cr].shape[1]) * 0.03
    for rr in rows:
        rr2 = rr - r0
        if 0 <= rr2 < img[cr].shape[0]:
            ax.plot([xs, xs + 8], [rr2, rr2], color=C_I, lw=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"4. Coverage S_K(R) = {len(rows) * a:.0f} µm\n(purple bars = hydride rows only)")
    ax = axes[1]
    breaks = link.breaks_px * a
    dmax = res.specimen["max_bridge_distance"]
    right = max(dmax * 2.0, 25.0)
    for d, col, lab in (("radial", C_R, "radial"), ("circumferential", C_C, "circumferential"), ("isotropic", C_I, "isotropic")):
        vals = link.values[d]
        xs_ = np.r_[breaks, right]
        ys_ = np.r_[vals, vals[-1]]
        ax.step(xs_, ys_, where="post", color=col, lw=3, label=f"C {lab}: HCI = {res.specimen['HCI'][d]:.3f}")
        if d == "radial":
            ok = xs_ <= dmax
            ax.fill_between(np.r_[xs_[ok], dmax], 0, np.r_[ys_[ok], ys_[ok][-1]], step="post", color=col, alpha=0.18)
        half = res.specimen["critical_linking_distance"][d]
        if half is not None and half < right:
            ax.plot([half], [0.5], "o", color=col, ms=10)
    ax.axvline(dmax, color="k", ls="--", lw=2)
    ax.text(dmax, 1.03, f"δ_max = {dmax:g} µm", fontsize=16, ha="center")
    ax.axhline(0.5, color="grey", ls=":", lw=1.5)
    ax.set_xlim(0, right); ax.set_ylim(0, 1.1)
    ax.set_xlabel("association distance δ (µm)"); ax.set_ylabel("connectivity C(δ)")
    ax.set_title("5. C(δ) step function; shaded area / δ_max = HCI radial")
    ax.legend(loc="lower right", fontsize=14)
    fig.tight_layout(); save(fig, f"{prefix}_stage4")
    # Stage 5: companions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    axes[0].imshow(viz.path_overlay(res, "radial")[cr], interpolation="nearest")
    axes[0].set_title(f"6. Weakest radial path: Π_R = {res.specimen['path_continuity']['radial']:.2f}")
    t = res.specimen["topology"]
    axes[1].imshow(viz.topology_overlay(res)[cr], interpolation="nearest")
    axes[1].set_title(f"7. Skeleton: {t['endpoints']} ends, {t['junctions']} junctions, {t['loops']} loops, κ = {t['network_closure']:.2f}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); save(fig, f"{prefix}_stage5")


def delta_rules() -> None:
    base = ROOT / "docs/hci_evidence"
    rules = [
        ("auto: 1/5 smallest\n(default)", json.loads((base / "summary.json").read_text(encoding="utf-8"))),
        ("auto: 1/5 P10", json.loads((base / "delta_max_rules/summary_p10.json").read_text(encoding="utf-8"))),
        ("auto: 1/5 median", json.loads((base / "delta_max_rules/summary_p50.json").read_text(encoding="utf-8"))),
        ("fixed 10 µm", json.loads((base / "delta_max_rules/summary_fixed10.json").read_text(encoding="utf-8"))),
    ]
    fams = ["gap_sweep", "fragmentation", "arrangement", "orientation"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    ax = axes[0]
    w = 0.2
    for i, (name, s) in enumerate(rules):
        vals = [100 * s["ordering_axioms"][f]["HCI_R"]["replicates_monotone"] / s["ordering_axioms"][f]["HCI_R"]["replicates"] for f in fams]
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, label=name.replace("\n", " "), color=["#8d99ae", "#adb5bd", "#2a9d8f", "#e76f51"][i])
    ax.set_xticks(np.arange(4), ["gap sweep", "fragmentation", "arrangement", "orientation"])
    ax.set_ylabel("replicates in expected order (%)"); ax.set_ylim(0, 115)
    ax.set_title("Ordering axioms: cross-image comparability")
    ax.legend(fontsize=13, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax = axes[1]
    brk = [100 * max(abs(s["invariance_relative_change"][b]["breaks"]["HCI_R"] or 0) for b in s["invariance_relative_change"]) for _, s in rules]
    ax.bar(np.arange(4), brk, color=["#8d99ae", "#adb5bd", "#2a9d8f", "#e76f51"])
    for i, v in enumerate(brk):
        ax.text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=16)
    ax.axhline(5, color="k", ls="--", lw=1.5); ax.text(3.45, 6.5, "5 % target", fontsize=14, ha="right")
    ax.set_xticks(np.arange(4), [r[0] for r in rules], fontsize=13)
    ax.set_ylabel("|change| under 2-px breaks (%)")
    ax.set_title("Robustness to a few segmentation breaks")
    fig.tight_layout()
    save(fig, "delta_rules")


def real_images() -> None:
    real = json.loads((ROOT / "docs/hci_evidence/real_report_images/report.json").read_text(encoding="utf-8"))
    rdir = ROOT / "docs/hci_evidence/real_report_images"
    specs = ["25ppm", "70ppm", "200ppm"]
    fig, axes = plt.subplots(3, 4, figsize=(19, 11.2))
    for r, spec in enumerate(specs):
        panels = [
            (np.asarray(Image.open(ROOT / f"test_data/hci_report_images/optical_{spec}.png").convert("RGB")), f"{spec.replace('ppm', ' ppm')}: optical (report)"),
            (np.asarray(Image.open(rdir / f"mask_{spec}_binary.png")), "report mask (intern)"),
            (np.asarray(Image.open(rdir / f"{spec}_report_mask_fixed_10um_clusters.png")), "HCI clusters, δ_max = 10 µm"),
            (np.asarray(Image.open(rdir / f"{spec}_hydride_ml_clusters.png")), "re-segmented (ML) → clusters"),
        ]
        for c, (img, title) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img if img.ndim == 3 else 255 - img, cmap=None if img.ndim == 3 else "gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=16)
    fig.tight_layout()
    save(fig, "real_grid")
    # Results chart
    ppm = np.array([25, 70, 200])
    fx = [real["specimens"][s]["report_mask"]["fixed_10um"] for s in specs]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    ax = axes[0]
    for key, col, lab in (("radial", C_R, "radial"), ("circumferential", C_C, "circumferential"), ("isotropic", C_I, "isotropic")):
        ax.plot(ppm, [f["HCI"][key] for f in fx], "o-", color=col, lw=3, ms=10, label=lab)
    ax.set_xlabel("hydrogen (ppm)"); ax.set_ylabel("HCI v1 (δ_max = 10 µm)"); ax.set_ylim(0, 0.45)
    ax.set_title("Connectivity by direction"); ax.legend(fontsize=14, loc="upper left")
    ax = axes[1]
    ax.plot(ppm, [f["delta_half"]["radial"] for f in fx], "o-", color=C_R, lw=3, ms=10, label="δ½ radial")
    ax.plot(ppm, [f["delta_half"]["circumferential"] for f in fx], "s-", color=C_C, lw=3, ms=10, label="δ½ circumferential")
    ax.set_xlabel("hydrogen (ppm)"); ax.set_ylabel("critical linking distance (µm)")
    ax.set_title("Hydrides link at smaller gaps"); ax.legend(fontsize=14)
    ax = axes[2]
    af = [100 * real["specimens"][s]["report_mask_area_fraction"] for s in specs]
    proto = [real["specimens"][s]["prototype_on_report_mask"]["workbook_hci"] for s in specs]
    ax.plot(ppm, af, "d-", color=C_P, lw=3, ms=10, label="area fraction (%)")
    ax.plot(ppm, proto, "x--", color=C_PROTO, lw=3, ms=12, label="prototype HCI (re-run)")
    ax.plot(ppm, [2.5891, 3.3764, 8.0192], "+:", color=C_PROTO, lw=2, ms=14, label="prototype HCI (report)")
    ax.set_xlabel("hydrogen (ppm)"); ax.set_title("Amount rises too (confound)"); ax.legend(fontsize=13)
    fig.tight_layout()
    save(fig, "real_results")


def evidence_png() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from hci_benchmark_evaluation import figure

    s = json.loads((ROOT / "docs/hci_evidence/delta_max_rules/summary_fixed10.json").read_text(encoding="utf-8"))
    figure(s["condition_stats"], OUT / "benchmark_evidence_fixed10.svg")
    figure(s["condition_stats"], OUT / "benchmark_evidence_fixed10.png")


def rasterise_svgs() -> None:
    for name in ("hci_concept", "hci_v1_algorithm_flow", "hci_synthetic_benchmark_overview"):
        src = ROOT / "docs" / "diagrams" / f"{name}.svg"
        shutil.copy(src, OUT / f"{name}.svg")
        text = src.read_text(encoding="utf-8")
        import re

        m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', text)
        w, h = (int(float(m.group(1))), int(float(m.group(2)))) if m else (1400, 900)
        html = OUT / f"_{name}.html"
        html.write_text(f"<html><body style='margin:0;background:white'><img src='{name}.svg' style='width:{w * 2}px;height:{h * 2}px'></body></html>", encoding="utf-8")
        png = OUT / f"{name}.png"
        png.unlink(missing_ok=True)
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", f"--screenshot={png}", f"--window-size={w * 2},{h * 2}", "--hide-scrollbars", html.as_uri()], check=False, capture_output=True, timeout=120)
        # The Edge launcher can return before its headless child has written the file.
        import time

        deadline = time.monotonic() + 60
        while not png.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        time.sleep(1.0)
        html.unlink(missing_ok=True)
        if not png.exists():
            raise RuntimeError(f"Edge did not rasterise {name}.svg")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [
        ("problem", problem_same_af),
        ("confound", prototype_confound),
        ("prototype failures", prototype_failures),
        ("construction", benchmark_construction),
        ("family strips", family_strips),
        ("stages (synthetic)", lambda: stages("arrangement__radial_stringers__r00", "syn", [0.0, 2.0, 3.5, 10.0], CAL, FIXED, (slice(0, 512), slice(0, 512)))),
        ("stages (200 ppm)", lambda: stages("", "real200", [0.0, 5.0, 10.0, 20.0], Calibration(100 / 46.5 * 513 / 344, "scale bar"), FIXED, None, np.asarray(Image.open(ROOT / "docs/hci_evidence/real_report_images/mask_200ppm_binary.png")) > 0)),
        ("delta rules", delta_rules),
        ("real images", real_images),
        ("evidence", evidence_png),
        ("svg rasters", rasterise_svgs),
    ]
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[assets] {i}/{len(steps)} {name}", flush=True)
        fn()
    print(f"assets: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
