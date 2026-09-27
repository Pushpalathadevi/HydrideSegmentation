"""Evaluate the production HCI analyzer on the synthetic benchmark.

Writes ``report.json`` (per-sample results), ``summary.json`` (condition
statistics, ordering axioms, invariance changes) and an evidence figure. The
results back ``docs/hci_specification.md`` §10 and the review deck.

Usage::

    python scripts/hci_benchmark_evaluation.py --benchmark test_data/hci_synthetic_v1 --output docs/hci_evidence
    python scripts/hci_benchmark_evaluation.py --set max_bridge_distance=10.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.microseg.evaluation.continuity import Calibration, ContinuityAnalysisConfig, analyze_continuity
from src.microseg.io.configuration import parse_set_overrides

MONOTONE_TOLERANCE = 0.01
ORDERED = {
    "gap_sweep": "increasing",
    "fragmentation": "decreasing",
    "arrangement": "increasing",
    "orientation": "increasing",
}
KEYS = ("HCI_R", "HCI_C", "HCI_iso", "C0_R", "PI_R", "delta_half_R", "dmax", "closure")


def evaluate(entry: dict[str, Any], root: Path, cfg: ContinuityAnalysisConfig) -> dict[str, Any]:
    from PIL import Image

    mask = np.array(Image.open(root / entry["mask_path"]))
    res = analyze_continuity(mask, cfg, Calibration(entry["pixel_size_um"], "manifest"))
    row: dict[str, Any] = {k: entry[k] for k in ("sample_id", "family", "condition", "replicate", "area_fraction")}
    row["status"] = res.status
    if res.ok:
        s = res.specimen
        row.update(
            {
                "HCI_R": s["HCI"]["radial"],
                "HCI_C": s["HCI"]["circumferential"],
                "HCI_iso": s["HCI"]["isotropic"],
                "C0_R": s["connectivity_at_report_distance"]["radial"],
                "PI_R": s["path_continuity"].get("radial"),
                "PI_C": s["path_continuity"].get("circumferential"),
                "delta_half_R": s["critical_linking_distance"]["radial"],
                "dmax": s["max_bridge_distance"],
                "closure": s["topology"]["network_closure"],
                "endpoints": s["topology"]["endpoints"],
                "junctions": s["topology"]["junctions"],
                "loops": s["topology"]["loops"],
                "runtime_s": res.provenance["runtime_s"],
            }
        )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        cond = stats.setdefault(r["family"], {}).setdefault(r["condition"], {k: [] for k in KEYS})
        for k in KEYS:
            v = r.get(k)
            cond[k].append(np.nan if v is None else float(v))
    table = {
        fam: {
            c: {k: ({"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v))} if np.isfinite(v).any() else None) for k, v in vals.items()}
            for c, vals in conds.items()
        }
        for fam, conds in stats.items()
    }
    axioms: dict[str, Any] = {}
    for fam, direction in ORDERED.items():
        conds = list(stats.get(fam, {}))
        order = {c: i for i, c in enumerate(conds)}
        fam_rows = [r for r in rows if r["family"] == fam]
        axioms[fam] = {}
        for k in ("HCI_R", "C0_R", "PI_R"):
            reps: dict[int, list[tuple[int, float]]] = {}
            for r in fam_rows:
                reps.setdefault(r["replicate"], []).append((order[r["condition"]], float(r[k])))
            ok = 0
            for seq in reps.values():
                d = np.diff([v for _, v in sorted(seq)])
                ok += bool(np.all(d >= -MONOTONE_TOLERANCE)) if direction == "increasing" else bool(np.all(d <= MONOTONE_TOLERANCE))
            rho = spearmanr([order[r["condition"]] for r in fam_rows], [r[k] for r in fam_rows]).statistic
            axioms[fam][k] = {"direction": direction, "replicates_monotone": ok, "replicates": len(reps), "spearman_rho": float(rho)}
    by_id = {r["sample_id"]: r for r in rows}
    inv: dict[str, dict[str, dict[str, float | None]]] = {}
    for r in rows:
        if r["family"] != "invariance" or r["status"] != "ok":
            continue
        base, kind = r["condition"].split("__")
        fam = "topology" if base == "honeycomb_mesh" else "arrangement"
        ref = by_id[f"{fam}__{base}__r00"]
        out = {}
        for k in ("HCI_R", "HCI_iso", "PI_R"):
            a_key = {"HCI_R": "HCI_C", "PI_R": "PI_C"}.get(k, k) if kind == "transpose" else k
            a, b = ref.get(a_key), r.get(k)
            out[k] = None if not a else round((b - a) / abs(a), 4)
        inv.setdefault(base, {})[kind] = out
    return {"schema_version": "microseg.hci_benchmark_summary.v1", "tolerance": MONOTONE_TOLERANCE, "condition_stats": table, "ordering_axioms": axioms, "invariance_relative_change": inv}


def figure(table: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("gap_sweep", "Gap sweep (Af = 5 %)"),
        ("fragmentation", "Fragmentation (Af = 5 %)"),
        ("arrangement", "Arrangement (Af = 5 %)"),
        ("topology", "Topology (Af = 5 %)"),
        ("orientation", "Orientation (Af = 5 %)"),
        ("boolean_model", "Boolean model (Af varies)"),
    ]
    series = [("HCI_R", "HCI radial", "#e76f51", "D"), ("HCI_iso", "HCI isotropic", "#6a4c93", "s"), ("C0_R", "C radial at δ_max", "#2a6f97", "o"), ("PI_R", "Π radial (path)", "#2a9d8f", "^")]
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (fam, title) in zip(axes.ravel(), panels):
        conds = list(table.get(fam, {}))
        x = np.arange(len(conds))
        groups = [c.split("_")[0] for c in conds] if fam == "boolean_model" else [""] * len(conds)
        for key, label, color, marker in series:
            m = np.array([table[fam][c][key]["mean"] if table[fam][c][key] else np.nan for c in conds])
            s = np.array([table[fam][c][key]["sd"] if table[fam][c][key] else 0 for c in conds])
            first = True
            for g in dict.fromkeys(groups):
                idx = [i for i, gg in enumerate(groups) if gg == g]
                ax.errorbar(x[idx], m[idx], yerr=s[idx], color=color, marker=marker, capsize=3, lw=2, ms=6, label=label if first else None)
                first = False
        labels = [c.replace("random_", "rnd ").replace("radial_", "").replace("theta_", "").replace("deg", "°").replace("_af", " ").replace("_", " ") for c in conds]
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="test_data/hci_synthetic_v1")
    parser.add_argument("--output", default="docs/hci_evidence")
    parser.add_argument("--set", action="append", default=[], help="HCI config override, e.g. max_bridge_distance=10.0")
    args = parser.parse_args(argv)
    cfg = ContinuityAnalysisConfig.from_mapping(parse_set_overrides(args.set))
    root = Path(args.benchmark)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.monotonic()
    n = len(manifest["samples"])
    for i, e in enumerate(manifest["samples"], 1):
        rows.append(evaluate(e, root, cfg))
        if i % 25 == 0 or i == n:
            el = time.monotonic() - t0
            print(f"[hci-benchmark] {i}/{n} ({100 * i / n:.0f}%) elapsed {el:.0f}s ETA {el / i * (n - i):.0f}s", flush=True)
    summary = summarize(rows)
    summary["config"] = cfg.to_dict()
    (out / "report.json").write_text(json.dumps({"schema_version": "microseg.hci_benchmark_report.v1", "config": cfg.to_dict(), "benchmark": manifest["schema_version"], "results": rows}, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    figure(summary["condition_stats"], out / "hci_benchmark_evidence.svg")
    print(f"summary: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
