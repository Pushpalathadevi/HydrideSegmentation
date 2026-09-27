"""HCI on the real hydrided micrographs from the intern report (Table 2: 25, 70, 200 ppm).

Inputs (``test_data/hci_report_images/``) are the figures embedded in the report:
``optical_<ppm>.png`` (513 × 384, 100 µm scale bar ≈ 46.5 px ⇒ 2.15 µm/px) and
``mask_<ppm>.png`` (the intern's segmentation, red on black, 344 × 258 ⇒ 3.21 µm/px).
They are figure resolutions, not the original acquisitions, so absolute values are
indicative only.

For each specimen the study computes the v1 HCI under three δ_max rules, runs the
intern prototype unchanged on the same mask, and re-segments the optical image with
the conventional and ML models to show the full pipeline. Results go to
``docs/hci_evidence/real_report_images/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.evaluation.continuity import Calibration, ContinuityAnalysisConfig, analyze_continuity  # noqa: E402
from src.microseg.evaluation.continuity import visualization as viz  # noqa: E402

SPECIMENS = ("25ppm", "70ppm", "200ppm")
OPTICAL_UM_PER_PX = 100.0 / 46.5
MASK_UM_PER_PX = OPTICAL_UM_PER_PX * 513.0 / 344.0
PROTOTYPE_REPORTED = {"25ppm": 2.5891, "70ppm": 3.3764, "200ppm": 8.0192}
RULES = {
    "auto_min": ContinuityAnalysisConfig(),
    "auto_median": ContinuityAnalysisConfig(auto_size_statistic="percentile", auto_size_percentile=50.0),
    "fixed_10um": ContinuityAnalysisConfig(max_bridge_distance=10.0),
}


def red_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return (a[..., 0] >= 128) & (a[..., 1] < 110) & (a[..., 2] < 110)


def summarise(res) -> dict:
    s = res.specimen
    return {
        "status": res.status,
        "HCI": s["HCI"],
        "C_at_dmax": s.get("connectivity_at_report_distance"),
        "delta_half": s.get("critical_linking_distance"),
        "max_bridge_distance_um": s.get("max_bridge_distance"),
        "max_bridge_distance_source": s.get("max_bridge_distance_source"),
        "smallest_hydride_length_um": s.get("smallest_hydride_length"),
        "path_continuity": s.get("path_continuity"),
        "topology": s.get("topology"),
        "area_fraction_cleaned": s.get("area_fraction_cleaned"),
        "n_components": s.get("n_components"),
        "quality_flags": res.quality_flags,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="test_data/hci_report_images")
    parser.add_argument("--output", default="docs/hci_evidence/real_report_images")
    parser.add_argument("--prototype", default="HARSHAD_WORK/HCI_2_4_works_1 (1).py")
    parser.add_argument("--skip-ml", action="store_true")
    args = parser.parse_args(argv)
    src = Path(args.images)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": "microseg.hci_real_image_study.v1",
        "pixel_size_um": {"optical": OPTICAL_UM_PER_PX, "report_mask": MASK_UM_PER_PX},
        "prototype_reported_hci": PROTOTYPE_REPORTED,
        "specimens": {},
    }
    for spec in SPECIMENS:
        print(f"[real] {spec}", flush=True)
        entry: dict = {}
        mask = red_mask(src / f"mask_{spec}.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"mask_{spec}_binary.png")
        entry["report_mask_area_fraction"] = float(mask.mean())
        entry["report_mask"] = {}
        for rule, cfg in RULES.items():
            res = analyze_continuity(mask, cfg, Calibration(MASK_UM_PER_PX, "scale bar"))
            entry["report_mask"][rule] = summarise(res)
            if rule in ("auto_min", "fixed_10um"):
                (out / f"{spec}_report_mask_{rule}_clusters.png").write_bytes(viz.to_png_bytes(viz.cluster_overlay(res)))
                (out / f"{spec}_report_mask_{rule}_curve.png").write_bytes(viz.curve_figure(res, dpi=150, title=f"{spec}: connectivity function ({rule})"))
            if rule == "fixed_10um":
                (out / f"{spec}_report_mask_topology.png").write_bytes(viz.to_png_bytes(viz.topology_overlay(res)))
                (out / f"{spec}_report_mask_path.png").write_bytes(viz.to_png_bytes(viz.path_overlay(res, "radial")))
                (out / f"{spec}_report_mask_hci_result.json").write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
        # The intern prototype, unchanged, on the same binary mask.
        proto = Path(args.prototype)
        if proto.exists():
            sys.path.insert(0, str(ROOT / "scripts"))
            from hci_candidate_study import run_prototype

            entry["prototype_on_report_mask"] = run_prototype(proto, out / f"mask_{spec}_binary.png", 600)
        # Full pipeline on the optical image.
        entry["resegmented"] = {}
        from hydride_segmentation.microseg_adapter import run_pipeline_array

        optical = np.asarray(Image.open(src / f"optical_{spec}.png").convert("RGB"))
        models = ["hydride_conventional"] + ([] if args.skip_ml else ["hydride_ml"])
        for model_id in models:
            try:
                result = run_pipeline_array(optical, source_name=f"optical_{spec}.png", model_id=model_id, params={}, include_analysis=False)
                seg = np.asarray(result.mask)
                Image.fromarray(((seg > 0) * 255).astype(np.uint8)).save(out / f"{spec}_{model_id}_mask.png")
                res = analyze_continuity(seg, RULES["fixed_10um"], Calibration(OPTICAL_UM_PER_PX, "scale bar"))
                res_auto = analyze_continuity(seg, RULES["auto_min"], Calibration(OPTICAL_UM_PER_PX, "scale bar"))
                entry["resegmented"][model_id] = {"fixed_10um": summarise(res), "auto_min": summarise(res_auto), "area_fraction": float((seg > 0).mean())}
                (out / f"{spec}_{model_id}_clusters.png").write_bytes(viz.to_png_bytes(viz.cluster_overlay(res)))
            except Exception as exc:  # recorded in the report
                entry["resegmented"][model_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        report["specimens"][spec] = entry
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
