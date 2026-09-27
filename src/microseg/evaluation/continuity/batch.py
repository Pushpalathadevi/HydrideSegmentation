"""Batch HCI analysis of mask files with per-image artefacts and a run report."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analyzer import analyze_continuity
from .config import Calibration, ContinuityAnalysisConfig

MASK_SUFFIXES = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
LOGGER = logging.getLogger("microseg.hci")


@dataclass
class HciBatchSummary:
    output_dir: Path
    report_path: Path
    n_images: int
    n_ok: int
    n_not_applicable: int
    n_failed: int


def collect_masks(path: str | Path) -> list[Path]:
    """A single mask file or every mask-like file in a folder (sorted, non-recursive)."""

    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in MASK_SUFFIXES)
    raise FileNotFoundError(f"mask path not found: {p}")


def _load(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path)
    arr = np.array(img)
    return arr


def _write_image_artifacts(result, out: Path, write_overlays: bool) -> dict[str, str]:
    from . import visualization as viz

    out.mkdir(parents=True, exist_ok=True)
    files = {"result": "hci_result.json"}
    (out / "hci_result.json").write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    if result.ok:
        with (out / "clusters.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster_id", "n_members", "area", "radial_coverage", "circumferential_coverage", "feret", "weight_radial", "weight_circumferential", "weight_isotropic", "n_bridges", "touches_border", "endpoints", "junctions"])
            for c in result.clusters:
                w.writerow([c.cluster_id, len(c.members), c.area, c.coverage_radial, c.coverage_circumferential, c.feret, c.weight["radial"], c.weight["circumferential"], c.weight["isotropic"], len(c.bridges), c.touches_border, c.topology.get("endpoints", 0), c.topology.get("junctions", 0)])
        files["clusters"] = "clusters.csv"
        if write_overlays:
            (out / "hci_clusters.png").write_bytes(viz.to_png_bytes(viz.cluster_overlay(result)))
            (out / "hci_topology.png").write_bytes(viz.to_png_bytes(viz.topology_overlay(result)))
            (out / "hci_path_radial.png").write_bytes(viz.to_png_bytes(viz.path_overlay(result, "radial")))
            (out / "hci_curve.png").write_bytes(viz.curve_figure(result))
            files.update({"clusters_overlay": "hci_clusters.png", "topology_overlay": "hci_topology.png", "path_overlay": "hci_path_radial.png", "curve": "hci_curve.png"})
    return files


def run_hci_batch(
    masks: list[Path],
    output_dir: str | Path,
    config: ContinuityAnalysisConfig,
    *,
    pixel_size_um: float | None = None,
    mask_source: str = "predicted",
    write_overlays: bool = True,
    log: Callable[[str], None] | None = None,
) -> HciBatchSummary:
    """Analyse each mask, writing per-image artefacts plus ``report.json`` and ``summary.csv``.

    A failure on one image is recorded with its error and does not stop the run.
    """

    emit = log or LOGGER.info
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = Calibration(pixel_size_um, "cli" if pixel_size_um else "none")
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    n = len(masks)
    for i, path in enumerate(masks, 1):
        entry: dict[str, Any] = {"image": path.name, "path": str(path)}
        try:
            res = analyze_continuity(_load(path), config, cal, mask_source=mask_source)
            files = _write_image_artifacts(res, out / path.stem, write_overlays)
            s = res.specimen
            entry.update({"status": res.status, "status_reason": res.status_reason, "unit": res.unit, "artifacts": {k: f"{path.stem}/{v}" for k, v in files.items()}})
            if res.ok:
                entry.update(
                    {
                        "HCI_radial": s["HCI"]["radial"],
                        "HCI_circumferential": s["HCI"]["circumferential"],
                        "HCI_isotropic": s["HCI"]["isotropic"],
                        "critical_linking_distance_radial": s["critical_linking_distance"]["radial"],
                        "max_bridge_distance": s["max_bridge_distance"],
                        "max_bridge_distance_source": s["max_bridge_distance_source"],
                        "path_continuity_radial": s["path_continuity"].get("radial"),
                        "network_closure": s.get("topology", {}).get("network_closure"),
                        "area_fraction_cleaned": s["area_fraction_cleaned"],
                        "n_components": s["n_components"],
                        "quality_flags": ";".join(res.quality_flags),
                    }
                )
        except Exception as exc:  # recorded, not fatal for the batch
            entry.update({"status": "failed", "status_reason": f"{type(exc).__name__}: {exc}"})
        rows.append(entry)
        elapsed = time.monotonic() - t0
        emit(f"[hci] {i}/{n} ({100 * i / n:.0f}%) {path.name}: {entry['status']} | elapsed {elapsed:.1f}s ETA {elapsed / i * (n - i):.1f}s")
    report = {
        "schema_version": "microseg.hci_batch_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "calibration": {"pixel_size_um": pixel_size_um, "source": cal.source},
        "mask_source": mask_source,
        "n_images": n,
        "results": rows,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    columns = ["image", "status", "unit", "HCI_radial", "HCI_circumferential", "HCI_isotropic", "critical_linking_distance_radial", "max_bridge_distance", "max_bridge_distance_source", "path_continuity_radial", "network_closure", "area_fraction_cleaned", "n_components", "quality_flags", "status_reason"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return HciBatchSummary(
        output_dir=out,
        report_path=report_path,
        n_images=n,
        n_ok=sum(r["status"] == "ok" for r in rows),
        n_not_applicable=sum(r["status"] == "not_applicable" for r in rows),
        n_failed=sum(r["status"] == "failed" for r in rows),
    )
