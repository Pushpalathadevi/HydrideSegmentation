"""Phase 27 tests for desktop batch summary export artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.app import DesktopResultExportConfig, DesktopResultExporter, DesktopWorkflowManager


def _img_a() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[8:45, 8:32] = 220
    arr[55:88, 50:85] = 200
    return arr


def _img_b() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[12:35, 12:75] = 230
    arr[48:78, 44:92] = 185
    return arr


def _write(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _conventional_model_name(workflow: DesktopWorkflowManager) -> str:
    return next(name for name in workflow.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase27_batch_export_outputs(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write(a, _img_a())
    _write(b, _img_b())

    rec_a = workflow.run_single(str(a), model_name=model_name, params={"image_path": str(a)}, include_analysis=True)
    rec_b = workflow.run_single(str(b), model_name=model_name, params={"image_path": str(b)}, include_analysis=True)

    out_dir = DesktopResultExporter().export_batch(
        [rec_a, rec_b],
        output_dir=tmp_path / "batch",
        annotator="batch_user",
        notes="batch test",
        config=DesktopResultExportConfig(
            write_html_report=True,
            write_pdf_report=False,
            write_csv_report=True,
            report_profile="balanced",
            selected_metric_keys=("hydride_count", "hydride_area_fraction_percent"),
            include_sections=("metadata", "scalar_table"),
        ),
    )

    assert (out_dir / "batch_results_summary.json").exists()
    assert (out_dir / "batch_results_report.html").exists()
    assert (out_dir / "batch_metrics.csv").exists()
    assert (out_dir / "artifacts_manifest.json").exists()
    assert (out_dir / "preview_images").exists()
    assert not (out_dir / "batch_results_report.pdf").exists()

    payload = json.loads((out_dir / "batch_results_summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.desktop_batch_results.v1"
    assert int(payload["run_count"]) == 2
    assert len(payload["rows"]) == 2
    assert all(str(row.get("input_preview_path", "")).startswith("preview_images/") for row in payload["rows"])
    assert all(str(row.get("mask_preview_path", "")).startswith("preview_images/") for row in payload["rows"])
    assert all(str(row.get("overlay_preview_path", "")).startswith("preview_images/") for row in payload["rows"])
    assert payload["aggregate_metrics"]
    assert "Hydride" in next(iter(payload["model_counts"].keys()))
