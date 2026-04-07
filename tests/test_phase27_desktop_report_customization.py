"""Phase 27 tests for single-run desktop report customization outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.app import DesktopResultExportConfig, DesktopResultExporter, DesktopWorkflowManager
from src.microseg.corrections import CorrectionSession


def _synthetic_image() -> np.ndarray:
    image = np.zeros((96, 96), dtype=np.uint8)
    image[12:42, 10:70] = 220
    image[55:80, 30:88] = 180
    return image


def _write_image(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _conventional_model_name(workflow: DesktopWorkflowManager) -> str:
    return next(name for name in workflow.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase27_single_run_report_customization(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)
    image_path = tmp_path / "input.png"
    _write_image(image_path, _synthetic_image())

    record = workflow.run_single(
        str(image_path),
        model_name=model_name,
        params={"image_path": str(image_path)},
        include_analysis=True,
    )
    sess = CorrectionSession(np.array(record.mask_image))
    sess.apply_brush(x=18, y=18, radius=4, mode="add", class_index=1)

    cfg = DesktopResultExportConfig(
        write_html_report=True,
        write_pdf_report=False,
        write_csv_report=True,
        report_profile="balanced",
        selected_metric_keys=("hydride_count", "orientation_mean_deg"),
        include_sections=("metadata", "key_summary", "scalar_table", "artifact_manifest"),
        top_k_key_metrics=2,
        include_artifact_manifest=True,
    )
    out_dir = DesktopResultExporter().export(
        record,
        output_dir=tmp_path / "results",
        corrected_mask=sess.current_mask,
        annotator="phase27",
        notes="customized",
        config=cfg,
    )

    summary = json.loads((out_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "microseg.desktop_results.v2"
    assert summary["applied_export_criteria"]["report_profile"] == "balanced"
    metrics = {row["metric"] for row in summary["selected_metric_rows"]}
    assert metrics == {"hydride_count", "orientation_mean_deg"}
    assert (out_dir / "results_metrics.csv").exists()
    assert (out_dir / "results_report.html").exists()
    assert not (out_dir / "results_report.pdf").exists()
    assert (out_dir / "artifacts_manifest.json").exists()

    html_text = (out_dir / "results_report.html").read_text(encoding="utf-8")
    assert "Key Metrics" in html_text
    assert "Scalar Statistics" in html_text
    assert "Distributions" not in html_text
