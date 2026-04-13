"""Phase 31 tests for unified desktop batch inference/export orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.app import (
    DesktopBatchProgress,
    DesktopResultExportConfig,
    DesktopResultExporter,
    DesktopWorkflowManager,
    collect_inference_images,
    run_desktop_batch_job,
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _conventional_model_name(workflow: DesktopWorkflowManager) -> str:
    return next(name for name in workflow.model_options() if resolve_gui_model_id(name) == "hydride_conventional")


def test_phase31_collect_inference_images_scans_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    _write(tmp_path / "a.png", _img_a())
    _write(nested / "b.tif", _img_b())

    found = collect_inference_images(
        image=None,
        image_dir=str(tmp_path),
        glob_patterns=["*.png", "*.tif"],
        recursive=True,
    )

    assert [path.name for path in found] == ["a.png", "b.tif"]


def test_phase31_batch_job_exports_runs_manifest_and_progress(tmp_path: Path) -> None:
    workflow = DesktopWorkflowManager()
    model_name = _conventional_model_name(workflow)
    a = tmp_path / "inputs" / "a.png"
    b = tmp_path / "inputs" / "nested" / "b.png"
    _write(a, _img_a())
    _write(b, _img_b())

    updates: list[DesktopBatchProgress] = []

    def _finalize(record) -> None:
        record.feedback_record_dir = f"feedback/{record.run_id}"
        record.feedback_record_id = f"fb-{record.run_id}"

    result = run_desktop_batch_job(
        workflow=workflow,
        result_exporter=DesktopResultExporter(),
        image_paths=[a, b],
        model_name=model_name,
        output_dir=tmp_path / "batch",
        params={"enable_gpu": False, "device_policy": "cpu"},
        include_analysis=True,
        annotator="phase31_user",
        notes="phase31 batch",
        export_config=DesktopResultExportConfig(write_pdf_report=False),
        resolved_config={"output_dir": str(tmp_path / "batch"), "recursive": True},
        finalize_record=_finalize,
        progress_callback=updates.append,
    )

    assert len(result.records) == 2
    assert result.summary_json_path.exists()
    assert result.resolved_config_path.exists()
    assert result.runs_dir.exists()
    assert (result.batch_dir / "artifacts_manifest.json").exists()

    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["report_outputs"]["runs_dir"] == "runs"
    assert summary_payload["report_outputs"]["resolved_config"] == "resolved_config.json"

    manifest_payload = json.loads((result.batch_dir / "artifacts_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {str(row.get("path", "")) for row in manifest_payload["files"]}
    assert "resolved_config.json" in manifest_paths
    assert any(path.startswith("runs/") and path.endswith("/manifest.json") for path in manifest_paths)
    assert any(path.startswith("runs/") and path.endswith("/metrics.json") for path in manifest_paths)

    assert updates
    assert updates[-1].percent_complete == 100
    assert updates[-1].stage == "done"
    assert updates[-1].batch_dir == str(result.batch_dir)
