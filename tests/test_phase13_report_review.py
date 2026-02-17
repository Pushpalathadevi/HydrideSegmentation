"""Phase 13 tests for run-report review and workflow profile persistence."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.app.report_review import compare_run_reports, summarize_run_report
from src.microseg.app.workflow_profiles import read_workflow_profile, write_workflow_profile


def test_phase13_summarize_and_compare_eval_reports(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    payload_a = {
        "schema_version": "microseg.pixel_eval.v2",
        "backend": "torch_pixel",
        "runtime_device": "cpu",
        "runtime_seconds": 10.0,
        "runtime_human": "10s",
        "config_sha256": "abc",
        "samples_evaluated": 12,
        "metrics": {"pixel_accuracy": 0.91, "macro_f1": 0.83, "mean_iou": 0.77},
        "tracked_samples": [{"sample_name": "x.png"}],
    }
    payload_b = {
        "schema_version": "microseg.pixel_eval.v2",
        "backend": "torch_pixel",
        "runtime_device": "cpu",
        "runtime_seconds": 11.0,
        "runtime_human": "11s",
        "config_sha256": "abc",
        "samples_evaluated": 12,
        "metrics": {"pixel_accuracy": 0.93, "macro_f1": 0.85, "mean_iou": 0.79},
        "tracked_samples": [{"sample_name": "x.png"}, {"sample_name": "y.png"}],
    }
    a.write_text(json.dumps(payload_a), encoding="utf-8")
    b.write_text(json.dumps(payload_b), encoding="utf-8")

    s1 = summarize_run_report(a)
    s2 = summarize_run_report(b)
    assert s1.report_kind == "evaluation"
    assert s1.metrics["pixel_accuracy"] == 0.91
    assert s2.tracked_samples == 2

    cmp_payload = compare_run_reports(s1, s2)
    assert cmp_payload["same_schema"] is True
    rows = {row["metric"]: row for row in cmp_payload["rows"]}
    assert rows["pixel_accuracy"]["delta"] > 0
    assert rows["macro_f1"]["delta"] > 0


def test_phase13_summarize_training_report(tmp_path: Path) -> None:
    report = tmp_path / "train_report.json"
    payload = {
        "schema_version": "microseg.training_report.v1",
        "backend": "unet_binary",
        "status": "completed",
        "runtime_seconds": 123.0,
        "runtime_human": "2m03s",
        "device": "cpu",
        "config_sha256": "cfg",
        "best_val_loss": 0.25,
        "progress": {"epochs_completed": 3},
        "history": [
            {"epoch": 1, "train_loss": 0.8, "train_iou": 0.2, "val_loss": 0.7, "val_iou": 0.25},
            {"epoch": 2, "train_loss": 0.6, "train_iou": 0.3, "val_loss": 0.5, "val_iou": 0.35},
            {"epoch": 3, "train_loss": 0.4, "train_iou": 0.5, "val_loss": 0.3, "val_iou": 0.6},
        ],
        "latest_tracked_samples": [{"sample_name": "val_001.png"}],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_run_report(report)
    assert summary.report_kind == "training"
    assert summary.metrics["best_val_loss"] == 0.25
    assert summary.metrics["val_iou"] == 0.6
    assert summary.tracked_samples == 1


def test_phase13_workflow_profile_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "profile.yml"
    out = write_workflow_profile(
        path,
        scope="dataset_prepare",
        values={
            "dataset_dir": "data",
            "output_dir": "outputs/prepared_dataset",
            "split_strategy": "leakage_aware",
        },
    )
    assert out.exists()
    loaded = read_workflow_profile(path)
    assert loaded["schema_version"] == "microseg.workflow_profile.v1"
    assert loaded["scope"] == "dataset_prepare"
    assert loaded["values"]["split_strategy"] == "leakage_aware"

