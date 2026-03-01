"""Phase 21 tests for enriched benchmark dashboard and summary artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


def test_phase21_benchmark_dashboard_includes_curves_and_training_stats(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    output_root = tmp_path / "suite"
    run_tag = "unet_binary_seed42"
    run_dir = output_root / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "microseg.training_report.v1",
        "status": "completed",
        "runtime_seconds": 12.0,
        "runtime_human": "00:00:12",
        "best_val_loss": 0.25,
        "progress": {"epochs_total": 2, "epochs_completed": 2},
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "train_accuracy": 0.74,
                "train_iou": 0.55,
                "val_loss": 0.7,
                "val_accuracy": 0.71,
                "val_iou": 0.52,
                "train_epoch_seconds": 3.0,
                "validation_epoch_seconds": 1.5,
                "epoch_runtime_seconds": 5.0,
                "tracked_samples": [
                    {
                        "sample_name": "val_a.png",
                        "iou": 0.41,
                        "panel": "eval_samples/epoch_001/val_a_panel.png",
                    }
                ],
            },
            {
                "epoch": 2,
                "train_loss": 0.6,
                "train_accuracy": 0.79,
                "train_iou": 0.61,
                "val_loss": 0.5,
                "val_accuracy": 0.76,
                "val_iou": 0.58,
                "train_epoch_seconds": 2.2,
                "validation_epoch_seconds": 1.1,
                "epoch_runtime_seconds": 4.0,
                "tracked_samples": [
                    {
                        "sample_name": "val_a.png",
                        "iou": 0.57,
                        "panel": "eval_samples/epoch_002/val_a_panel.png",
                    }
                ],
            },
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    cfg = {
        "dataset_dir": "outputs/prepared_dataset_hydride_v1",
        "output_root": str(output_root),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42],
        "benchmark_mode": False,
        "experiments": [
            {
                "name": "unet_binary",
                "train_config": "configs/hydride/train.unet_binary.baseline.yml",
            }
        ],
    }
    cfg_path = tmp_path / "suite.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg_path), "--skip-train", "--skip-eval"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    summary_path = output_root / "benchmark_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["training_runtime_seconds"] == 12.0
    assert row["last_val_accuracy"] == 0.76
    assert row["mean_train_epoch_seconds"] == pytest.approx(2.6)
    assert row["mean_validation_epoch_seconds"] == pytest.approx(1.3)
    assert row["mean_epoch_runtime_seconds"] == pytest.approx(4.5)
    assert row["loss_curve_png"]
    assert row["accuracy_curve_png"]
    assert row["iou_curve_png"]
    assert row["tracked_sample_evolution_count"] == 1
    assert Path(row["loss_curve_png"]).exists()
    assert Path(row["accuracy_curve_png"]).exists()
    assert Path(row["iou_curve_png"]).exists()
    assert (output_root / "summary.json").exists()
    assert (output_root / "summary.html").exists()

    dashboard_text = (output_root / "benchmark_dashboard.html").read_text(encoding="utf-8")
    assert "Training Curve Gallery" in dashboard_text
    assert "Accuracy vs Epoch" in dashboard_text
    assert "Tracked Sample Evolution" in dashboard_text
    assert "Last Val Accuracy" in dashboard_text
    assert "Last Train Loss" in dashboard_text
    assert "Eval Mean IoU" in dashboard_text
    assert "Train Epoch Time" in dashboard_text

    with (output_root / "benchmark_aggregate.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "mean_last_val_accuracy" in rows[0]
    assert "mean_train_epoch_seconds" in rows[0]
    assert "mean_cohen_kappa" in rows[0]
    assert "quality_score" in rows[0]
