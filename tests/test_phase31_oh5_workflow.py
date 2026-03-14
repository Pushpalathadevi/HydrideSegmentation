"""Tests for raw `.oh5` extraction and workflow orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.microseg.data_preparation.oh5 import Oh5ExtractionConfig, extract_oh5_dataset
from src.microseg.workflows.phaseid_benchmark import (
    PhaseIdBenchmarkWorkflowConfig,
    run_phaseid_benchmark_workflow,
)


h5py = pytest.importorskip("h5py")


def _write_oh5(path: Path) -> None:
    import numpy as np

    with h5py.File(path, "w") as handle:
        handle.create_dataset("/Data/Image", data=np.arange(36, dtype=np.float32).reshape(6, 6))
        phase = np.zeros((6, 6), dtype=np.uint8)
        phase[1:3, 1:4] = 7
        handle.create_dataset("/Data/PhaseId", data=phase)


def test_extract_oh5_dataset_creates_png_pairs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_oh5(raw_dir / "sample.oh5")

    result = extract_oh5_dataset(
        Oh5ExtractionConfig(
            input_dir=str(raw_dir),
            output_dir=str(tmp_path / "extracted"),
            image_dataset_candidates=("/Data/Image",),
            phase_dataset_candidates=("/Data/PhaseId",),
            foreground_phase_ids=(7,),
        )
    )

    assert result.sample_count == 1
    assert (Path(result.images_dir) / "sample.png").exists()
    assert (Path(result.masks_dir) / "sample.png").exists()
    payload = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["samples"][0]["unique_phase_ids"] == [0, 7]


def test_phaseid_benchmark_workflow_runs_all_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_oh5(raw_dir / "sample.oh5")

    benchmark_template = tmp_path / "benchmark_template.yml"
    benchmark_template.write_text(
        "\n".join(
            [
                "dataset_dir: placeholder",
                f"output_root: {str((tmp_path / 'benchmarks').resolve())}",
                "eval_split: val",
                "effective_seeds: [42]",
                "experiments:",
                "  - name: unet_binary_debug",
                "    train_config: configs/hydride/train.unet_binary.baseline.yml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], cwd: str, text: bool, capture_output: bool, check: bool) -> object:
        calls.append(cmd)
        command_text = " ".join(cmd)
        if "hydride_benchmark_suite.py" in command_text:
            output_root = tmp_path / "workflow" / "benchmarks"
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "benchmark_dashboard.html").write_text("<html></html>", encoding="utf-8")
            (output_root / "benchmark_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_dir": str((tmp_path / "workflow" / "prepared_dataset").resolve()),
                        "eval_split": "val",
                        "effective_seeds": [42],
                        "rows": [{"model": "unet_binary_debug", "seed": 42, "status": "ok"}],
                        "aggregate": [
                            {
                                "model": "unet_binary_debug",
                                "rank_quality": 1,
                                "mean_mean_iou": 0.75,
                                "mean_macro_f1": 0.81,
                                "mean_foreground_dice": 0.79,
                                "mean_total_runtime_seconds": 12.0,
                                "runs": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return type("Proc", (), {"returncode": 0, "stdout": "suite ok", "stderr": ""})()
        if "generate_benchmark_lab_meeting_ppt.py" in command_text:
            deck_dir = tmp_path / "workflow" / "lab_meeting_deck"
            deck_dir.mkdir(parents=True, exist_ok=True)
            (deck_dir / "benchmark_summary_lab_meeting.pptx").write_text("pptx", encoding="utf-8")
            return type("Proc", (), {"returncode": 0, "stdout": "deck ok", "stderr": ""})()
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr("src.microseg.workflows.phaseid_benchmark.subprocess.run", _fake_run)

    report = run_phaseid_benchmark_workflow(
        PhaseIdBenchmarkWorkflowConfig(
            raw_input_dir=str(raw_dir),
            working_dir=str(tmp_path / "workflow"),
            benchmark_template=str(benchmark_template),
            extraction=Oh5ExtractionConfig(
                input_dir=str(raw_dir),
                output_dir=str(tmp_path / "workflow" / "oh5_extracted"),
                image_dataset_candidates=("/Data/Image",),
                phase_dataset_candidates=("/Data/PhaseId",),
                foreground_phase_ids=(7,),
            ),
        )
    )

    assert len(calls) == 2
    assert Path(report["prepared_dataset_dir"]).exists()
    assert Path(report["dataset_qa_report_path"]).exists()
    assert Path(report["benchmark_summary_json"]).exists()
    assert Path(report["report_path"]).exists()
