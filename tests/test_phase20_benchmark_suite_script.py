"""Phase 20 tests for hydride benchmark suite orchestration script."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


def test_phase20_benchmark_suite_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    cfg = {
        "dataset_dir": "outputs/prepared_dataset_hydride_v1",
        "output_root": str(tmp_path / "suite"),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42],
        "experiments": [
            {
                "name": "unet_binary",
                "train_config": "configs/hydride/train.unet_binary.baseline.yml",
            },
            {
                "name": "hf_segformer_b0",
                "train_config": "configs/hydride/train.hf_segformer_b0_scratch.yml",
            },
        ],
    }
    cfg_path = tmp_path / "suite.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg_path), "--dry-run"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_root = tmp_path / "suite"
    assert (out_root / "benchmark_summary.json").exists()
    assert (out_root / "benchmark_summary.csv").exists()
    assert (out_root / "benchmark_aggregate.csv").exists()
    assert (out_root / "benchmark_dashboard.html").exists()
