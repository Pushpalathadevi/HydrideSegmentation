"""Phase 20 tests for hydride benchmark suite orchestration script."""

from __future__ import annotations

import importlib.util
import json
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import sys

import yaml


def _load_suite_module() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "hydride_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("hydride_benchmark_suite_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load hydride benchmark suite module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite = _load_suite_module()


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


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
        "benchmark_mode": False,
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
    assert (out_root / "summary.json").exists()
    assert (out_root / "summary.html").exists()


def test_phase20_benchmark_mode_autogenerates_missing_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    dataset = tmp_path / "dataset"
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    _write_png(dataset / "train" / "images" / "s1.png", img)
    _write_png(dataset / "train" / "masks" / "s1.png", mask)
    _write_png(dataset / "val" / "images" / "s2.png", img)
    _write_png(dataset / "val" / "masks" / "s2.png", mask)
    _write_png(dataset / "test" / "images" / "s3.png", img)
    _write_png(dataset / "test" / "masks" / "s3.png", mask)

    cfg = {
        "dataset_dir": str(dataset),
        "output_root": str(tmp_path / "suite"),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42],
        "benchmark_mode": True,
        "experiments": [
            {
                "name": "unet_binary",
                "train_config": "configs/hydride/train.unet_binary.baseline.yml",
            }
        ],
    }
    cfg_path = tmp_path / "suite_manifest.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg_path), "--dry-run"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (dataset / "dataset_manifest.json").exists()


def test_phase20_missing_pretrained_weights_skips_and_continues(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    train_cfg = tmp_path / "train_local_missing.yml"
    train_cfg.write_text(
        yaml.safe_dump(
            {
                "backend": "unet_binary",
                "model_architecture": "unet_binary",
                "pretrained_init_mode": "local",
                "pretrained_model_id": "missing_model_id",
                "pretrained_registry_path": "pre_trained_weights/registry.json",
                "epochs": 1,
                "batch_size": 1,
                "enable_gpu": False,
                "device_policy": "cpu",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = {
        "dataset_dir": "outputs/prepared_dataset_hydride_v1",
        "output_root": str(tmp_path / "suite"),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42],
        "benchmark_mode": False,
        "experiments": [
            {"name": "local_missing", "train_config": str(train_cfg)},
            {"name": "scratch_ok", "train_config": "configs/hydride/train.unet_binary.baseline.yml"},
        ],
    }
    cfg_path = tmp_path / "suite_pretrained_missing.yml"
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
    summary = json.loads((out_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    statuses = {str(row.get("model")): str(row.get("status")) for row in summary.get("rows", [])}
    assert statuses.get("local_missing") == "pretrained_missing"
    assert statuses.get("scratch_ok") == "ok"
    assert summary.get("failure_count", 0) >= 1

    skip_log = out_root / "logs" / "local_missing_seed42" / "train.log"
    assert skip_log.exists()
    text = skip_log.read_text(encoding="utf-8")
    assert "download_pretrained_weights.py" in text
    assert "validate-pretrained" in text


def test_phase20_run_cmd_streams_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys,time; "
            "print('first-line', flush=True); "
            "time.sleep(0.1); "
            "print('second-line', flush=True)"
        ),
    ]
    rc = suite._run_cmd(
        cmd,
        log_path,
        dry_run=False,
        run_label="unit:stream",
        idle_timeout_seconds=None,
        wall_timeout_seconds=None,
        terminate_grace_seconds=1.0,
        poll_interval_seconds=0.1,
    )
    assert rc == 0
    text = log_path.read_text(encoding="utf-8")
    assert "first-line" in text
    assert "second-line" in text


def test_phase20_run_cmd_idle_watchdog_timeout(tmp_path: Path) -> None:
    log_path = tmp_path / "train_timeout.log"
    cmd = [
        sys.executable,
        "-c",
        (
            "import time; "
            "print('start', flush=True); "
            "time.sleep(2.0)"
        ),
    ]
    rc = suite._run_cmd(
        cmd,
        log_path,
        dry_run=False,
        run_label="unit:watchdog",
        idle_timeout_seconds=0.4,
        wall_timeout_seconds=None,
        terminate_grace_seconds=0.5,
        poll_interval_seconds=0.1,
    )
    assert rc == 124
    text = log_path.read_text(encoding="utf-8")
    assert "start" in text
    assert "[watchdog] idle_timeout triggered" in text
