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
    assert (out_root / "logs" / "suite_events.jsonl").exists()

    payload = json.loads((out_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert payload.get("suite_event_log")
    assert payload.get("schema_version") == "microseg.hydride_benchmark_suite.v4"
    assert payload.get("rows")
    assert "run_events_log" in payload["rows"][0]
    assert "train_log" in payload["rows"][0]
    assert "eval_log" in payload["rows"][0]
    assert "mean_train_epoch_seconds" in payload["rows"][0]
    assert "mean_validation_epoch_seconds" in payload["rows"][0]
    assert "mean_epoch_runtime_seconds" in payload["rows"][0]


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
    rc, pid = suite._run_cmd(
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
    assert pid is not None
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
    rc, pid = suite._run_cmd(
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
    assert pid is not None
    text = log_path.read_text(encoding="utf-8")
    assert "start" in text
    assert "[watchdog] idle_timeout triggered" in text


def test_phase20_suite_reorders_experiments_transformer_first_unet_last(tmp_path: Path) -> None:
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
            {"name": "unet_binary", "train_config": "configs/hydride/train.unet_binary.baseline.yml"},
            {"name": "smp_fpn_resnet101", "train_config": "configs/hydride/train.smp_fpn_resnet101_scratch.yml"},
            {
                "name": "smp_deeplabv3plus_resnet101",
                "train_config": "configs/hydride/train.smp_deeplabv3plus_resnet101_scratch.yml",
            },
            {"name": "hf_segformer_b0", "train_config": "configs/hydride/train.hf_segformer_b0_scratch.yml"},
        ],
    }
    cfg_path = tmp_path / "suite_order.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg_path), "--dry-run"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((tmp_path / "suite" / "benchmark_summary.json").read_text(encoding="utf-8"))
    observed = [str(row.get("model")) for row in payload.get("rows", [])]
    assert observed == [
        "hf_segformer_b0",
        "smp_deeplabv3plus_resnet101",
        "smp_fpn_resnet101",
        "unet_binary",
    ]
    families = [str(row.get("execution_family")) for row in payload.get("rows", [])]
    assert families == ["transformer", "deeplab", "advanced", "unet"]


def test_phase20_single_seed_override_uses_first_seed_only(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    cfg = {
        "dataset_dir": "outputs/prepared_dataset_hydride_v1",
        "output_root": str(tmp_path / "suite"),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42, 43, 44],
        "benchmark_mode": False,
        "experiments": [
            {"name": "unet_binary", "train_config": "configs/hydride/train.unet_binary.baseline.yml"},
            {"name": "hf_segformer_b0", "train_config": "configs/hydride/train.hf_segformer_b0_scratch.yml"},
        ],
    }
    cfg_path = tmp_path / "suite_single_seed.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg_path), "--dry-run", "--single-seed"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((tmp_path / "suite" / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert payload.get("configured_seeds") == [42, 43, 44]
    assert payload.get("effective_seeds") == [42]
    assert payload.get("single_seed_override") is True
    rows = payload.get("rows", [])
    assert len(rows) == 2
    assert {int(row.get("seed")) for row in rows} == {42}


def test_phase20_suite_template_watchdog_defaults_three_hours() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [
        "configs/hydride/benchmark_suite.top5.yml",
        "configs/hydride/benchmark_suite.top5_local_pretrained.yml",
        "configs/hydride/benchmark_suite.top5_scratch.debug.yml",
        "configs/hydride/benchmark_suite.top5_local_pretrained.debug.yml",
        "configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.debug.yml",
        "configs/hydride/benchmark_suite.top5_scratch.realdata.template.yml",
        "configs/hydride/benchmark_suite.top5_local_pretrained.realdata.template.yml",
        "configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml",
    ]
    for rel_path in config_paths:
        cfg = yaml.safe_load((repo_root / rel_path).read_text(encoding="utf-8"))
        assert int(cfg.get("command_idle_timeout_seconds", 0)) == 10800
        assert int(cfg.get("command_wall_timeout_seconds", 0)) == 10800


def test_phase20_gpu_discovery_prefers_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7")
    info = suite._discover_visible_gpus()
    assert info.count == 3
    assert info.worker_gpu_ids == ["2", "5", "7"]
    assert info.source == "cuda_visible_devices"


def test_phase20_scheduler_parallel_summary_fields(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "hydride_benchmark_suite.py"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    cfg = {
        "dataset_dir": "outputs/prepared_dataset_hydride_v1",
        "output_root": str(tmp_path / "suite"),
        "eval_config": "configs/hydride/evaluate.hydride.yml",
        "eval_split": "test",
        "python_executable": sys.executable,
        "seeds": [42, 43],
        "benchmark_mode": False,
        "experiments": [
            {"name": "unet_binary", "train_config": "configs/hydride/train.unet_binary.baseline.yml"},
        ],
    }
    cfg_path = tmp_path / "suite_parallel.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(cfg_path),
            "--dry-run",
            "--max-parallel-gpus",
            "auto",
            "--parallel-jobs",
            "auto",
            "--failure-policy",
            "continue",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((tmp_path / "suite" / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert payload.get("scheduler_mode") == "parallel"
    assert int(payload.get("worker_count", 0)) == 2
    assert payload.get("visible_gpus") == ["0", "1"]
    assert payload.get("failure_policy") == "continue"
    assert (tmp_path / "suite" / "subjobs" / "unet_binary_seed42" / "metadata.json").exists()
    assert (tmp_path / "suite" / "subjobs" / "unet_binary_seed42" / "stdout.log").exists()
