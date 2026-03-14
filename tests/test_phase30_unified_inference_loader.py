"""Phase 30 tests for unified architecture-aware inference loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydride_segmentation.inference import run_model
from hydride_segmentation.microseg_adapter import get_gui_model_options
from src.microseg.inference import discover_inference_references
from src.microseg.inference.trained_model_loader import load_reference_from_run_dir
from src.microseg.training.unet_binary import _build_binary_model


def _write_synthetic_run(root: Path, *, name: str, status: str = "ok") -> Path:
    import torch

    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_binary_model(
        architecture="unet_binary",
        base_channels=8,
        transformer_depth=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        segformer_patch_size=4,
    )
    ckpt_path = run_dir / "best_model.pt"
    torch.save(
        {
            "schema_version": "microseg.torch_unet_binary.v1",
            "model_state_dict": model.state_dict(),
            "model_architecture": "unet_binary",
            "backend": "unet_binary_local_pretrained",
            "config": {"model_architecture": "unet_binary", "model_base_channels": 8},
        },
        ckpt_path,
    )

    (run_dir / "report.json").write_text(
        json.dumps({"status": status, "model_path": "best_model.pt", "model_architecture": "unet_binary"}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "training_manifest.json").write_text(
        json.dumps({"model_path": "best_model.pt", "model_architecture": "unet_binary"}, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_phase30_discovery_rejects_failed_runs(tmp_path: Path) -> None:
    runs = tmp_path / "outputs" / "runs"
    _write_synthetic_run(runs, name="ok_run", status="ok")
    _write_synthetic_run(runs, name="failed_run", status="failed")

    refs, warnings = discover_inference_references(runs_root=runs, include_registry=False)
    ids = [ref.reference_id for ref in refs]

    assert "run::ok_run" in ids
    assert "run::failed_run" not in ids
    assert any("failed_run" in warn for warn in warnings)


def test_phase30_inference_run_dir_loads_and_predicts(tmp_path: Path) -> None:
    runs = tmp_path / "outputs" / "runs"
    run_dir = _write_synthetic_run(runs, name="unet_binary_local_pretrained_seed42", status="ok")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = 255
    img_path = tmp_path / "sample.png"
    Image.fromarray(image).save(img_path)

    ref = load_reference_from_run_dir(run_dir)
    assert ref.architecture == "unet_binary"

    rgb, mask = run_model(
        str(img_path),
        params={"run_dir": str(run_dir), "enable_gpu": False, "device_policy": "cpu"},
    )
    assert rgb.shape == image.shape
    assert mask.shape == image.shape[:2]


def test_phase30_gui_options_include_discovered_trained_models(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "outputs" / "runs"
    _write_synthetic_run(runs, name="gui_run_model", status="ok")
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    from src.microseg.inference import predictors as pred

    def _fake_discover(*, include_registry=True):  # noqa: ARG001
        return discover_inference_references(runs_root=runs, include_registry=False)

    monkeypatch.setattr(pred, "discover_inference_references", _fake_discover)

    options = get_gui_model_options()
    assert any("gui_run_model" in name for name in options)
