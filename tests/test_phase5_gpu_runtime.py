"""Phase 5 tests for GPU-compatible runtime with CPU fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.core import resolve_torch_device
from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator
from src.microseg.training import TorchPixelClassifierTrainer, TorchPixelTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _build_dataset(root: Path) -> Path:
    ds = root / "dataset"
    for i in range(2):
        img = np.zeros((24, 24, 3), dtype=np.uint8)
        msk = np.zeros((24, 24), dtype=np.uint8)
        img[:, 12:] = 230
        msk[:, 12:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"train_{i}.png", img, msk)

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    msk = np.zeros((24, 24), dtype=np.uint8)
    img[:12, :] = 220
    msk[:12, :] = 1
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "val_0.png", img, msk)
    return ds


def test_phase5_device_resolution_defaults_to_cpu_when_gpu_disabled() -> None:
    res = resolve_torch_device(enable_gpu=False, policy="auto")
    assert res.selected_device == "cpu"
    assert "disabled" in res.reason.lower()


def test_phase5_torch_train_and_eval_with_gpu_enabled_falls_back_on_cpu(tmp_path: Path) -> None:
    ds = _build_dataset(tmp_path)
    out = tmp_path / "training"

    trainer = TorchPixelClassifierTrainer()
    trained = trainer.train(
        TorchPixelTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            train_split="train",
            max_samples=1000,
            epochs=3,
            batch_size=256,
            learning_rate=0.05,
            seed=9,
            enable_gpu=True,
            device_policy="auto",
        )
    )

    model_path = Path(trained["model_path"])
    assert model_path.exists()

    report_path = tmp_path / "eval" / "report.json"
    payload = PixelModelEvaluator().evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(ds),
            model_path=str(model_path),
            split="val",
            output_path=str(report_path),
            enable_gpu=True,
            device_policy="auto",
        )
    )

    assert report_path.exists()
    assert payload["backend"] == "torch_pixel"
    assert payload["runtime_device"] in {"cpu", "cuda", "mps"}
    assert payload["metrics"]["pixel_accuracy"] >= 0.6
    assert payload["metrics"]["foreground_dice"] >= 0.0


def test_phase5_evaluate_accepts_pth_suffix_for_torch_checkpoints(tmp_path: Path) -> None:
    ds = _build_dataset(tmp_path)
    out = tmp_path / "training"

    trained = TorchPixelClassifierTrainer().train(
        TorchPixelTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            train_split="train",
            max_samples=800,
            epochs=2,
            batch_size=256,
            learning_rate=0.03,
            seed=11,
            enable_gpu=False,
            device_policy="cpu",
        )
    )

    model_pt = Path(trained["model_path"])
    model_pth = model_pt.with_suffix(".pth")
    model_pth.write_bytes(model_pt.read_bytes())

    report_path = tmp_path / "eval_pth" / "report.json"
    payload = PixelModelEvaluator().evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(ds),
            model_path=str(model_pth),
            split="val",
            output_path=str(report_path),
            enable_gpu=False,
            device_policy="cpu",
        )
    )
    assert report_path.exists()
    assert payload["backend"] == "torch_pixel"
