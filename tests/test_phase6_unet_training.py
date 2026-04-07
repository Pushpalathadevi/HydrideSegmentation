"""Phase 6 tests for UNet training backend with checkpoint resume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator
from src.microseg.training import UNetBinaryTrainer, UNetBinaryTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _dataset(root: Path) -> Path:
    ds = root / "dataset"
    for i in range(2):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        m = np.zeros((32, 32), dtype=np.uint8)
        img[:, 16:] = 220
        m[:, 16:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"t{i}.png", img, m)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    m = np.zeros((32, 32), dtype=np.uint8)
    img[:16, :] = 220
    m[:16, :] = 1
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "v0.png", img, m)
    return ds


def test_phase6_unet_training_and_resume(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    out = tmp_path / "training"

    trainer = UNetBinaryTrainer()
    first = trainer.train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            seed=7,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=2,
            checkpoint_every=1,
        )
    )

    best = Path(first["model_path"])
    last = out / "last_checkpoint.pt"
    assert best.exists()
    assert last.exists()
    assert (out / "training_manifest.json").exists()

    second = trainer.train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=3,
            batch_size=1,
            learning_rate=1e-3,
            seed=7,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=2,
            checkpoint_every=1,
            resume_checkpoint=str(last),
        )
    )
    assert Path(second["model_path"]).exists()


def test_phase6_unet_evaluation_path(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    out = tmp_path / "training"

    model_path = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=9,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=1,
            checkpoint_every=1,
        )
    )["model_path"]

    report = tmp_path / "eval" / "report.json"
    payload = PixelModelEvaluator().evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(ds),
            model_path=str(model_path),
            split="val",
            output_path=str(report),
            enable_gpu=True,
            device_policy="auto",
        )
    )
    assert report.exists()
    assert payload["backend"] == "unet_binary"
    assert payload["runtime_device"] in {"cpu", "cuda", "mps"}
