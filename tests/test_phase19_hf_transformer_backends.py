"""Phase 19 tests for Hugging Face transformer segmentation backends (scratch init)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator
from src.microseg.training import UNetBinaryTrainer, UNetBinaryTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _dataset(root: Path) -> Path:
    ds = root / "dataset"
    for i in range(3):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        m = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 220
        m[:, 32:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"t{i}.png", img, m)

    for i in range(2):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        m = np.zeros((64, 64), dtype=np.uint8)
        img[:32, :] = 220
        m[:32, :] = 1
        _write_pair(ds / "val" / "images", ds / "val" / "masks", f"v{i}.png", img, m)
    return ds


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_phase19_hf_segformer_scratch_train_eval(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    ds = _dataset(tmp_path)
    out = tmp_path / "training_hf_b0"

    model_path = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=19,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=1,
            checkpoint_every=1,
            model_architecture="hf_segformer_b0",
            backend_label="hf_segformer_b0",
        )
    )["model_path"]

    report = tmp_path / "eval" / "hf_b0_report.json"
    payload = PixelModelEvaluator().evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(ds),
            model_path=str(model_path),
            split="val",
            output_path=str(report),
            enable_gpu=False,
            device_policy="cpu",
        )
    )
    assert report.exists()
    assert payload["backend"] == "hf_segformer_b0"
