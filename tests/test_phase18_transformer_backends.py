"""Phase 18 tests for transformer-based segmentation backend variants."""

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


def _train_variant(ds: Path, out: Path, *, architecture: str, backend_label: str) -> str:
    result = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=13,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=1,
            checkpoint_every=1,
            model_architecture=architecture,
            model_base_channels=8,
            transformer_depth=1,
            transformer_num_heads=4,
            transformer_mlp_ratio=2.0,
            transformer_dropout=0.0,
            segformer_patch_size=4,
            backend_label=backend_label,
        )
    )
    return str(result["model_path"])


def test_phase18_transformer_backends_train_and_evaluate(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)

    for architecture, backend_label in [("transunet_tiny", "transunet_tiny"), ("segformer_mini", "segformer_mini")]:
        run_dir = tmp_path / backend_label
        model_path = _train_variant(ds, run_dir, architecture=architecture, backend_label=backend_label)
        assert Path(model_path).exists()

        report_path = tmp_path / "eval" / f"{backend_label}.json"
        payload = PixelModelEvaluator().evaluate(
            PixelEvaluationConfig(
                dataset_dir=str(ds),
                model_path=model_path,
                split="val",
                output_path=str(report_path),
                enable_gpu=False,
                device_policy="cpu",
            )
        )
        assert report_path.exists()
        assert payload["backend"] == backend_label
