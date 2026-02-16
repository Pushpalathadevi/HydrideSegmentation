"""Phase 7 tests for UNet reporting and validation sample tracking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.training import UNetBinaryTrainer, UNetBinaryTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _dataset(root: Path) -> Path:
    ds = root / "dataset"
    for i in range(2):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        msk = np.zeros((28, 28), dtype=np.uint8)
        img[:, 14:] = 220
        msk[:, 14:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"train_{i}.png", img, msk)

    img_a = np.zeros((28, 28, 3), dtype=np.uint8)
    m_a = np.zeros((28, 28), dtype=np.uint8)
    img_a[:14, :] = 230
    m_a[:14, :] = 1
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "val_000.png", img_a, m_a)

    img_b = np.zeros((28, 28, 3), dtype=np.uint8)
    m_b = np.zeros((28, 28), dtype=np.uint8)
    img_b[:, :14] = 230
    m_b[:, :14] = 1
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "val_111.png", img_b, m_b)
    return ds


def test_phase7_unet_training_writes_reports_and_tracking_artifacts(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    out = tmp_path / "training"

    result = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=7,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=2,
            checkpoint_every=1,
            val_tracking_samples=2,
            val_tracking_fixed_samples=("val_000.png",),
            val_tracking_seed=5,
            write_html_report=True,
            progress_log_interval_pct=50,
        )
    )

    assert result["status"] in {"completed", "interrupted"}
    assert Path(result["report_path"]).exists()
    assert Path(result["html_report_path"]).exists()
    assert (out / "epoch_history.jsonl").exists()
    assert (out / "eval_samples" / "epoch_001").exists()
    assert any(p.name.startswith("val_000_") for p in (out / "eval_samples" / "epoch_001").glob("*.png"))

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.training_report.v1"
    assert "history" in payload
    assert payload["history"]
