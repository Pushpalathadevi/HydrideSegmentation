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

    wide_img = np.zeros((32, 240, 3), dtype=np.uint8)
    wide_mask = np.zeros((32, 240), dtype=np.uint8)
    wide_img[:, 100:180, :] = 250
    wide_mask[:, 100:180] = 1
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "val_wide.png", wide_img, wide_mask)
    return ds


def test_phase7_unet_training_writes_reports_and_tracking_artifacts(tmp_path: Path, caplog) -> None:
    ds = _dataset(tmp_path)
    out = tmp_path / "training"

    caplog.set_level("INFO", logger="microseg.training.unet_binary")

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
            val_tracking_fixed_samples=("val_000.png", "val_wide.png"),
            val_tracking_seed=5,
            write_html_report=True,
            progress_log_interval_pct=50,
            input_hw=(32, 32),
            tracking_max_vis_width=128,
            tracking_max_vis_height=64,
        )
    )

    assert result["status"] in {"completed", "interrupted"}
    assert Path(result["report_path"]).exists()
    assert Path(result["html_report_path"]).exists()
    assert (out / "epoch_history.jsonl").exists()
    assert (out / "eval_samples" / "epoch_001").exists()
    assert any(p.name.startswith("val_000_") for p in (out / "eval_samples" / "epoch_001").glob("*.png"))
    wide_panel = out / "eval_samples" / "epoch_001" / "val_wide_panel.png"
    assert wide_panel.exists()
    panel_arr = np.asarray(Image.open(wide_panel).convert("RGB"), dtype=np.uint8)
    assert panel_arr.shape[0] <= 64
    assert panel_arr.shape[1] <= 128

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.training_report.v1"
    assert "history" in payload
    assert payload["history"]
    first = payload["history"][0]
    assert "train_accuracy" in first
    assert "val_accuracy" in first
    assert "train_epoch_seconds" in first
    assert "validation_epoch_seconds" in first
    assert float(first["train_epoch_seconds"]) >= 0.0
    assert float(first["validation_epoch_seconds"]) >= 0.0
    assert payload.get("mean_train_epoch_seconds") is not None
    assert payload.get("mean_validation_epoch_seconds") is not None

    html_text = (out / "training_report.html").read_text(encoding="utf-8")
    assert "Tracked Validation Samples By Epoch" in html_text
    assert "Epoch 1" in html_text
    assert "pixel accuracy" in html_text

    log_text = "\n".join(caplog.messages)
    assert "VAL_START" in log_text
    assert "VAL_PROGRESS" in log_text
    assert "VAL_END" in log_text
    assert "TRACK_EXPORT_START" in log_text
    assert "TRACK_EXPORT_END" in log_text
    assert "TRACK_SAMPLE_SHAPES" in log_text
    assert "TRACK_SAMPLE_FILE_WRITE_END" in log_text
    assert "EPOCH_HISTORY_WRITE_START" in log_text
    assert "EPOCH_HISTORY_WRITE_END" in log_text
    assert "CKPT_SAVE_START" in log_text
    assert "CKPT_SAVE_END" in log_text
    assert "REPORT_UPDATE_START" in log_text
    assert "REPORT_UPDATE_END" in log_text
