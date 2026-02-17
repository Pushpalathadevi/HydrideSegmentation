"""Phase 22 tests for local pretrained initialization and offline registry validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.microseg.plugins import (
    resolve_bundle_paths,
    resolve_pretrained_record,
    validate_pretrained_registry,
)
from src.microseg.training import UNetBinaryTrainer, UNetBinaryTrainingConfig


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _tiny_dataset(root: Path) -> Path:
    ds = root / "dataset"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    m = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 200
    m[:, 32:] = 1
    _write_pair(ds / "train" / "images", ds / "train" / "masks", "train_0.png", img, m)
    _write_pair(ds / "train" / "images", ds / "train" / "masks", "train_1.png", img, m)
    _write_pair(ds / "val" / "images", ds / "val" / "masks", "val_0.png", img, m)
    return ds


def test_phase22_pretrained_registry_validation_and_resolution(tmp_path: Path) -> None:
    root = tmp_path / "pre_trained_weights"
    bundle = root / "toy_model"
    bundle.mkdir(parents=True, exist_ok=True)
    weights = bundle / "weights.pt"
    weights.write_bytes(b"toy-weights")
    metadata = bundle / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "microseg.pretrained_weights_registry.v1",
                "models": [
                    {
                        "model_id": "toy_model",
                        "architecture": "smp_unet_resnet18",
                        "framework": "segmentation_models_pytorch",
                        "source": "toy",
                        "source_revision": "r1",
                        "bundle_dir": "toy_model",
                        "weights_path": "weights.pt",
                        "weights_format": "torch_state_dict",
                        "metadata_path": "metadata.json",
                        "files": [
                            {
                                "path": "weights.pt",
                                "sha256": _sha256(weights),
                            }
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = validate_pretrained_registry(registry, verify_sha256=True)
    assert report.ok is True
    assert not report.errors

    rec = resolve_pretrained_record(model_id="toy_model", registry_path=registry)
    bundle_abs, weights_abs, metadata_abs = resolve_bundle_paths(rec, registry_path=registry)
    assert bundle_abs == bundle.resolve()
    assert weights_abs == weights.resolve()
    assert metadata_abs == metadata.resolve()


def test_phase22_pretrained_registry_detects_checksum_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "pre_trained_weights"
    bundle = root / "toy_bad"
    bundle.mkdir(parents=True, exist_ok=True)
    weights = bundle / "weights.pt"
    weights.write_bytes(b"toy-weights")

    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "microseg.pretrained_weights_registry.v1",
                "models": [
                    {
                        "model_id": "toy_bad",
                        "architecture": "smp_unet_resnet18",
                        "framework": "segmentation_models_pytorch",
                        "source": "toy",
                        "source_revision": "r1",
                        "bundle_dir": "toy_bad",
                        "weights_path": "weights.pt",
                        "weights_format": "torch_state_dict",
                        "metadata_path": "",
                        "files": [
                            {
                                "path": "weights.pt",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = validate_pretrained_registry(registry, verify_sha256=True)
    assert report.ok is False
    assert any("checksum mismatch" in err for err in report.errors)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_phase22_hf_local_pretrained_init(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    ds = _tiny_dataset(tmp_path)
    out = tmp_path / "train_hf"

    bundle_root = tmp_path / "pre_trained_weights"
    bundle = bundle_root / "toy_hf"
    model_dir = bundle / "hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = SegformerForSemanticSegmentation(
        SegformerConfig(
            num_labels=2,
            num_channels=3,
            hidden_sizes=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            num_attention_heads=[1, 1, 2, 4],
            decoder_hidden_size=64,
            sr_ratios=[8, 4, 2, 1],
            mlp_ratios=[2, 2, 2, 2],
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            reshape_last_stage=True,
        )
    )
    model.save_pretrained(model_dir)

    registry = bundle_root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "microseg.pretrained_weights_registry.v1",
                "models": [
                    {
                        "model_id": "toy_hf",
                        "architecture": "hf_segformer_b0",
                        "framework": "transformers",
                        "source": "toy",
                        "source_revision": "r1",
                        "bundle_dir": "toy_hf",
                        "weights_path": "hf_model",
                        "weights_format": "hf_model_dir",
                        "metadata_path": "",
                        "files": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=22,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=1,
            checkpoint_every=1,
            model_architecture="hf_segformer_b0",
            backend_label="hf_segformer_b0_local",
            pretrained_init_mode="local",
            pretrained_model_id="toy_hf",
            pretrained_registry_path=str(registry),
            pretrained_verify_sha256=False,
        )
    )
    assert result["model_initialization"] == "local_pretrained"


def test_phase22_smp_local_pretrained_init(tmp_path: Path) -> None:
    pytest.importorskip("segmentation_models_pytorch")
    import torch
    import segmentation_models_pytorch as smp

    ds = _tiny_dataset(tmp_path)
    out = tmp_path / "train_smp"

    bundle_root = tmp_path / "pre_trained_weights"
    bundle = bundle_root / "toy_smp"
    bundle.mkdir(parents=True, exist_ok=True)
    weights_path = bundle / "weights.pt"
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    torch.save(model.state_dict(), weights_path)

    registry = bundle_root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "microseg.pretrained_weights_registry.v1",
                "models": [
                    {
                        "model_id": "toy_smp",
                        "architecture": "smp_unet_resnet18",
                        "framework": "segmentation_models_pytorch",
                        "source": "toy",
                        "source_revision": "r1",
                        "bundle_dir": "toy_smp",
                        "weights_path": "weights.pt",
                        "weights_format": "torch_state_dict",
                        "metadata_path": "",
                        "files": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = UNetBinaryTrainer().train(
        UNetBinaryTrainingConfig(
            dataset_dir=str(ds),
            output_dir=str(out),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=22,
            enable_gpu=False,
            device_policy="cpu",
            early_stopping_patience=1,
            checkpoint_every=1,
            model_architecture="smp_unet_resnet18",
            backend_label="smp_unet_resnet18_local",
            pretrained_init_mode="local",
            pretrained_model_id="toy_smp",
            pretrained_registry_path=str(registry),
            pretrained_verify_sha256=False,
        )
    )
    assert result["model_initialization"] == "local_pretrained"
