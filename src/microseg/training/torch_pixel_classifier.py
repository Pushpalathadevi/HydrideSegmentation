"""Torch-based baseline pixel classifier with optional GPU acceleration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.core import resolve_torch_device
from src.microseg.corrections.classes import to_index_mask


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"missing images/masks under split dir: {split_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*")):
        if not img.is_file():
            continue
        msk = masks_dir / img.name
        if msk.exists() and msk.is_file():
            pairs.append((img, msk))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found in {split_dir}")
    return pairs


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    return to_index_mask(np.asarray(Image.open(path).convert("L"), dtype=np.uint8))


def _build_samples(
    pairs: list[tuple[Path, Path]],
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_pair = max(1, int(max_samples // max(1, len(pairs))))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for img_path, mask_path in pairs:
        img = _load_rgb(img_path)
        mask = _load_mask(mask_path)
        if mask.shape != img.shape[:2]:
            raise ValueError(f"shape mismatch: {img_path} vs {mask_path}")

        feat = img.reshape(-1, 3).astype(np.float32) / 255.0
        lab = mask.reshape(-1).astype(np.int32)

        take = min(per_pair, feat.shape[0])
        idx = rng.choice(feat.shape[0], size=take, replace=False)
        xs.append(feat[idx])
        ys.append(lab[idx])

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    return x, y


@dataclass(frozen=True)
class TorchPixelTrainingConfig:
    """Configuration for torch-based pixel classifier training."""

    dataset_dir: str
    output_dir: str
    train_split: str = "train"
    max_samples: int = 250_000
    epochs: int = 8
    batch_size: int = 4096
    learning_rate: float = 1e-2
    seed: int = 42
    enable_gpu: bool = False
    device_policy: str = "cpu"


class TorchPixelClassifierTrainer:
    """Train and persist torch-based baseline pixel classifier."""

    def train(self, config: TorchPixelTrainingConfig) -> dict[str, Any]:
        import torch

        torch.manual_seed(int(config.seed))

        dataset_root = Path(config.dataset_dir)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = _collect_pairs(dataset_root / config.train_split)
        x_np, y_raw = _build_samples(pairs, max_samples=config.max_samples, seed=config.seed)

        class_values = np.unique(y_raw)
        if class_values.size < 2:
            raise ValueError("training labels must contain at least 2 classes")

        class_to_idx = {int(v): idx for idx, v in enumerate(class_values.tolist())}
        y_idx = np.array([class_to_idx[int(v)] for v in y_raw], dtype=np.int64)

        resolved = resolve_torch_device(enable_gpu=config.enable_gpu, policy=config.device_policy)
        device = resolved.selected_device

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_idx).to(device)

        model = torch.nn.Linear(3, int(class_values.size)).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

        n = x.shape[0]
        batch_size = max(1, int(config.batch_size))
        for _epoch in range(int(config.epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits = model(x[idx])
                loss = criterion(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        model_path = output_dir / "torch_pixel_classifier.pt"
        manifest_path = output_dir / "training_manifest.json"

        checkpoint = {
            "schema_version": "microseg.torch_pixel_classifier.v1",
            "created_utc": _utc_now(),
            "config": asdict(config),
            "class_values": [int(v) for v in class_values.tolist()],
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, model_path)

        manifest = {
            "schema_version": "microseg.training_manifest.v1",
            "backend": "torch_pixel",
            "created_utc": _utc_now(),
            "device": device,
            "device_reason": resolved.reason,
            "config": asdict(config),
            "train_pairs": len(pairs),
            "train_samples": int(x_np.shape[0]),
            "model_file": model_path.name,
            "classes": checkpoint["class_values"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "backend": "torch_pixel",
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "device": device,
            "classes": checkpoint["class_values"],
            "train_samples": int(x_np.shape[0]),
        }


def load_torch_pixel_classifier(
    model_path: str | Path,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> dict[str, Any]:
    """Load persisted torch pixel classifier checkpoint."""

    import torch

    resolved = resolve_torch_device(enable_gpu=enable_gpu, policy=device_policy)
    device = resolved.selected_device

    ckpt = torch.load(Path(model_path), map_location="cpu")
    classes = [int(v) for v in ckpt["class_values"]]
    model = torch.nn.Linear(3, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "class_values": np.array(classes, dtype=np.int32),
        "device": device,
        "reason": resolved.reason,
    }


def predict_index_mask_torch(image: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    """Predict indexed mask from RGB image using loaded torch bundle."""

    import torch

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB with shape (H, W, 3)")

    model = bundle["model"]
    class_values = bundle["class_values"]
    device = bundle["device"]

    feat = image.reshape(-1, 3).astype(np.float32) / 255.0
    x = torch.from_numpy(feat).to(device)
    with torch.no_grad():
        pred_idx = model(x).argmax(dim=1).cpu().numpy()

    pred_values = class_values[pred_idx]
    return pred_values.astype(np.uint8).reshape(image.shape[:2])


def infer_image_with_torch_pixel_classifier(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> dict[str, str]:
    """Run one-image inference and export indexed prediction mask."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    bundle = load_torch_pixel_classifier(model_path, enable_gpu=enable_gpu, device_policy=device_policy)
    pred = predict_index_mask_torch(image, bundle)

    stem = Path(image_path).stem
    out_img = output_root / f"{stem}_input.png"
    out_mask = output_root / f"{stem}_prediction_indexed.png"
    Image.fromarray(image).save(out_img)
    Image.fromarray(pred).save(out_mask)

    return {
        "input": str(out_img),
        "prediction_indexed": str(out_mask),
    }
