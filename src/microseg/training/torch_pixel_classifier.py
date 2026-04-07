"""Torch-based baseline pixel classifier with optional GPU acceleration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.core import resolve_torch_device
from src.microseg.corrections.classes import binary_remapped_foreground_values, normalize_binary_index_mask


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ensure_logger() -> logging.Logger:
    logger = logging.getLogger("microseg.training.torch_pixel")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


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


def _load_mask(path: Path, *, binary_mask_normalization: str) -> np.ndarray:
    return normalize_binary_index_mask(
        np.asarray(Image.open(path).convert("L"), dtype=np.uint8),
        mode=str(binary_mask_normalization),
    )


def _warn_binary_mask_remap_values(
    pairs: list[tuple[Path, Path]],
    *,
    binary_mask_normalization: str,
    logger: logging.Logger,
) -> None:
    mode = str(binary_mask_normalization).strip().lower()
    if mode not in {"two_value_zero_background", "nonzero_foreground"}:
        return
    remapped: set[int] = set()
    for _img_path, mask_path in pairs:
        arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        remapped.update(binary_remapped_foreground_values(arr, mode=mode))
    if remapped:
        logger.warning(
            "binary_mask_normalization=%s remapped non-zero mask values %s to foreground class 1 during training.",
            mode,
            sorted(remapped),
        )


def _build_samples(
    pairs: list[tuple[Path, Path]],
    *,
    max_samples: int,
    seed: int,
    binary_mask_normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_pair = max(1, int(max_samples // max(1, len(pairs))))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for img_path, mask_path in pairs:
        img = _load_rgb(img_path)
        mask = _load_mask(mask_path, binary_mask_normalization=binary_mask_normalization)
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
    binary_mask_normalization: str = "off"


class TorchPixelClassifierTrainer:
    """Train and persist torch-based baseline pixel classifier."""

    def train(self, config: TorchPixelTrainingConfig) -> dict[str, Any]:
        import torch

        logger = _ensure_logger()
        run_start = time.perf_counter()
        torch.manual_seed(int(config.seed))

        dataset_root = Path(config.dataset_dir)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = _collect_pairs(dataset_root / config.train_split)
        _warn_binary_mask_remap_values(
            pairs,
            binary_mask_normalization=str(config.binary_mask_normalization),
            logger=logger,
        )
        x_np, y_raw = _build_samples(
            pairs,
            max_samples=config.max_samples,
            seed=config.seed,
            binary_mask_normalization=config.binary_mask_normalization,
        )

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
        logger.info("VAL_START | backend=torch_pixel epoch=0 note=not_applicable")
        logger.info("VAL_END | backend=torch_pixel epoch=0 note=not_applicable")
        logger.info("METRIC_REDUCTION_START | backend=torch_pixel epoch=0")
        logger.info("METRIC_REDUCTION_END | backend=torch_pixel epoch=0")
        logger.info("TRACK_EXPORT_START | backend=torch_pixel epoch=0 selected=0 note=unsupported")
        logger.info("TRACK_EXPORT_END | backend=torch_pixel epoch=0 selected=0 elapsed=00:00:00")

        epoch_train_seconds: list[float] = []
        for epoch_idx in range(int(config.epochs)):
            logger.info("EPOCH_TRAIN_START | backend=torch_pixel epoch=%d/%d", epoch_idx + 1, int(config.epochs))
            epoch_start = time.perf_counter()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits = model(x[idx])
                loss = criterion(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_train_seconds.append(float(epoch_elapsed))
            logger.info(
                "EPOCH_TRAIN_END | backend=torch_pixel epoch=%d/%d elapsed=%s",
                epoch_idx + 1,
                int(config.epochs),
                _format_seconds(epoch_elapsed),
            )

        model_path = output_dir / "torch_pixel_classifier.pt"
        manifest_path = output_dir / "training_manifest.json"
        training_runtime_seconds = time.perf_counter() - run_start
        mean_train_epoch_seconds = (
            float(sum(epoch_train_seconds) / len(epoch_train_seconds)) if epoch_train_seconds else None
        )

        checkpoint = {
            "schema_version": "microseg.torch_pixel_classifier.v1",
            "created_utc": _utc_now(),
            "config": asdict(config),
            "class_values": [int(v) for v in class_values.tolist()],
            "state_dict": model.state_dict(),
        }
        logger.info("CKPT_SAVE_START | backend=torch_pixel path=%s", model_path)
        torch.save(checkpoint, model_path)
        logger.info("CKPT_SAVE_END | backend=torch_pixel path=%s size_bytes=%d", model_path, int(model_path.stat().st_size))

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
            "training_runtime_seconds": float(training_runtime_seconds),
            "training_runtime_human": _format_seconds(training_runtime_seconds),
            "training_epoch_count": int(len(epoch_train_seconds)),
            "mean_train_epoch_seconds": mean_train_epoch_seconds,
            "mean_validation_epoch_seconds": None,
            "mean_epoch_runtime_seconds": mean_train_epoch_seconds,
            "validation_timing_supported": False,
        }
        logger.info("REPORT_UPDATE_START | backend=torch_pixel path=%s", manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("REPORT_UPDATE_END | backend=torch_pixel path=%s", manifest_path)

        return {
            "backend": "torch_pixel",
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "device": device,
            "classes": checkpoint["class_values"],
            "train_samples": int(x_np.shape[0]),
            "training_runtime_seconds": manifest["training_runtime_seconds"],
            "mean_train_epoch_seconds": manifest["mean_train_epoch_seconds"],
            "mean_validation_epoch_seconds": manifest["mean_validation_epoch_seconds"],
            "mean_epoch_runtime_seconds": manifest["mean_epoch_runtime_seconds"],
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
