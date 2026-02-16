"""CPU-first baseline pixel classifier training and inference utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier

from src.microseg.corrections.classes import to_index_mask


try:  # pragma: no cover - import guard
    import joblib
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("joblib is required for training utilities.") from exc


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

    # Hard cap if rounding above max_samples.
    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    return x, y


@dataclass(frozen=True)
class PixelTrainingConfig:
    """Configuration for baseline pixel classifier training."""

    dataset_dir: str
    output_dir: str
    train_split: str = "train"
    max_samples: int = 250_000
    max_iter: int = 500
    seed: int = 42


class PixelClassifierTrainer:
    """Train and persist baseline pixel classifier model artifacts."""

    def train(self, config: PixelTrainingConfig) -> dict[str, Any]:
        dataset_root = Path(config.dataset_dir)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        split_dir = dataset_root / config.train_split
        pairs = _collect_pairs(split_dir)
        x, y = _build_samples(pairs, max_samples=config.max_samples, seed=config.seed)

        if np.unique(y).size < 2:
            raise ValueError("training labels must contain at least 2 classes")

        clf = SGDClassifier(
            loss="log_loss",
            max_iter=int(config.max_iter),
            tol=1e-3,
            random_state=int(config.seed),
        )
        clf.fit(x, y)

        model_path = output_dir / "pixel_classifier.joblib"
        metadata_path = output_dir / "training_manifest.json"
        joblib.dump(clf, model_path)

        payload = {
            "schema_version": "microseg.pixel_classifier.v1",
            "created_utc": _utc_now(),
            "config": asdict(config),
            "train_pairs": len(pairs),
            "train_samples": int(x.shape[0]),
            "classes": [int(v) for v in np.unique(y).tolist()],
            "model_file": model_path.name,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "model_path": str(model_path),
            "manifest_path": str(metadata_path),
            "classes": payload["classes"],
            "train_samples": payload["train_samples"],
        }


def load_pixel_classifier(model_path: str | Path) -> Any:
    """Load persisted baseline pixel classifier model."""

    return joblib.load(Path(model_path))


def predict_index_mask(image: np.ndarray, model: Any) -> np.ndarray:
    """Predict an indexed segmentation mask from RGB image array."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB with shape (H, W, 3)")
    feat = image.reshape(-1, 3).astype(np.float32) / 255.0
    pred = model.predict(feat)
    out = np.asarray(pred, dtype=np.uint8).reshape(image.shape[:2])
    return out


def infer_image_with_pixel_classifier(
    image_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Run inference for one image and export indexed + color-free artifacts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    model = load_pixel_classifier(model_path)
    pred = predict_index_mask(image, model)

    stem = Path(image_path).stem
    out_img = output_root / f"{stem}_input.png"
    out_mask = output_root / f"{stem}_prediction_indexed.png"
    Image.fromarray(image).save(out_img)
    Image.fromarray(pred).save(out_mask)

    return {
        "input": str(out_img),
        "prediction_indexed": str(out_mask),
    }
