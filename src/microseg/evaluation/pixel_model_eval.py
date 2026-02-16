"""Evaluation utilities for baseline pixel classifier models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

from src.microseg.corrections.classes import to_index_mask
from src.microseg.training.pixel_classifier import load_pixel_classifier, predict_index_mask
from src.microseg.training.torch_pixel_classifier import (
    load_torch_pixel_classifier,
    predict_index_mask_torch,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"missing images/masks under split dir: {split_dir}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*")):
        msk = masks_dir / img.name
        if img.is_file() and msk.exists() and msk.is_file():
            pairs.append((img, msk))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found in {split_dir}")
    return pairs


def _mean_iou(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    vals: list[float] = []
    for cls in labels:
        t = y_true == cls
        p = y_pred == cls
        union = int(np.count_nonzero(t | p))
        if union == 0:
            continue
        inter = int(np.count_nonzero(t & p))
        iou = inter / union
        per_class[str(int(cls))] = float(iou)
        vals.append(float(iou))
    return (float(np.mean(vals)) if vals else 1.0, per_class)


@dataclass(frozen=True)
class PixelEvaluationConfig:
    """Configuration for evaluating baseline pixel classifier models."""

    dataset_dir: str
    model_path: str
    split: str = "val"
    output_path: str = "outputs/evaluation/pixel_eval_report.json"
    enable_gpu: bool = False
    device_policy: str = "cpu"


class PixelModelEvaluator:
    """Evaluate a trained baseline pixel classifier on a dataset split."""

    def evaluate(self, config: PixelEvaluationConfig) -> dict[str, Any]:
        dataset_root = Path(config.dataset_dir)
        split_dir = dataset_root / config.split
        pairs = _collect_pairs(split_dir)

        model_path = Path(config.model_path)
        if model_path.suffix == ".pt":
            bundle = load_torch_pixel_classifier(
                model_path,
                enable_gpu=bool(config.enable_gpu),
                device_policy=str(config.device_policy),
            )
            predictor = lambda image: predict_index_mask_torch(image, bundle)
            backend = "torch_pixel"
            device = str(bundle["device"])
        else:
            model = load_pixel_classifier(config.model_path)
            predictor = lambda image: predict_index_mask(image, model)
            backend = "sklearn_pixel"
            device = "cpu"

        y_true_blocks: list[np.ndarray] = []
        y_pred_blocks: list[np.ndarray] = []
        for img_path, mask_path in pairs:
            image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            gt = to_index_mask(np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8))
            pred = predictor(image)
            if gt.shape != pred.shape:
                raise ValueError(f"shape mismatch during evaluation: {img_path}")
            y_true_blocks.append(gt.reshape(-1))
            y_pred_blocks.append(pred.reshape(-1))

        y_true = np.concatenate(y_true_blocks)
        y_pred = np.concatenate(y_pred_blocks)
        labels = np.unique(np.concatenate([y_true, y_pred]))

        pixel_acc = float(np.mean(y_true == y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        mean_iou, per_class_iou = _mean_iou(y_true, y_pred, labels)

        payload = {
            "schema_version": "microseg.pixel_eval.v1",
            "created_utc": _utc_now(),
            "config": asdict(config),
            "backend": backend,
            "runtime_device": device,
            "samples_evaluated": len(pairs),
            "classes": [int(v) for v in labels.tolist()],
            "metrics": {
                "pixel_accuracy": pixel_acc,
                "macro_f1": macro_f1,
                "mean_iou": mean_iou,
            },
            "per_class_iou": per_class_iou,
        }

        out_path = Path(config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
