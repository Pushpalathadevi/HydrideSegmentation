"""Evaluation utilities for baseline pixel classifier models."""

from __future__ import annotations

import html
import hashlib
import json
import logging
import random
import time
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
from src.microseg.training.unet_binary import load_unet_binary_model, predict_unet_binary_mask


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_logger() -> logging.Logger:
    logger = logging.getLogger("microseg.evaluation.pixel")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _to_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _code_version() -> str:
    try:
        from hydride_segmentation.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _config_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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


def _binary_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_u8 = (gt.astype(np.uint8) * 255)
    pred_u8 = (pred.astype(np.uint8) * 255)
    diff_u8 = ((gt.astype(np.uint8) != pred.astype(np.uint8)).astype(np.uint8) * 255)
    gt_rgb = np.stack([gt_u8, gt_u8, gt_u8], axis=2)
    pred_rgb = np.stack([pred_u8, pred_u8, pred_u8], axis=2)
    diff_rgb = np.stack([diff_u8, diff_u8, diff_u8], axis=2)
    return np.concatenate([image, gt_rgb, pred_rgb, diff_rgb], axis=1).astype(np.uint8)


def _write_eval_html(payload: dict[str, Any], output_path: Path) -> None:
    metrics = payload.get("metrics", {})
    samples = payload.get("tracked_samples", [])
    rows = []
    for sample in samples:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(sample.get('sample_name', '')))}</td>"
            f"<td>{float(sample.get('pixel_accuracy', 0.0)):.4f}</td>"
            f"<td>{float(sample.get('macro_f1', 0.0)):.4f}</td>"
            f"<td>{float(sample.get('mean_iou', 0.0)):.4f}</td>"
            "</tr>"
        )
    gallery = []
    for sample in samples:
        panel = html.escape(str(sample.get("panel", "")))
        gallery.append(
            "<div style='margin:10px 0;padding:10px;border:1px solid #ddd;'>"
            f"<div><b>{html.escape(str(sample.get('sample_name', '')))}</b></div>"
            f"<img src='{panel}' style='max-width:100%;border:1px solid #333;'>"
            "</div>"
        )

    html_text = (
        "<html><head><meta charset='utf-8'><title>MicroSeg Evaluation Report</title></head><body>"
        "<h1>MicroSeg Evaluation Report</h1>"
        f"<p><b>Backend:</b> {html.escape(str(payload.get('backend', '')))}</p>"
        f"<p><b>Runtime Device:</b> {html.escape(str(payload.get('runtime_device', '')))}</p>"
        f"<p><b>Samples Evaluated:</b> {int(payload.get('samples_evaluated', 0))}</p>"
        f"<p><b>Runtime:</b> {html.escape(str(payload.get('runtime_human', '')))}</p>"
        "<h2>Metrics</h2>"
        "<ul>"
        f"<li>Pixel Accuracy: {float(metrics.get('pixel_accuracy', 0.0)):.6f}</li>"
        f"<li>Macro F1: {float(metrics.get('macro_f1', 0.0)):.6f}</li>"
        f"<li>Mean IoU: {float(metrics.get('mean_iou', 0.0)):.6f}</li>"
        "</ul>"
        "<h2>Tracked Sample Metrics</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Sample</th><th>Pixel Acc</th><th>Macro F1</th><th>Mean IoU</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h2>Tracked Samples (Input | GT | Pred | Diff)</h2>"
        + "".join(gallery)
        + "</body></html>"
    )
    output_path.write_text(html_text, encoding="utf-8")


@dataclass(frozen=True)
class PixelEvaluationConfig:
    """Configuration for evaluating baseline pixel classifier models."""

    dataset_dir: str
    model_path: str
    split: str = "val"
    output_path: str = "outputs/evaluation/pixel_eval_report.json"
    enable_gpu: bool = False
    device_policy: str = "cpu"
    write_html_report: bool = True
    tracking_samples: int = 8
    tracking_seed: int = 17


class PixelModelEvaluator:
    """Evaluate a trained baseline pixel classifier on a dataset split."""

    def evaluate(self, config: PixelEvaluationConfig) -> dict[str, Any]:
        logger = _ensure_logger()

        dataset_root = Path(config.dataset_dir)
        split_dir = dataset_root / config.split
        pairs = _collect_pairs(split_dir)
        out_path = Path(config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples_dir = out_path.parent / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(config.model_path)
        if model_path.suffix == ".pt":
            import torch

            ckpt = torch.load(model_path, map_location="cpu")
            schema = str(ckpt.get("schema_version", ""))

            if schema == "microseg.torch_unet_binary.v1":
                bundle = load_unet_binary_model(
                    model_path,
                    enable_gpu=bool(config.enable_gpu),
                    device_policy=str(config.device_policy),
                )
                predictor = lambda image: predict_unet_binary_mask(image, bundle)
                backend = "unet_binary"
                device = str(bundle["device"])
            else:
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

        started_utc = _utc_now()
        run_start = time.perf_counter()
        logger.info(
            "evaluation started | dataset=%s split=%s model=%s backend=%s device=%s samples=%d",
            dataset_root,
            config.split,
            model_path,
            backend,
            device,
            len(pairs),
        )

        y_true_blocks: list[np.ndarray] = []
        y_pred_blocks: list[np.ndarray] = []
        sample_metrics: list[dict[str, Any]] = []
        tracked_set: set[str] = set()
        if int(config.tracking_samples) > 0:
            rng = random.Random(int(config.tracking_seed))
            tracked = rng.sample(pairs, k=min(int(config.tracking_samples), len(pairs)))
            tracked_set = {p[0].name for p in tracked}

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):
            image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            gt = to_index_mask(np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8))
            pred = predictor(image)
            if gt.shape != pred.shape:
                raise ValueError(f"shape mismatch during evaluation: {img_path}")

            y_true_blocks.append(gt.reshape(-1))
            y_pred_blocks.append(pred.reshape(-1))

            labels_local = np.unique(np.concatenate([gt.reshape(-1), pred.reshape(-1)]))
            acc_local = float(np.mean(gt == pred))
            f1_local = float(
                f1_score(gt.reshape(-1), pred.reshape(-1), labels=labels_local, average="macro", zero_division=0)
            )
            iou_local, _ = _mean_iou(gt.reshape(-1), pred.reshape(-1), labels_local)

            sample_item: dict[str, Any] = {
                "sample_name": img_path.name,
                "pixel_accuracy": acc_local,
                "macro_f1": f1_local,
                "mean_iou": iou_local,
            }
            if img_path.name in tracked_set:
                panel = _binary_panel(image, (gt > 0).astype(np.uint8), (pred > 0).astype(np.uint8))
                panel_path = samples_dir / f"{img_path.stem}_panel.png"
                Image.fromarray(panel).save(panel_path)
                sample_item["panel"] = _to_rel(panel_path, out_path.parent)
            sample_metrics.append(sample_item)

            if idx == 1 or idx == len(pairs) or idx % max(1, len(pairs) // 10) == 0:
                elapsed = time.perf_counter() - run_start
                eta = (elapsed / idx) * (len(pairs) - idx)
                logger.info(
                    "evaluation progress %d/%d (%.1f%%) | elapsed=%s | eta=%s",
                    idx,
                    len(pairs),
                    (100.0 * idx) / len(pairs),
                    _format_seconds(elapsed),
                    _format_seconds(eta),
                )

        y_true = np.concatenate(y_true_blocks)
        y_pred = np.concatenate(y_pred_blocks)
        labels = np.unique(np.concatenate([y_true, y_pred]))

        pixel_acc = float(np.mean(y_true == y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        mean_iou, per_class_iou = _mean_iou(y_true, y_pred, labels)

        runtime_seconds = time.perf_counter() - run_start
        config_payload = asdict(config)
        payload = {
            "schema_version": "microseg.pixel_eval.v2",
            "created_utc": _utc_now(),
            "started_utc": started_utc,
            "config": config_payload,
            "config_sha256": _config_hash(config_payload),
            "backend": backend,
            "runtime_device": device,
            "runtime_seconds": runtime_seconds,
            "runtime_human": _format_seconds(runtime_seconds),
            "code_version": _code_version(),
            "samples_evaluated": len(pairs),
            "classes": [int(v) for v in labels.tolist()],
            "metrics": {
                "pixel_accuracy": pixel_acc,
                "macro_f1": macro_f1,
                "mean_iou": mean_iou,
            },
            "per_class_iou": per_class_iou,
            "tracked_samples": [s for s in sample_metrics if "panel" in s],
            "sample_metrics": sample_metrics,
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if bool(config.write_html_report):
            html_path = out_path.with_suffix(".html")
            _write_eval_html(payload, html_path)
            payload["html_report_path"] = str(html_path)

        logger.info(
            "evaluation complete | metrics: pixel_acc=%.4f macro_f1=%.4f mean_iou=%.4f | runtime=%s",
            pixel_acc,
            macro_f1,
            mean_iou,
            _format_seconds(runtime_seconds),
        )
        return payload
