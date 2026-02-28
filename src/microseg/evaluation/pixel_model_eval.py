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
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from src.microseg.corrections.classes import binary_remapped_foreground_values, normalize_binary_index_mask
from src.microseg.training.pixel_classifier import load_pixel_classifier, predict_index_mask
from src.microseg.training.torch_pixel_classifier import (
    load_torch_pixel_classifier,
    predict_index_mask_torch,
)
from src.microseg.training.unet_binary import load_unet_binary_model, predict_unet_binary_mask
from src.microseg.evaluation.hydride_metrics import scientific_distance_metrics


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
        from src.microseg.version import __version__

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
            "binary_mask_normalization=%s remapped non-zero mask values %s to foreground class 1 during evaluation.",
            mode,
            sorted(remapped),
        )


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


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _advanced_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    per_class_iou: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, int], dict[str, Any]]:
    label_list = [int(v) for v in labels.tolist()]
    precision_vals = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_vals = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_vals = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_precision = {str(int(lbl)): float(v) for lbl, v in zip(label_list, precision_vals.tolist())}
    per_class_recall = {str(int(lbl)): float(v) for lbl, v in zip(label_list, recall_vals.tolist())}
    per_class_f1 = {str(int(lbl)): float(v) for lbl, v in zip(label_list, f1_vals.tolist())}
    class_support = {str(int(lbl)): int(np.count_nonzero(y_true == lbl)) for lbl in label_list}

    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    balanced_accuracy = macro_recall
    cohen_kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))

    total = float(y_true.size)
    freq_weighted_iou = 0.0
    for lbl in label_list:
        support = float(class_support.get(str(lbl), 0))
        iou = float(per_class_iou.get(str(lbl), 0.0))
        freq_weighted_iou += _safe_div(support, total) * iou

    out: dict[str, float] = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "cohen_kappa": cohen_kappa,
        "frequency_weighted_iou": float(freq_weighted_iou),
    }

    cm_payload: dict[str, Any] = {"labels": label_list, "counts": [], "row_normalized": [], "column_normalized": []}
    if label_list:
        counts = np.zeros((len(label_list), len(label_list)), dtype=np.int64)
        for i, gt_lbl in enumerate(label_list):
            gt_mask = y_true == gt_lbl
            for j, pred_lbl in enumerate(label_list):
                counts[i, j] = int(np.count_nonzero(gt_mask & (y_pred == pred_lbl)))
        cm_payload["counts"] = counts.tolist()
        row_denom = counts.sum(axis=1, keepdims=True).astype(np.float64)
        col_denom = counts.sum(axis=0, keepdims=True).astype(np.float64)
        row_norm = np.divide(counts, row_denom, out=np.zeros_like(counts, dtype=np.float64), where=row_denom != 0)
        col_norm = np.divide(counts, col_denom, out=np.zeros_like(counts, dtype=np.float64), where=col_denom != 0)
        cm_payload["row_normalized"] = row_norm.tolist()
        cm_payload["column_normalized"] = col_norm.tolist()

    # Binary-only foreground metrics when labels are exactly {0, 1}.
    if set(label_list) == {0, 1}:
        true_fg = y_true == 1
        pred_fg = y_pred == 1
        tp = float(np.count_nonzero(true_fg & pred_fg))
        tn = float(np.count_nonzero((~true_fg) & (~pred_fg)))
        fp = float(np.count_nonzero((~true_fg) & pred_fg))
        fn = float(np.count_nonzero(true_fg & (~pred_fg)))

        fg_precision = _safe_div(tp, tp + fp)
        fg_recall = _safe_div(tp, tp + fn)
        fg_specificity = _safe_div(tn, tn + fp)
        fg_iou = _safe_div(tp, tp + fp + fn)
        fg_dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
        fpr = _safe_div(fp, fp + tn)
        fnr = _safe_div(fn, fn + tp)

        mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = 0.0 if mcc_denom <= 0 else float((tp * tn - fp * fn) / np.sqrt(mcc_denom))

        out.update(
            {
                "foreground_precision": fg_precision,
                "foreground_recall": fg_recall,
                "foreground_specificity": fg_specificity,
                "foreground_iou": fg_iou,
                "foreground_dice": fg_dice,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "matthews_corrcoef": mcc,
                "gt_foreground_fraction": float(np.mean(true_fg)),
                "pred_foreground_fraction": float(np.mean(pred_fg)),
            }
        )

    return out, per_class_precision, per_class_recall, class_support, {"per_class_f1": per_class_f1, **cm_payload}


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
    scientific = payload.get("scientific_metrics", {})
    samples = payload.get("tracked_samples", [])
    per_class_iou = payload.get("per_class_iou", {})
    per_class_precision = payload.get("per_class_precision", {})
    per_class_recall = payload.get("per_class_recall", {})
    per_class_f1 = payload.get("per_class_f1", {})
    class_support = payload.get("class_support", {})
    confusion = payload.get("confusion_matrix", {})

    sample_metric_order = [
        "pixel_accuracy",
        "macro_f1",
        "mean_iou",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
        "cohen_kappa",
        "frequency_weighted_iou",
        "foreground_precision",
        "foreground_recall",
        "foreground_specificity",
        "foreground_iou",
        "foreground_dice",
        "false_positive_rate",
        "false_negative_rate",
        "matthews_corrcoef",
        "mask_area_fraction_abs_error",
        "hydride_count_abs_error",
        "hydride_size_wasserstein",
        "hydride_orientation_wasserstein",
    ]

    sample_columns = [key for key in sample_metric_order if any(key in sample for sample in samples)]
    if not sample_columns:
        sample_columns = ["pixel_accuracy", "macro_f1", "mean_iou"]

    rows: list[str] = []
    for sample in samples:
        metric_cells: list[str] = []
        for key in sample_columns:
            value = sample.get(key)
            if value is None:
                metric_cells.append("<td>-</td>")
                continue
            try:
                metric_cells.append(f"<td>{float(value):.6f}</td>")
            except Exception:
                metric_cells.append("<td>-</td>")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(sample.get('sample_name', '')))}</td>"
            + "".join(metric_cells)
            + "</tr>"
        )

    def _sample_metrics_block(sample: dict[str, Any]) -> str:
        items: list[str] = []
        for key in sample_columns:
            if key not in sample:
                continue
            try:
                value = float(sample.get(key))
            except Exception:
                continue
            items.append(
                "<li><b>"
                + html.escape(key.replace("_", " "))
                + "</b>: "
                + f"{value:.6f}"
                + "</li>"
            )
        if not items:
            return ""
        return "<ul style='margin:8px 0 0 18px;'>" + "".join(items) + "</ul>"

    gallery: list[str] = []
    for sample in samples:
        panel = html.escape(str(sample.get("panel", "")))
        gallery.append(
            "<div style='margin:10px 0;padding:10px;border:1px solid #ddd;'>"
            f"<div><b>{html.escape(str(sample.get('sample_name', '')))}</b></div>"
            f"<img src='{panel}' style='max-width:100%;border:1px solid #333;'>"
            + _sample_metrics_block(sample)
            + "</div>"
        )

    class_rows: list[str] = []
    class_keys = sorted(
        {str(k) for k in per_class_iou.keys()} | {str(k) for k in per_class_precision.keys()} | {str(k) for k in per_class_recall.keys()}
    )
    for class_key in class_keys:
        class_rows.append(
            "<tr>"
            f"<td>{html.escape(class_key)}</td>"
            f"<td>{int(class_support.get(class_key, 0))}</td>"
            f"<td>{float(per_class_iou.get(class_key, 0.0)):.6f}</td>"
            f"<td>{float(per_class_precision.get(class_key, 0.0)):.6f}</td>"
            f"<td>{float(per_class_recall.get(class_key, 0.0)):.6f}</td>"
            f"<td>{float(per_class_f1.get(class_key, 0.0)):.6f}</td>"
            "</tr>"
        )

    binary_rows: list[str] = []
    for key in [
        "foreground_precision",
        "foreground_recall",
        "foreground_specificity",
        "foreground_iou",
        "foreground_dice",
        "false_positive_rate",
        "false_negative_rate",
        "matthews_corrcoef",
        "cohen_kappa",
        "gt_foreground_fraction",
        "pred_foreground_fraction",
    ]:
        if key in metrics:
            binary_rows.append(
                "<tr>"
                f"<td>{html.escape(key)}</td>"
                f"<td>{float(metrics.get(key, 0.0)):.6f}</td>"
                "</tr>"
            )

    confusion_labels = confusion.get("labels", [])
    confusion_counts = confusion.get("counts", [])
    confusion_rows: list[str] = []
    if isinstance(confusion_labels, list) and isinstance(confusion_counts, list):
        header = "".join(f"<th>pred {html.escape(str(lbl))}</th>" for lbl in confusion_labels)
        confusion_rows.append(f"<tr><th>GT \\ Pred</th>{header}</tr>")
        for idx, lbl in enumerate(confusion_labels):
            row_vals = confusion_counts[idx] if idx < len(confusion_counts) and isinstance(confusion_counts[idx], list) else []
            row_html = "".join(f"<td>{int(v)}</td>" for v in row_vals)
            confusion_rows.append(f"<tr><th>{html.escape(str(lbl))}</th>{row_html}</tr>")

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
        f"<li>Macro Precision: {float(metrics.get('macro_precision', 0.0)):.6f}</li>"
        f"<li>Macro Recall: {float(metrics.get('macro_recall', 0.0)):.6f}</li>"
        f"<li>Weighted F1: {float(metrics.get('weighted_f1', 0.0)):.6f}</li>"
        f"<li>Balanced Accuracy: {float(metrics.get('balanced_accuracy', 0.0)):.6f}</li>"
        f"<li>Cohen Kappa: {float(metrics.get('cohen_kappa', 0.0)):.6f}</li>"
        f"<li>Frequency-Weighted IoU: {float(metrics.get('frequency_weighted_iou', 0.0)):.6f}</li>"
        "</ul>"
        "<h2>Per-Class Metrics</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Class</th><th>Support</th><th>IoU</th><th>Precision</th><th>Recall</th><th>F1</th></tr>"
        + "".join(class_rows)
        + "</table>"
        "<h2>Binary Foreground Diagnostics</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Metric</th><th>Value</th></tr>"
        + ("".join(binary_rows) if binary_rows else "<tr><td colspan='2'>Not applicable (non-binary labels).</td></tr>")
        + "</table>"
        "<h2>Confusion Matrix (Counts)</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        + ("".join(confusion_rows) if confusion_rows else "<tr><td>No confusion matrix available.</td></tr>")
        + "</table>"
        "<h2>Scientific Metrics (Mean across evaluated samples)</h2>"
        "<ul>"
        f"<li>Area Fraction Abs Error: {float(scientific.get('mask_area_fraction_abs_error', 0.0)):.6f}</li>"
        f"<li>Hydride Count Abs Error: {float(scientific.get('hydride_count_abs_error', 0.0)):.6f}</li>"
        f"<li>Size Wasserstein: {float(scientific.get('hydride_size_wasserstein', 0.0)):.6f}</li>"
        f"<li>Orientation Wasserstein: {float(scientific.get('hydride_orientation_wasserstein', 0.0)):.6f}</li>"
        "</ul>"
        "<h2>Tracked Sample Metrics</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Sample</th>"
        + "".join(f"<th>{html.escape(col)}</th>" for col in sample_columns)
        + "</tr>"
        + "".join(rows)
        + "</table>"
        "<h2>Tracked Samples (Input | GT | Pred | Diff)</h2>"
        "<p>Each sample panel includes per-image values for all available run metrics.</p>"
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
    binary_mask_normalization: str = "off"


class PixelModelEvaluator:
    """Evaluate a trained baseline pixel classifier on a dataset split."""

    def evaluate(self, config: PixelEvaluationConfig) -> dict[str, Any]:
        logger = _ensure_logger()

        dataset_root = Path(config.dataset_dir)
        split_dir = dataset_root / config.split
        pairs = _collect_pairs(split_dir)
        _warn_binary_mask_remap_values(
            pairs,
            binary_mask_normalization=str(config.binary_mask_normalization),
            logger=logger,
        )
        out_path = Path(config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples_dir = out_path.parent / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(config.model_path)
        model_initialization = "unknown"
        if model_path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            import torch

            ckpt = torch.load(model_path, map_location="cpu")
            schema = str(ckpt.get("schema_version", ""))

            if schema in {
                "microseg.torch_unet_binary.v1",
                "microseg.torch_segmentation_binary.v2",
                "microseg.hf_transformer_segmentation.v1",
            }:
                bundle = load_unet_binary_model(
                    model_path,
                    enable_gpu=bool(config.enable_gpu),
                    device_policy=str(config.device_policy),
                )
                predictor = lambda image: predict_unet_binary_mask(image, bundle)
                backend = str(bundle.get("backend", bundle.get("architecture", "unet_binary")))
                device = str(bundle["device"])
                model_initialization = str(bundle.get("model_initialization", "unknown"))
            else:
                bundle = load_torch_pixel_classifier(
                    model_path,
                    enable_gpu=bool(config.enable_gpu),
                    device_policy=str(config.device_policy),
                )
                predictor = lambda image: predict_index_mask_torch(image, bundle)
                backend = "torch_pixel"
                device = str(bundle["device"])
                model_initialization = str(bundle.get("init", "unknown"))
        else:
            model = load_pixel_classifier(config.model_path)
            predictor = lambda image: predict_index_mask(image, model)
            backend = "sklearn_pixel"
            device = "cpu"
            model_initialization = "native"

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
        scientific_rows: list[dict[str, float]] = []
        tracked_set: set[str] = set()
        if int(config.tracking_samples) > 0:
            rng = random.Random(int(config.tracking_seed))
            tracked = rng.sample(pairs, k=min(int(config.tracking_samples), len(pairs)))
            tracked_set = {p[0].name for p in tracked}

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):
            image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            gt = normalize_binary_index_mask(
                np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8),
                mode=str(config.binary_mask_normalization),
            )
            pred = predictor(image)
            if gt.shape != pred.shape:
                raise ValueError(f"shape mismatch during evaluation: {img_path}")

            y_true_blocks.append(gt.reshape(-1))
            y_pred_blocks.append(pred.reshape(-1))

            gt_flat = gt.reshape(-1)
            pred_flat = pred.reshape(-1)
            labels_local = np.unique(np.concatenate([gt_flat, pred_flat]))
            acc_local = float(np.mean(gt_flat == pred_flat))
            f1_local = float(
                f1_score(gt_flat, pred_flat, labels=labels_local, average="macro", zero_division=0)
            )
            iou_local, per_class_iou_local = _mean_iou(gt_flat, pred_flat, labels_local)
            advanced_local, _, _, _, _ = _advanced_metrics(
                gt_flat,
                pred_flat,
                labels_local,
                per_class_iou_local,
            )

            sample_item: dict[str, Any] = {
                "sample_name": img_path.name,
                "pixel_accuracy": acc_local,
                "macro_f1": f1_local,
                "mean_iou": iou_local,
                **advanced_local,
            }
            sci = scientific_distance_metrics((gt > 0).astype(np.uint8), (pred > 0).astype(np.uint8))
            sample_item.update(sci)
            scientific_rows.append(sci)
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
        advanced, per_class_precision, per_class_recall, class_support, extra = _advanced_metrics(
            y_true,
            y_pred,
            labels,
            per_class_iou,
        )
        per_class_f1 = extra.get("per_class_f1", {}) if isinstance(extra, dict) else {}
        confusion_matrix = {
            "labels": extra.get("labels", []),
            "counts": extra.get("counts", []),
            "row_normalized": extra.get("row_normalized", []),
            "column_normalized": extra.get("column_normalized", []),
        }

        runtime_seconds = time.perf_counter() - run_start
        scientific_metrics: dict[str, float] = {}
        if scientific_rows:
            sci_keys = scientific_rows[0].keys()
            scientific_metrics = {
                key: float(np.mean([float(r.get(key, 0.0)) for r in scientific_rows])) for key in sci_keys
            }

        config_payload = asdict(config)
        payload = {
            "schema_version": "microseg.pixel_eval.v4",
            "created_utc": _utc_now(),
            "started_utc": started_utc,
            "config": config_payload,
            "config_sha256": _config_hash(config_payload),
            "backend": backend,
            "model_initialization": model_initialization,
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
                **advanced,
            },
            "per_class_iou": per_class_iou,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "class_support": class_support,
            "confusion_matrix": confusion_matrix,
            "scientific_metrics": scientific_metrics,
            "tracked_samples": [s for s in sample_metrics if "panel" in s],
            "sample_metrics": sample_metrics,
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if bool(config.write_html_report):
            html_path = out_path.with_suffix(".html")
            _write_eval_html(payload, html_path)
            payload["html_report_path"] = str(html_path)

        logger.info(
            "evaluation complete | metrics: pixel_acc=%.4f macro_f1=%.4f mean_iou=%.4f weighted_f1=%.4f kappa=%.4f | runtime=%s",
            pixel_acc,
            macro_f1,
            mean_iou,
            float(advanced.get("weighted_f1", 0.0)),
            float(advanced.get("cohen_kappa", 0.0)),
            _format_seconds(runtime_seconds),
        )
        return payload
