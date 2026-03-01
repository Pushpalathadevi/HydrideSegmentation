"""Canary-shadow deployment comparison for candidate vs baseline packages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from src.microseg.quality import DEPLOY_INFERENCE_FAILED

from .package_bundle import build_predictor_from_artifact, resolve_model_artifact_from_package


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "sample") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _collect_image_paths(image_paths: tuple[str, ...], image_dir: str, glob_patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for raw in image_paths:
        p = Path(str(raw)).resolve()
        if p.exists() and p.is_file():
            out.append(p)
    if str(image_dir).strip():
        root = Path(image_dir).resolve()
        if root.exists() and root.is_dir():
            for pattern in glob_patterns:
                out.extend(sorted(root.glob(pattern)))
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _resolve_gt_mask(image_path: Path, mask_dir: str) -> Path | None:
    if not str(mask_dir).strip():
        return None
    root = Path(mask_dir).resolve()
    candidates = [
        root / image_path.name,
        root / f"{image_path.stem}.png",
        root / f"{image_path.stem}_mask.png",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))

    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    fpr_den = fp + tn
    fnr_den = fn + tp
    return {
        "iou": float(tp / iou_den) if iou_den > 0 else 1.0,
        "dice": float((2 * tp) / dice_den) if dice_den > 0 else 1.0,
        "false_positive_rate": float(fp / fpr_den) if fpr_den > 0 else 0.0,
        "false_negative_rate": float(fn / fnr_den) if fnr_den > 0 else 0.0,
    }


@dataclass(frozen=True)
class CanaryShadowConfig:
    baseline_package_dir: str
    candidate_package_dir: str
    output_dir: str = "outputs/deployments/canary_shadow"
    image_paths: tuple[str, ...] = ()
    image_dir: str = ""
    glob_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    mask_dir: str = ""
    enable_gpu: bool = False
    device_policy: str = "cpu"


@dataclass
class CanaryShadowItem:
    image_path: str
    ok: bool
    runtime_seconds: float
    error_code: str = ""
    message: str = ""
    disagreement_fraction: float = 0.0
    baseline_foreground_fraction: float = 0.0
    candidate_foreground_fraction: float = 0.0
    gt_mask_path: str = ""
    baseline_iou: float = 0.0
    candidate_iou: float = 0.0
    baseline_dice: float = 0.0
    candidate_dice: float = 0.0
    candidate_iou_gain: float = 0.0
    candidate_dice_gain: float = 0.0
    diff_mask_path: str = ""


@dataclass
class CanaryShadowReport:
    schema_version: str
    created_utc: str
    baseline_package_dir: str
    candidate_package_dir: str
    output_dir: str
    total_images: int
    ok_images: int
    failed_images: int
    mean_disagreement_fraction: float
    mean_candidate_iou_gain: float
    mean_candidate_dice_gain: float
    items: list[CanaryShadowItem] = field(default_factory=list)


@dataclass
class CanaryShadowResult:
    schema_version: str
    created_utc: str
    report_path: str
    ok: bool
    total_images: int
    failed_images: int


def run_canary_shadow_compare(config: CanaryShadowConfig, *, report_path: str | Path = "") -> CanaryShadowResult:
    """Compare baseline and candidate package predictions on same input set."""

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _base_manifest, baseline_artifact = resolve_model_artifact_from_package(
        config.baseline_package_dir,
        verify_sha256=True,
    )
    _cand_manifest, candidate_artifact = resolve_model_artifact_from_package(
        config.candidate_package_dir,
        verify_sha256=True,
    )
    baseline_predictor = build_predictor_from_artifact(
        baseline_artifact,
        enable_gpu=bool(config.enable_gpu),
        device_policy=str(config.device_policy),
    )
    candidate_predictor = build_predictor_from_artifact(
        candidate_artifact,
        enable_gpu=bool(config.enable_gpu),
        device_policy=str(config.device_policy),
    )

    image_list = _collect_image_paths(config.image_paths, config.image_dir, config.glob_patterns)
    items: list[CanaryShadowItem] = []

    for image_path in image_list:
        started = perf_counter()
        try:
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            baseline_pred = np.asarray(baseline_predictor(image_rgb), dtype=np.uint8) > 0
            candidate_pred = np.asarray(candidate_predictor(image_rgb), dtype=np.uint8) > 0
        except Exception as exc:
            items.append(
                CanaryShadowItem(
                    image_path=str(image_path),
                    ok=False,
                    runtime_seconds=float(perf_counter() - started),
                    error_code=DEPLOY_INFERENCE_FAILED,
                    message=str(exc),
                )
            )
            continue

        disagreement = np.mean(baseline_pred != candidate_pred)
        stem = _safe_name(image_path.stem, fallback="sample")
        diff_path = out_dir / f"{stem}_diff.png"
        Image.fromarray(((baseline_pred != candidate_pred).astype(np.uint8) * 255)).save(diff_path)

        item = CanaryShadowItem(
            image_path=str(image_path),
            ok=True,
            runtime_seconds=float(perf_counter() - started),
            disagreement_fraction=float(disagreement),
            baseline_foreground_fraction=float(np.mean(baseline_pred)),
            candidate_foreground_fraction=float(np.mean(candidate_pred)),
            diff_mask_path=str(diff_path),
        )

        gt_path = _resolve_gt_mask(image_path, config.mask_dir)
        if gt_path is not None:
            gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8) > 0
            base_m = _binary_metrics(baseline_pred, gt)
            cand_m = _binary_metrics(candidate_pred, gt)
            item.gt_mask_path = str(gt_path)
            item.baseline_iou = float(base_m["iou"])
            item.baseline_dice = float(base_m["dice"])
            item.candidate_iou = float(cand_m["iou"])
            item.candidate_dice = float(cand_m["dice"])
            item.candidate_iou_gain = float(cand_m["iou"] - base_m["iou"])
            item.candidate_dice_gain = float(cand_m["dice"] - base_m["dice"])

        items.append(item)

    ok_items = [row for row in items if row.ok]
    iou_gain_rows = [row.candidate_iou_gain for row in ok_items if row.gt_mask_path]
    dice_gain_rows = [row.candidate_dice_gain for row in ok_items if row.gt_mask_path]
    report = CanaryShadowReport(
        schema_version="microseg.canary_shadow_report.v1",
        created_utc=_utc_now(),
        baseline_package_dir=str(Path(config.baseline_package_dir).resolve()),
        candidate_package_dir=str(Path(config.candidate_package_dir).resolve()),
        output_dir=str(out_dir),
        total_images=len(items),
        ok_images=len(ok_items),
        failed_images=len(items) - len(ok_items),
        mean_disagreement_fraction=float(np.mean([row.disagreement_fraction for row in ok_items])) if ok_items else 0.0,
        mean_candidate_iou_gain=float(np.mean(iou_gain_rows)) if iou_gain_rows else 0.0,
        mean_candidate_dice_gain=float(np.mean(dice_gain_rows)) if dice_gain_rows else 0.0,
        items=items,
    )

    out = Path(report_path).resolve() if str(report_path).strip() else (out_dir / "canary_shadow_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    return CanaryShadowResult(
        schema_version="microseg.canary_shadow_result.v1",
        created_utc=_utc_now(),
        report_path=str(out),
        ok=report.failed_images == 0,
        total_images=report.total_images,
        failed_images=report.failed_images,
    )
