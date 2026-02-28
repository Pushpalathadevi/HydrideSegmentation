"""Deployment runtime health checks and queue-safe batch execution."""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.quality import (
    DEPLOY_INFERENCE_FAILED,
    DEPLOY_MODEL_LOAD_FAILED,
    DEPLOY_MODEL_RESOLVE_FAILED,
    DEPLOY_OUTPUT_WRITE_FAILED,
    DEPLOY_PACKAGE_INVALID,
    DEPLOY_PREPROCESS_FAILED,
    INPUT_INVALID,
    INPUT_NOT_FOUND,
    classify_exception,
)

from .package_bundle import predict_from_artifact, resolve_model_artifact_from_package


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "sample") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


@dataclass
class HealthStep:
    step: str
    ok: bool
    error_code: str = ""
    message: str = ""
    duration_seconds: float = 0.0


@dataclass
class HealthItem:
    image_path: str
    ok: bool
    runtime_seconds: float
    steps: list[HealthStep] = field(default_factory=list)
    output_mask_path: str = ""
    output_overlay_path: str = ""


@dataclass(frozen=True)
class RuntimeHealthConfig:
    package_dir: str
    output_dir: str = "outputs/deployments/health"
    image_paths: tuple[str, ...] = ()
    image_dir: str = ""
    glob_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    max_workers: int = 1
    enable_gpu: bool = False
    device_policy: str = "cpu"


@dataclass
class RuntimeHealthReport:
    schema_version: str
    created_utc: str
    package_dir: str
    output_dir: str
    max_workers: int
    global_steps: list[HealthStep] = field(default_factory=list)
    items: list[HealthItem] = field(default_factory=list)
    ok: bool = False
    total_images: int = 0
    ok_images: int = 0
    failed_images: int = 0


@dataclass
class RuntimeHealthResult:
    schema_version: str
    created_utc: str
    report_path: str
    ok: bool
    total_images: int
    failed_images: int


def _collect_image_paths(cfg: RuntimeHealthConfig) -> list[Path]:
    out: list[Path] = []
    for raw in cfg.image_paths:
        p = Path(str(raw)).resolve()
        if p.exists() and p.is_file():
            out.append(p)
    if str(cfg.image_dir).strip():
        base = Path(cfg.image_dir).resolve()
        if base.exists() and base.is_dir():
            for pattern in cfg.glob_patterns:
                out.extend(sorted(base.glob(pattern)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _blend_overlay(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    fg = mask_u8 > 0
    overlay[fg, 0] = np.clip(0.65 * overlay[fg, 0] + 0.35 * 255.0, 0, 255).astype(np.uint8)
    overlay[fg, 1] = np.clip(0.65 * overlay[fg, 1], 0, 255).astype(np.uint8)
    overlay[fg, 2] = np.clip(0.65 * overlay[fg, 2], 0, 255).astype(np.uint8)
    return overlay


def _run_one(
    image_path: Path,
    *,
    model_artifact: Path,
    output_dir: Path,
    enable_gpu: bool,
    device_policy: str,
) -> HealthItem:
    steps: list[HealthStep] = []
    t0 = perf_counter()

    try:
        prep_started = perf_counter()
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        steps.append(
            HealthStep(
                step="preprocess",
                ok=True,
                duration_seconds=float(perf_counter() - prep_started),
            )
        )
    except Exception as exc:
        steps.append(
            HealthStep(
                step="preprocess",
                ok=False,
                error_code=DEPLOY_PREPROCESS_FAILED,
                message=str(exc),
                duration_seconds=float(perf_counter() - prep_started),
            )
        )
        return HealthItem(
            image_path=str(image_path),
            ok=False,
            runtime_seconds=float(perf_counter() - t0),
            steps=steps,
        )

    pred_started = perf_counter()
    try:
        pred = predict_from_artifact(
            image_rgb,
            model_artifact,
            enable_gpu=bool(enable_gpu),
            device_policy=str(device_policy),
        )
        pred_u8 = ((np.asarray(pred, dtype=np.uint8) > 0).astype(np.uint8) * 255)
        steps.append(
            HealthStep(
                step="inference",
                ok=True,
                duration_seconds=float(perf_counter() - pred_started),
            )
        )
    except Exception as exc:
        steps.append(
            HealthStep(
                step="inference",
                ok=False,
                error_code=DEPLOY_INFERENCE_FAILED,
                message=str(exc),
                duration_seconds=float(perf_counter() - pred_started),
            )
        )
        return HealthItem(
            image_path=str(image_path),
            ok=False,
            runtime_seconds=float(perf_counter() - t0),
            steps=steps,
        )

    write_started = perf_counter()
    try:
        stem = _safe_name(image_path.stem, fallback="sample")
        out_mask = output_dir / f"{stem}_health_mask.png"
        out_overlay = output_dir / f"{stem}_health_overlay.png"
        Image.fromarray(pred_u8).save(out_mask)
        Image.fromarray(_blend_overlay(image_rgb, pred_u8)).save(out_overlay)
        steps.append(
            HealthStep(
                step="output_write",
                ok=True,
                duration_seconds=float(perf_counter() - write_started),
            )
        )
    except Exception as exc:
        steps.append(
            HealthStep(
                step="output_write",
                ok=False,
                error_code=DEPLOY_OUTPUT_WRITE_FAILED,
                message=str(exc),
                duration_seconds=float(perf_counter() - write_started),
            )
        )
        return HealthItem(
            image_path=str(image_path),
            ok=False,
            runtime_seconds=float(perf_counter() - t0),
            steps=steps,
        )

    return HealthItem(
        image_path=str(image_path),
        ok=True,
        runtime_seconds=float(perf_counter() - t0),
        steps=steps,
        output_mask_path=str(out_mask),
        output_overlay_path=str(out_overlay),
    )


def run_runtime_health_checks(config: RuntimeHealthConfig) -> RuntimeHealthReport:
    """Run deployment runtime health checks for one or multiple images."""

    package_dir = Path(config.package_dir).resolve()
    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = RuntimeHealthReport(
        schema_version="microseg.runtime_health_report.v1",
        created_utc=_utc_now(),
        package_dir=str(package_dir),
        output_dir=str(out_dir),
        max_workers=max(1, int(config.max_workers)),
        ok=False,
    )

    if not str(config.package_dir).strip():
        report.global_steps.append(
            HealthStep(step="input_validation", ok=False, error_code=INPUT_INVALID, message="package_dir is required")
        )
        return report

    validate_started = perf_counter()
    try:
        _manifest, model_artifact = resolve_model_artifact_from_package(package_dir, verify_sha256=True)
        report.global_steps.append(
            HealthStep(
                step="package_validation",
                ok=True,
                duration_seconds=float(perf_counter() - validate_started),
            )
        )
    except Exception as exc:
        report.global_steps.append(
            HealthStep(
                step="package_validation",
                ok=False,
                error_code=DEPLOY_PACKAGE_INVALID,
                message=str(exc),
                duration_seconds=float(perf_counter() - validate_started),
            )
        )
        return report

    images = _collect_image_paths(config)
    if not images:
        report.global_steps.append(
            HealthStep(
                step="input_validation",
                ok=False,
                error_code=INPUT_NOT_FOUND,
                message="no input images found for runtime health check",
            )
        )
        return report

    load_started = perf_counter()
    try:
        # Warm load check: run one tiny dry infer call to ensure artifact is loadable.
        dummy = np.zeros((8, 8, 3), dtype=np.uint8)
        _ = predict_from_artifact(
            dummy,
            model_artifact,
            enable_gpu=bool(config.enable_gpu),
            device_policy=str(config.device_policy),
        )
        report.global_steps.append(
            HealthStep(
                step="model_load",
                ok=True,
                duration_seconds=float(perf_counter() - load_started),
            )
        )
    except Exception as exc:
        code = DEPLOY_MODEL_LOAD_FAILED
        if "artifact" in str(exc).lower() or "manifest" in str(exc).lower():
            code = DEPLOY_MODEL_RESOLVE_FAILED
        report.global_steps.append(
            HealthStep(
                step="model_load",
                ok=False,
                error_code=code,
                message=str(exc),
                duration_seconds=float(perf_counter() - load_started),
            )
        )
        return report

    workers = max(1, int(config.max_workers))
    if workers == 1:
        items = [
            _run_one(
                p,
                model_artifact=model_artifact,
                output_dir=out_dir,
                enable_gpu=bool(config.enable_gpu),
                device_policy=str(config.device_policy),
            )
            for p in images
        ]
    else:
        items = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="microseg_health") as ex:
            futures = {
                ex.submit(
                    _run_one,
                    p,
                    model_artifact=model_artifact,
                    output_dir=out_dir,
                    enable_gpu=bool(config.enable_gpu),
                    device_policy=str(config.device_policy),
                ): p
                for p in images
            }
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    items.append(fut.result())
                except Exception as exc:  # pragma: no cover - guarded fallback
                    items.append(
                        HealthItem(
                            image_path=str(p),
                            ok=False,
                            runtime_seconds=0.0,
                            steps=[
                                HealthStep(
                                    step="runtime",
                                    ok=False,
                                    error_code=classify_exception(exc, stage="deploy_inference"),
                                    message=f"{exc}\n{traceback.format_exc(limit=2)}",
                                )
                            ],
                        )
                    )

    items_sorted = sorted(items, key=lambda row: str(row.image_path))
    report.items = items_sorted
    report.total_images = len(items_sorted)
    report.ok_images = sum(1 for row in items_sorted if row.ok)
    report.failed_images = int(report.total_images - report.ok_images)
    report.ok = all(step.ok for step in report.global_steps) and report.failed_images == 0
    return report


def write_runtime_health_report(report: RuntimeHealthReport, output_path: str | Path) -> Path:
    """Write runtime health report JSON."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out


def run_runtime_health(config: RuntimeHealthConfig, *, report_path: str | Path = "") -> RuntimeHealthResult:
    """Run checks and persist report to default or user-provided output path."""

    report = run_runtime_health_checks(config)
    if str(report_path).strip():
        out = write_runtime_health_report(report, report_path)
    else:
        out = write_runtime_health_report(
            report,
            Path(config.output_dir).resolve() / "runtime_health_report.json",
        )
    return RuntimeHealthResult(
        schema_version="microseg.runtime_health_result.v1",
        created_utc=_utc_now(),
        report_path=str(out),
        ok=bool(report.ok),
        total_images=int(report.total_images),
        failed_images=int(report.failed_images),
    )
