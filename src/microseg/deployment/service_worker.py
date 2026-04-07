"""Queue-safe deployment service worker for batch/API integration."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.microseg.feedback import FeedbackArtifactWriter, FeedbackCaptureConfig
from src.microseg.quality import (
    DEPLOY_INFERENCE_FAILED,
    DEPLOY_MODEL_LOAD_FAILED,
    DEPLOY_OUTPUT_WRITE_FAILED,
    DEPLOY_PREPROCESS_FAILED,
    INPUT_NOT_FOUND,
    SERVICE_JOB_NOT_FOUND,
    SERVICE_JOB_TIMEOUT,
    SERVICE_QUEUE_FULL,
    classify_exception,
)

from .package_bundle import build_predictor_from_artifact, resolve_model_artifact_from_package


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "sample") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _overlay(image_rgb: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    fg = pred_u8 > 0
    out[fg, 0] = np.clip(0.65 * out[fg, 0] + 0.35 * 255.0, 0, 255).astype(np.uint8)
    out[fg, 1] = np.clip(0.65 * out[fg, 1], 0, 255).astype(np.uint8)
    out[fg, 2] = np.clip(0.65 * out[fg, 2], 0, 255).astype(np.uint8)
    return out


@dataclass
class ServiceJobResult:
    schema_version: str
    job_id: str
    image_path: str
    status: Literal["queued", "running", "completed", "failed", "rejected", "not_found", "timeout"]
    submitted_utc: str
    started_utc: str = ""
    completed_utc: str = ""
    runtime_seconds: float = 0.0
    error_code: str = ""
    message: str = ""
    output_mask_path: str = ""
    output_overlay_path: str = ""
    feedback_record_dir: str = ""
    feedback_record_id: str = ""


@dataclass(frozen=True)
class ServiceWorkerConfig:
    package_dir: str
    output_dir: str = "outputs/deployments/service"
    max_workers: int = 2
    max_queue_size: int = 32
    enable_gpu: bool = False
    device_policy: str = "cpu"
    capture_feedback: bool = True
    feedback_root: str = "outputs/feedback_records"
    deployment_id: str = "deployment_service"
    operator_id: str = "unknown_operator"


@dataclass
class ServiceBatchResult:
    schema_version: str
    created_utc: str
    output_dir: str
    report_path: str
    total_submitted: int
    accepted: int
    rejected: int
    completed: int
    failed: int
    jobs: list[ServiceJobResult] = field(default_factory=list)


class DeploymentServiceWorker:
    """Thread-safe bounded-queue deployment service worker."""

    def __init__(self, config: ServiceWorkerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(config.max_workers)), thread_name_prefix="microseg_svc")
        self._capacity = threading.Semaphore(max(1, int(config.max_queue_size)))
        self._predict_lock = threading.Lock()
        self._lock = threading.Lock()
        self._jobs: dict[str, ServiceJobResult] = {}
        self._futures: dict[str, Future[ServiceJobResult]] = {}

        self._feedback_writer: FeedbackArtifactWriter | None = None
        if bool(config.capture_feedback):
            self._feedback_writer = FeedbackArtifactWriter(
                FeedbackCaptureConfig(
                    feedback_root=str(config.feedback_root),
                    deployment_id=str(config.deployment_id),
                    operator_id=str(config.operator_id),
                    source="service_worker",
                )
            )

        try:
            manifest, model_artifact = resolve_model_artifact_from_package(
                config.package_dir,
                verify_sha256=True,
            )
            self._deployment_manifest = dict(manifest)
            self._predictor = build_predictor_from_artifact(
                model_artifact,
                enable_gpu=bool(config.enable_gpu),
                device_policy=str(config.device_policy),
            )
            self._model_ready_error: tuple[str, str] | None = None
        except Exception as exc:
            self._deployment_manifest = {}
            self._predictor = None
            self._model_ready_error = (DEPLOY_MODEL_LOAD_FAILED, str(exc))

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def submit(self, image_path: str | Path) -> ServiceJobResult:
        image = Path(image_path).resolve()
        now = _utc_now()
        job_id = uuid.uuid4().hex
        if self._model_ready_error is not None:
            code, message = self._model_ready_error
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image),
                status="rejected",
                submitted_utc=now,
                completed_utc=now,
                error_code=code,
                message=message,
            )

        if not image.exists() or not image.is_file():
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image),
                status="rejected",
                submitted_utc=now,
                completed_utc=now,
                error_code=INPUT_NOT_FOUND,
                message=f"image path not found: {image}",
            )

        if not self._capacity.acquire(blocking=False):
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image),
                status="rejected",
                submitted_utc=now,
                completed_utc=now,
                error_code=SERVICE_QUEUE_FULL,
                message=(
                    f"worker queue full (max_queue_size={int(self.config.max_queue_size)}); "
                    "retry later"
                ),
            )

        queued = ServiceJobResult(
            schema_version="microseg.service_job_result.v1",
            job_id=job_id,
            image_path=str(image),
            status="queued",
            submitted_utc=now,
        )
        with self._lock:
            self._jobs[job_id] = queued

        future = self._executor.submit(self._execute_job, queued)
        with self._lock:
            self._futures[job_id] = future

        def _done(fut: Future[ServiceJobResult]) -> None:
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - defensive branch
                result = ServiceJobResult(
                    schema_version="microseg.service_job_result.v1",
                    job_id=job_id,
                    image_path=str(image),
                    status="failed",
                    submitted_utc=now,
                    completed_utc=_utc_now(),
                    error_code=classify_exception(exc, stage="deploy_inference"),
                    message=str(exc),
                )
            finally:
                self._capacity.release()
            with self._lock:
                self._jobs[job_id] = result

        future.add_done_callback(_done)
        return queued

    def get(self, job_id: str) -> ServiceJobResult:
        with self._lock:
            existing = self._jobs.get(str(job_id).strip())
            if existing is None:
                return ServiceJobResult(
                    schema_version="microseg.service_job_result.v1",
                    job_id=str(job_id),
                    image_path="",
                    status="not_found",
                    submitted_utc="",
                    completed_utc=_utc_now(),
                    error_code=SERVICE_JOB_NOT_FOUND,
                    message=f"job_id not found: {job_id}",
                )
            return ServiceJobResult(**asdict(existing))

    def wait(self, job_id: str, *, timeout_seconds: float | None = None) -> ServiceJobResult:
        with self._lock:
            fut = self._futures.get(str(job_id).strip())
            current = self._jobs.get(str(job_id).strip())
        if fut is None:
            return self.get(job_id)
        try:
            return fut.result(timeout=timeout_seconds)
        except TimeoutError:
            image_path = str(current.image_path) if current is not None else ""
            submitted_utc = str(current.submitted_utc) if current is not None else ""
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=str(job_id),
                image_path=image_path,
                status="timeout",
                submitted_utc=submitted_utc,
                completed_utc=_utc_now(),
                error_code=SERVICE_JOB_TIMEOUT,
                message=f"wait timeout expired ({timeout_seconds}s)",
            )

    def run_batch(
        self,
        image_paths: list[str | Path],
        *,
        await_completion: bool = True,
        timeout_seconds: float | None = None,
    ) -> list[ServiceJobResult]:
        queued = [self.submit(path) for path in image_paths]
        if not await_completion:
            return queued
        out: list[ServiceJobResult] = []
        for item in queued:
            if item.status in {"rejected", "not_found"}:
                out.append(item)
                continue
            out.append(self.wait(item.job_id, timeout_seconds=timeout_seconds))
        return out

    def _execute_job(self, queued: ServiceJobResult) -> ServiceJobResult:
        started_utc = _utc_now()
        start = perf_counter()
        job_id = str(queued.job_id)
        image_path = Path(queued.image_path)

        with self._lock:
            self._jobs[job_id] = ServiceJobResult(
                **{
                    **asdict(queued),
                    "status": "running",
                    "started_utc": started_utc,
                }
            )

        try:
            prep_started = perf_counter()
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            _ = prep_started  # reserved for parity with step timings
        except Exception as exc:
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image_path),
                status="failed",
                submitted_utc=queued.submitted_utc,
                started_utc=started_utc,
                completed_utc=_utc_now(),
                runtime_seconds=float(perf_counter() - start),
                error_code=DEPLOY_PREPROCESS_FAILED,
                message=str(exc),
            )

        try:
            with self._predict_lock:
                pred = np.asarray(self._predictor(image_rgb), dtype=np.uint8)
            pred_u8 = ((pred > 0).astype(np.uint8) * 255)
        except Exception as exc:
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image_path),
                status="failed",
                submitted_utc=queued.submitted_utc,
                started_utc=started_utc,
                completed_utc=_utc_now(),
                runtime_seconds=float(perf_counter() - start),
                error_code=DEPLOY_INFERENCE_FAILED,
                message=str(exc),
            )

        stem = _safe_name(Path(image_path).stem, fallback=job_id)
        out_mask = self.output_dir / f"{stem}_{job_id[:8]}_mask.png"
        out_overlay = self.output_dir / f"{stem}_{job_id[:8]}_overlay.png"
        overlay_rgb = _overlay(image_rgb, pred_u8)
        try:
            Image.fromarray(pred_u8).save(out_mask)
            Image.fromarray(overlay_rgb).save(out_overlay)
        except Exception as exc:
            return ServiceJobResult(
                schema_version="microseg.service_job_result.v1",
                job_id=job_id,
                image_path=str(image_path),
                status="failed",
                submitted_utc=queued.submitted_utc,
                started_utc=started_utc,
                completed_utc=_utc_now(),
                runtime_seconds=float(perf_counter() - start),
                error_code=DEPLOY_OUTPUT_WRITE_FAILED,
                message=str(exc),
            )

        feedback_record_dir = ""
        feedback_record_id = ""
        if self._feedback_writer is not None:
            try:
                model_info = self._deployment_manifest.get("model", {})
                runtime_hints = self._deployment_manifest.get("runtime_hints", {})
                capture = self._feedback_writer.create_from_inference_arrays(
                    run_id=job_id,
                    image_path=str(image_path),
                    input_image_rgb=image_rgb,
                    predicted_mask_indexed=pred_u8,
                    predicted_overlay_rgb=overlay_rgb,
                    model_id=str(model_info.get("registry_model_id", model_info.get("artifact_rel_path", "deployment_model"))),
                    model_name=str(model_info.get("display_name", model_info.get("artifact_rel_path", "deployment_model"))),
                    model_artifact_hint=str(model_info.get("artifact_rel_path", "")),
                    started_utc=started_utc,
                    finished_utc=_utc_now(),
                    inference_manifest={
                        "pipeline": "microseg.deployment_service_worker",
                        "version": "v1",
                        "deployment_manifest_schema": str(self._deployment_manifest.get("schema_version", "")),
                        "runtime_hints": runtime_hints,
                    },
                    resolved_config={
                        "package_dir": str(self.config.package_dir),
                        "output_dir": str(self.config.output_dir),
                        "max_workers": int(self.config.max_workers),
                        "max_queue_size": int(self.config.max_queue_size),
                        "enable_gpu": bool(self.config.enable_gpu),
                        "device_policy": str(self.config.device_policy),
                        "runtime_hints": runtime_hints,
                        "preprocess_contract": self._deployment_manifest.get("preprocess_contract", {}),
                        "postprocess_contract": self._deployment_manifest.get("postprocess_contract", {}),
                    },
                    params={},
                    runtime={
                        "mode": "service_worker",
                        "enable_gpu": bool(self.config.enable_gpu),
                        "device_policy": str(self.config.device_policy),
                        "max_workers": int(self.config.max_workers),
                        "max_queue_size": int(self.config.max_queue_size),
                    },
                    source="service_worker",
                    operator_id=str(self.config.operator_id),
                )
                feedback_record_dir = str(capture.record_dir)
                feedback_record_id = str(capture.record_id)
            except Exception:
                # Feedback capture must not fail deployment inference path.
                feedback_record_dir = ""
                feedback_record_id = ""

        return ServiceJobResult(
            schema_version="microseg.service_job_result.v1",
            job_id=job_id,
            image_path=str(image_path),
            status="completed",
            submitted_utc=queued.submitted_utc,
            started_utc=started_utc,
            completed_utc=_utc_now(),
            runtime_seconds=float(perf_counter() - start),
            output_mask_path=str(out_mask),
            output_overlay_path=str(out_overlay),
            feedback_record_dir=feedback_record_dir,
            feedback_record_id=feedback_record_id,
        )


def _collect_images(paths: tuple[str, ...], image_dir: str, glob_patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
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


def run_service_worker_batch(
    config: ServiceWorkerConfig,
    *,
    image_paths: tuple[str, ...] = (),
    image_dir: str = "",
    glob_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"),
    await_completion: bool = True,
    timeout_seconds: float | None = None,
    report_path: str | Path = "",
) -> ServiceBatchResult:
    """Run worker in batch mode for CLI orchestration and smoke validation."""

    worker = DeploymentServiceWorker(config)
    try:
        images = _collect_images(image_paths, image_dir, glob_patterns)
        jobs = worker.run_batch(
            [str(p) for p in images],
            await_completion=bool(await_completion),
            timeout_seconds=timeout_seconds,
        )
    finally:
        worker.shutdown(wait=True)

    accepted = sum(1 for row in jobs if row.status not in {"rejected", "not_found"})
    rejected = sum(1 for row in jobs if row.status == "rejected")
    completed = sum(1 for row in jobs if row.status == "completed")
    failed = sum(1 for row in jobs if row.status in {"failed", "timeout"})

    result = ServiceBatchResult(
        schema_version="microseg.service_batch_result.v1",
        created_utc=_utc_now(),
        output_dir=str(Path(config.output_dir).resolve()),
        report_path="",
        total_submitted=len(jobs),
        accepted=accepted,
        rejected=rejected,
        completed=completed,
        failed=failed,
        jobs=jobs,
    )

    if str(report_path).strip():
        out = Path(report_path).resolve()
    else:
        out = Path(config.output_dir).resolve() / "service_batch_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    result.report_path = str(out)
    return result
