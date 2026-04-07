"""Deployment performance harness (latency and throughput)."""

from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _collect_images(image_paths: tuple[str, ...], image_dir: str, glob_patterns: tuple[str, ...]) -> list[Path]:
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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass(frozen=True)
class DeploymentPerfConfig:
    package_dir: str
    output_dir: str = "outputs/deployments/perf"
    image_paths: tuple[str, ...] = ()
    image_dir: str = ""
    glob_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    warmup_runs: int = 1
    repeat: int = 1
    max_workers: int = 1
    enable_gpu: bool = False
    device_policy: str = "cpu"


@dataclass
class DeploymentPerfSample:
    image_path: str
    request_index: int
    latency_ms: float
    ok: bool
    error_code: str = ""
    message: str = ""


@dataclass
class DeploymentPerfReport:
    schema_version: str
    created_utc: str
    package_dir: str
    output_dir: str
    total_requests: int
    ok_requests: int
    failed_requests: int
    wall_time_seconds: float
    throughput_images_per_second: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    latency_ms_p99: float
    max_workers: int
    warmup_runs: int
    repeat: int
    samples: list[DeploymentPerfSample] = field(default_factory=list)


@dataclass
class DeploymentPerfResult:
    schema_version: str
    created_utc: str
    report_path: str
    csv_path: str
    ok: bool
    failed_requests: int


def run_deployment_perf(config: DeploymentPerfConfig, *, report_path: str | Path = "") -> DeploymentPerfResult:
    """Run deployment load/perf benchmark and write JSON/CSV artifacts."""

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _manifest, model_artifact = resolve_model_artifact_from_package(config.package_dir, verify_sha256=True)
    predictor = build_predictor_from_artifact(
        model_artifact,
        enable_gpu=bool(config.enable_gpu),
        device_policy=str(config.device_policy),
    )
    predict_lock = threading.Lock()

    images = _collect_images(config.image_paths, config.image_dir, config.glob_patterns)
    if not images:
        raise ValueError("no input images found for deploy-perf")

    for _ in range(max(0, int(config.warmup_runs))):
        for p in images:
            image_rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            with predict_lock:
                _ = predictor(image_rgb)

    requests: list[tuple[int, Path]] = []
    req_id = 0
    for _ in range(max(1, int(config.repeat))):
        for path in images:
            req_id += 1
            requests.append((req_id, path))

    def _run_one(idx: int, image_path: Path) -> DeploymentPerfSample:
        started = perf_counter()
        try:
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            with predict_lock:
                _ = predictor(image_rgb)
            latency_ms = float((perf_counter() - started) * 1000.0)
            return DeploymentPerfSample(
                image_path=str(image_path),
                request_index=int(idx),
                latency_ms=latency_ms,
                ok=True,
            )
        except Exception as exc:
            latency_ms = float((perf_counter() - started) * 1000.0)
            return DeploymentPerfSample(
                image_path=str(image_path),
                request_index=int(idx),
                latency_ms=latency_ms,
                ok=False,
                error_code=DEPLOY_INFERENCE_FAILED,
                message=str(exc),
            )

    workers = max(1, int(config.max_workers))
    samples: list[DeploymentPerfSample] = []
    wall_started = perf_counter()
    if workers == 1:
        for idx, path in requests:
            samples.append(_run_one(idx, path))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="microseg_perf") as ex:
            futures = {ex.submit(_run_one, idx, path): (idx, path) for idx, path in requests}
            for fut in as_completed(futures):
                samples.append(fut.result())
    wall_seconds = float(perf_counter() - wall_started)

    samples = sorted(samples, key=lambda row: int(row.request_index))
    lat_ok = [row.latency_ms for row in samples if row.ok]
    ok_requests = len(lat_ok)
    failed_requests = len(samples) - ok_requests

    report = DeploymentPerfReport(
        schema_version="microseg.deployment_perf_report.v1",
        created_utc=_utc_now(),
        package_dir=str(Path(config.package_dir).resolve()),
        output_dir=str(out_dir),
        total_requests=len(samples),
        ok_requests=ok_requests,
        failed_requests=failed_requests,
        wall_time_seconds=wall_seconds,
        throughput_images_per_second=float(ok_requests / wall_seconds) if wall_seconds > 0 else 0.0,
        latency_ms_mean=float(np.mean(lat_ok)) if lat_ok else 0.0,
        latency_ms_p50=_percentile(lat_ok, 50),
        latency_ms_p90=_percentile(lat_ok, 90),
        latency_ms_p95=_percentile(lat_ok, 95),
        latency_ms_p99=_percentile(lat_ok, 99),
        max_workers=workers,
        warmup_runs=max(0, int(config.warmup_runs)),
        repeat=max(1, int(config.repeat)),
        samples=samples,
    )

    json_path = Path(report_path).resolve() if str(report_path).strip() else (out_dir / "deployment_perf_report.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["request_index", "image_path", "latency_ms", "ok", "error_code", "message"],
        )
        writer.writeheader()
        for row in samples:
            writer.writerow(
                {
                    "request_index": int(row.request_index),
                    "image_path": str(row.image_path),
                    "latency_ms": float(row.latency_ms),
                    "ok": bool(row.ok),
                    "error_code": str(row.error_code),
                    "message": str(row.message),
                }
            )

    return DeploymentPerfResult(
        schema_version="microseg.deployment_perf_result.v1",
        created_utc=_utc_now(),
        report_path=str(json_path),
        csv_path=str(csv_path),
        ok=failed_requests == 0,
        failed_requests=failed_requests,
    )
