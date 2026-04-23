"""Shared batch inference/export orchestration for desktop and CLI entry points."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.microseg.app.desktop_result_export import DesktopResultExportConfig, DesktopResultExporter
from src.microseg.app.desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from src.microseg.corrections.classes import SegmentationClassMap

ProgressCallback = Callable[["DesktopBatchProgress"], None]
RecordFinalizeCallback = Callable[[DesktopRunRecord], None]


@dataclass(frozen=True)
class DesktopBatchProgress:
    """Structured progress snapshot for long-running batch inference/export jobs."""

    stage: str
    message: str
    completed_steps: int
    total_steps: int
    completed_images: int
    total_images: int
    percent_complete: int
    elapsed_seconds: float
    eta_seconds: float | None = None
    current_image: str = ""
    batch_dir: str = ""


@dataclass(frozen=True)
class DesktopBatchJobResult:
    """Final outputs produced by a batch inference/export job."""

    records: list[DesktopRunRecord]
    batch_dir: Path
    summary_json_path: Path
    resolved_config_path: Path
    runs_dir: Path


def collect_inference_images(
    *,
    image: str | None,
    image_dir: str | None,
    glob_patterns: list[str],
    recursive: bool,
) -> list[Path]:
    """Resolve explicit image inputs and optional directory scans into file paths."""

    candidates: list[Path] = []
    if str(image or "").strip():
        candidates.append(Path(str(image)).expanduser().resolve())
    if str(image_dir or "").strip():
        root = Path(str(image_dir)).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"image directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"image directory is not a directory: {root}")
        patterns = [p.strip() for p in glob_patterns if p and p.strip()]
        if not patterns:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        scanner = root.rglob if recursive else root.glob
        for pattern in patterns:
            candidates.extend(scanner(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            unique.append(path)
    return unique


def _parse_utc_timestamp(text: str) -> datetime | None:
    """Parse a UTC ISO timestamp stored in run metadata."""

    cleaned = str(text).strip()
    if not cleaned:
        return None
    try:
        value = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def run_desktop_batch_job(
    *,
    workflow: DesktopWorkflowManager,
    result_exporter: DesktopResultExporter,
    image_paths: list[str | Path],
    model_name: str,
    output_dir: str | Path,
    params: dict[str, Any] | None = None,
    include_analysis: bool = False,
    annotator: str = "",
    notes: str = "",
    class_map: SegmentationClassMap | None = None,
    export_config: DesktopResultExportConfig | None = None,
    resolved_config: dict[str, Any] | None = None,
    finalize_record: RecordFinalizeCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    initial_per_image_seconds: float | None = None,
) -> DesktopBatchJobResult:
    """Run batch inference and export all per-run and aggregate artifacts in one pass."""

    resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
    if not resolved_paths:
        raise ValueError("image_paths is empty")

    started_at = time.monotonic()
    total_images = len(resolved_paths)
    total_steps = total_images + total_images + 1 + 1
    completed_steps = 0
    completed_images = 0
    records: list[DesktopRunRecord] = []
    params_template = dict(params or {})
    export_cfg = export_config or DesktopResultExportConfig(
        write_batch_summary=True,
        write_pdf_report=False,
    )

    def _eta_seconds() -> float | None:
        elapsed = max(0.0, time.monotonic() - started_at)
        remaining_steps = max(0, total_steps - completed_steps)
        if remaining_steps <= 0:
            return 0.0
        if completed_steps > 0:
            avg_step = elapsed / float(completed_steps)
            return max(0.0, avg_step * remaining_steps)
        if initial_per_image_seconds and initial_per_image_seconds > 0:
            estimate_total = (float(initial_per_image_seconds) * float(total_images)) + max(
                2.0,
                float(total_images) * 0.35,
            )
            return max(0.0, estimate_total - elapsed)
        return None

    def _emit(stage: str, message: str, *, current_image: str = "", batch_dir: Path | None = None) -> None:
        if progress_callback is None:
            return
        elapsed = max(0.0, time.monotonic() - started_at)
        total = max(1, total_steps)
        percent = int(round((float(completed_steps) / float(total)) * 100.0))
        progress_callback(
            DesktopBatchProgress(
                stage=str(stage),
                message=str(message),
                completed_steps=int(completed_steps),
                total_steps=int(total_steps),
                completed_images=int(completed_images),
                total_images=int(total_images),
                percent_complete=max(0, min(100, percent)),
                elapsed_seconds=float(elapsed),
                eta_seconds=_eta_seconds(),
                current_image=str(current_image),
                batch_dir="" if batch_dir is None else str(batch_dir),
            )
        )

    _emit(
        "infer",
        f"Starting batch inference for {total_images} image(s) with {model_name}.",
    )

    for index, image_path in enumerate(resolved_paths, start=1):
        image_name = image_path.name
        _emit(
            "infer",
            f"[{index}/{total_images}] Running inference for {image_name}.",
            current_image=image_name,
        )
        row_params = dict(params_template)
        row_params["image_path"] = str(image_path)
        record = workflow.run_single(
            str(image_path),
            model_name=model_name,
            params=row_params,
            include_analysis=include_analysis,
        )
        records.append(record)
        completed_images = index
        completed_steps += 1
        _emit(
            "infer",
            f"[{index}/{total_images}] Inference complete for {image_name}.",
            current_image=image_name,
        )

        _emit(
            "finalize",
            f"[{index}/{total_images}] Writing feedback and provenance for {image_name}.",
            current_image=image_name,
        )
        if finalize_record is not None:
            finalize_record(record)
        completed_steps += 1
        _emit(
            "finalize",
            f"[{index}/{total_images}] Finalized {image_name}.",
            current_image=image_name,
        )

    _emit("export", f"Exporting aggregate batch summary for {total_images} run(s).")
    batch_dir = result_exporter.export_batch(
        records,
        output_dir=output_dir,
        annotator=annotator,
        notes=notes,
        class_map=class_map,
        config=export_cfg,
    )
    completed_steps += 1
    _emit(
        "export",
        f"Batch summary export written to {batch_dir}.",
        batch_dir=batch_dir,
    )

    resolved_config_path = batch_dir / "resolved_config.json"
    _emit("export", "Writing resolved batch configuration manifest.", batch_dir=batch_dir)
    resolved_payload = resolved_config if isinstance(resolved_config, dict) else {}
    resolved_config_path.write_text(json.dumps(resolved_payload, indent=2), encoding="utf-8")
    completed_steps += 1
    job_finished_at = time.monotonic()
    job_elapsed_seconds = max(0.0, job_finished_at - started_at)
    run_durations: list[float] = []
    run_started_utc: list[datetime] = []
    run_finished_utc: list[datetime] = []
    for record in records:
        started = _parse_utc_timestamp(record.started_utc)
        finished = _parse_utc_timestamp(record.finished_utc)
        if started is not None:
            run_started_utc.append(started)
        if finished is not None:
            run_finished_utc.append(finished)
        if started is not None and finished is not None:
            run_durations.append(max(0.0, (finished - started).total_seconds()))

    telemetry: dict[str, Any] = {
        "job_elapsed_seconds": float(job_elapsed_seconds),
        "job_elapsed_human": f"{job_elapsed_seconds:.2f}s",
        "throughput_images_per_second": float(total_images / job_elapsed_seconds) if job_elapsed_seconds > 0 else None,
        "total_images": int(total_images),
        "completed_images": int(completed_images),
        "total_steps": int(total_steps),
        "completed_steps": int(completed_steps),
        "run_duration_seconds_total": float(sum(run_durations)) if run_durations else 0.0,
        "run_duration_seconds_mean": float(sum(run_durations) / len(run_durations)) if run_durations else 0.0,
        "run_duration_seconds_min": float(min(run_durations)) if run_durations else 0.0,
        "run_duration_seconds_max": float(max(run_durations)) if run_durations else 0.0,
        "earliest_run_started_utc": min(run_started_utc).isoformat() if run_started_utc else "",
        "latest_run_finished_utc": max(run_finished_utc).isoformat() if run_finished_utc else "",
        "batch_completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": str(model_name),
    }
    summary_json_path = batch_dir / "batch_results_summary.json"
    summary_payload: dict[str, Any] | None = None
    if summary_json_path.exists():
        summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
        if isinstance(summary_payload, dict):
            report_outputs = summary_payload.get("report_outputs", {})
            if not isinstance(report_outputs, dict):
                report_outputs = {}
            report_outputs["runs_dir"] = "runs"
            report_outputs["resolved_config"] = "resolved_config.json"
            summary_payload["report_outputs"] = report_outputs
            summary_payload["telemetry"] = telemetry
            summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            html_path = batch_dir / "batch_results_report.html"
            if html_path.exists():
                html_path.write_text(result_exporter._build_batch_html(summary_payload), encoding="utf-8")
            pdf_path = batch_dir / "batch_results_report.pdf"
            if pdf_path.exists():
                result_exporter._write_batch_pdf(pdf_path=pdf_path, payload=summary_payload)
    if bool(export_cfg.include_artifact_manifest):
        result_exporter.write_batch_artifact_manifest(batch_dir)
    _emit(
        "done",
        f"Batch inference and export complete for {total_images} image(s).",
        batch_dir=batch_dir,
    )
    return DesktopBatchJobResult(
        records=list(records),
        batch_dir=batch_dir,
        summary_json_path=summary_json_path,
        resolved_config_path=resolved_config_path,
        runs_dir=batch_dir / "runs",
    )
