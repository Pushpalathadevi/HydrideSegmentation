"""Run report loading, summarization, and comparison utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


@dataclass(frozen=True)
class RunReportSummary:
    """Normalized summary for a run report JSON file."""

    path: str
    report_kind: str
    schema_version: str
    backend: str
    status: str
    runtime_seconds: float | None
    runtime_human: str
    device: str
    config_sha256: str
    samples_evaluated: int | None
    tracked_samples: int
    html_report_path: str
    metrics: dict[str, float]
    payload: dict[str, Any]


def load_report_payload(path: str | Path) -> dict[str, Any]:
    """Load JSON payload from a report path."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"report not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {p}")
    return payload


def summarize_run_report(path: str | Path) -> RunReportSummary:
    """Summarize training/evaluation report payload into a common schema."""

    p = Path(path)
    payload = load_report_payload(p)
    schema = str(payload.get("schema_version", ""))
    backend = str(payload.get("backend", ""))
    status = str(payload.get("status", ""))
    runtime_seconds = _as_float(payload.get("runtime_seconds"))
    runtime_human = str(payload.get("runtime_human", ""))
    device = str(payload.get("runtime_device", payload.get("device", "")))
    config_sha256 = str(payload.get("config_sha256", ""))
    html_report_path = str(payload.get("html_report_path", ""))
    tracked_samples = len(payload.get("tracked_samples", payload.get("latest_tracked_samples", [])) or [])
    samples_evaluated = payload.get("samples_evaluated")
    samples_count = int(samples_evaluated) if isinstance(samples_evaluated, (int, float)) else None
    metrics: dict[str, float] = {}

    if schema.startswith("microseg.pixel_eval"):
        report_kind = "evaluation"
        raw_metrics = payload.get("metrics", {})
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                num = _as_float(value)
                if num is not None:
                    metrics[str(key)] = num
        mean_iou = _as_float(payload.get("metrics", {}).get("mean_iou") if isinstance(payload.get("metrics"), dict) else None)
        if mean_iou is not None:
            metrics.setdefault("mean_iou", mean_iou)
    elif schema.startswith("microseg.training_report"):
        report_kind = "training"
        progress = payload.get("progress", {})
        if isinstance(progress, dict):
            ep_completed = _as_float(progress.get("epochs_completed"))
            if ep_completed is not None:
                metrics["epochs_completed"] = ep_completed
        best_val_loss = _as_float(payload.get("best_val_loss"))
        if best_val_loss is not None:
            metrics["best_val_loss"] = best_val_loss
        history = payload.get("history", [])
        if isinstance(history, list) and history:
            latest = history[-1]
            if isinstance(latest, dict):
                for key in ["train_loss", "train_iou", "val_loss", "val_iou", "epoch_runtime_seconds"]:
                    num = _as_float(latest.get(key))
                    if num is not None:
                        metrics[key] = num
        if not status:
            status = "running"
    else:
        report_kind = "unknown"

    return RunReportSummary(
        path=str(p),
        report_kind=report_kind,
        schema_version=schema,
        backend=backend,
        status=status,
        runtime_seconds=runtime_seconds,
        runtime_human=runtime_human,
        device=device,
        config_sha256=config_sha256,
        samples_evaluated=samples_count,
        tracked_samples=tracked_samples,
        html_report_path=html_report_path,
        metrics=metrics,
        payload=payload,
    )


def compare_run_reports(
    baseline: RunReportSummary,
    candidate: RunReportSummary,
) -> dict[str, Any]:
    """Compare two run summaries and compute metric deltas."""

    all_keys = sorted(set(baseline.metrics.keys()) | set(candidate.metrics.keys()))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        base_val = baseline.metrics.get(key)
        cand_val = candidate.metrics.get(key)
        delta = None
        delta_pct = None
        if base_val is not None and cand_val is not None:
            delta = cand_val - base_val
            if abs(base_val) > 1e-12:
                delta_pct = (delta / base_val) * 100.0
        rows.append(
            {
                "metric": key,
                "baseline": base_val,
                "candidate": cand_val,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )

    return {
        "schema_version": "microseg.run_report_compare.v1",
        "baseline_path": baseline.path,
        "candidate_path": candidate.path,
        "baseline_kind": baseline.report_kind,
        "candidate_kind": candidate.report_kind,
        "baseline_schema": baseline.schema_version,
        "candidate_schema": candidate.schema_version,
        "same_kind": baseline.report_kind == candidate.report_kind,
        "same_schema": baseline.schema_version == candidate.schema_version,
        "same_backend": baseline.backend == candidate.backend,
        "same_config_sha256": bool(baseline.config_sha256) and baseline.config_sha256 == candidate.config_sha256,
        "rows": rows,
    }

