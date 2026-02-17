"""Validation utilities for frozen checkpoint registry metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .frozen_checkpoints import REGISTRY_SCHEMA, registry_path


@dataclass
class RegistryValidationReport:
    """Validation report for frozen model registry."""

    schema_version: str
    registry_path: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_count: int = 0


_REQUIRED_MODEL_FIELDS = (
    "model_id",
    "model_nickname",
    "model_type",
    "framework",
    "input_size",
    "input_dimensions",
    "checkpoint_path_hint",
    "application_remarks",
)
_ALLOWED_ARTIFACT_STAGES = {"", "smoke", "candidate", "promoted", "builtin", "deprecated"}


def _is_absolute_hint(value: str) -> bool:
    text = str(value).strip()
    return text.startswith("/") or text.startswith("\\") or (len(text) >= 2 and text[1] == ":")


def validate_frozen_registry(path: str | Path | None = None) -> RegistryValidationReport:
    """Validate frozen checkpoint registry payload and semantic constraints."""

    reg_path = Path(path) if path else registry_path()
    report = RegistryValidationReport(
        schema_version="microseg.registry_validation.v1",
        registry_path=str(reg_path),
        ok=False,
    )
    if not reg_path.exists():
        report.errors.append(f"registry file does not exist: {reg_path}")
        return report

    try:
        payload = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.errors.append(f"registry is not valid JSON: {exc}")
        return report

    if not isinstance(payload, dict):
        report.errors.append("registry root must be a JSON object")
        return report
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        report.errors.append(
            f"unsupported schema_version: {payload.get('schema_version')!r}; expected {REGISTRY_SCHEMA!r}"
        )

    models = payload.get("models")
    if not isinstance(models, list):
        report.errors.append("registry 'models' field must be a list")
        return report

    seen_ids: set[str] = set()
    seen_nicks: set[str] = set()
    report.model_count = len(models)

    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            report.errors.append(f"models[{idx}] must be an object")
            continue

        for field_name in _REQUIRED_MODEL_FIELDS:
            if not str(item.get(field_name, "")).strip():
                report.errors.append(f"models[{idx}] missing required field: {field_name}")

        model_id = str(item.get("model_id", "")).strip()
        nick = str(item.get("model_nickname", "")).strip()
        if model_id:
            if model_id in seen_ids:
                report.errors.append(f"duplicate model_id: {model_id}")
            seen_ids.add(model_id)
        if nick:
            if nick in seen_nicks:
                report.errors.append(f"duplicate model_nickname: {nick}")
            seen_nicks.add(nick)

        hint = str(item.get("checkpoint_path_hint", "")).strip()
        if hint and hint.lower() != "n/a" and _is_absolute_hint(hint):
            report.errors.append(f"model '{model_id}' uses absolute checkpoint_path_hint; use repo-relative path")

        artifact_stage = str(item.get("artifact_stage", "")).strip().lower()
        if artifact_stage not in _ALLOWED_ARTIFACT_STAGES:
            report.errors.append(
                f"model '{model_id}' has unsupported artifact_stage={artifact_stage!r}; "
                f"allowed={sorted(_ALLOWED_ARTIFACT_STAGES)}"
            )

        source_run_manifest = str(item.get("source_run_manifest", "")).strip()
        if source_run_manifest and _is_absolute_hint(source_run_manifest):
            report.errors.append(f"model '{model_id}' uses absolute source_run_manifest path; use repo-relative path")

        quality_report_path = str(item.get("quality_report_path", "")).strip()
        if quality_report_path and _is_absolute_hint(quality_report_path):
            report.errors.append(f"model '{model_id}' uses absolute quality_report_path; use repo-relative path")

        size_value = item.get("file_size_bytes")
        if size_value not in (None, ""):
            try:
                size_int = int(size_value)
            except Exception:
                report.errors.append(f"model '{model_id}' file_size_bytes must be an integer when provided")
            else:
                if size_int <= 0:
                    report.errors.append(f"model '{model_id}' file_size_bytes must be > 0 when provided")

        classes = item.get("classes", [])
        if not isinstance(classes, list):
            report.errors.append(f"model '{model_id}' classes must be a list")
        else:
            class_indices: set[int] = set()
            for c_idx, cls in enumerate(classes):
                if not isinstance(cls, dict):
                    report.errors.append(f"model '{model_id}' classes[{c_idx}] must be object")
                    continue
                try:
                    index = int(cls["index"])
                except Exception:
                    report.errors.append(f"model '{model_id}' classes[{c_idx}] missing valid integer 'index'")
                    continue
                if index < 0:
                    report.errors.append(f"model '{model_id}' classes[{c_idx}] has negative index")
                if index in class_indices:
                    report.errors.append(f"model '{model_id}' duplicate class index: {index}")
                class_indices.add(index)
                if not str(cls.get("name", "")).strip():
                    report.errors.append(f"model '{model_id}' classes[{c_idx}] missing class name")
            if classes and 0 not in class_indices:
                report.warnings.append(f"model '{model_id}' has no class index 0 (background is typically index 0)")

    report.ok = len(report.errors) == 0
    return report


def write_registry_validation_report(
    report: RegistryValidationReport,
    output_path: str | Path,
) -> Path:
    """Write validation report to JSON output path."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out
