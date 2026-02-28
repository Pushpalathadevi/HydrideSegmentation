"""YAML workflow profile persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SCHEMA_VERSION = "microseg.workflow_profile.v1"
SUPPORTED_SCOPES = {"dataset_prepare", "training", "evaluation", "hpc_ga", "preflight", "deployment"}


def _yaml_module():
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency import branch
        raise RuntimeError("PyYAML is required for workflow profile persistence") from exc
    return yaml


def write_workflow_profile(path: str | Path, *, scope: str, values: dict[str, Any]) -> Path:
    """Write a workflow profile YAML file."""

    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported workflow profile scope: {scope!r}")
    if not isinstance(values, dict):
        raise ValueError("workflow profile values must be a mapping")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "values": values,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml_module()
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def read_workflow_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a workflow profile YAML file."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"workflow profile not found: {p}")

    yaml = _yaml_module()
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("workflow profile payload must be a mapping")

    schema = str(payload.get("schema_version", ""))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported workflow profile schema: {schema!r}")

    scope = str(payload.get("scope", ""))
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported workflow profile scope: {scope!r}")

    values = payload.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("workflow profile values must be a mapping")

    return {
        "schema_version": schema,
        "scope": scope,
        "values": values,
    }
