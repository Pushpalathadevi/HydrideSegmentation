"""Frozen checkpoint registry helpers for model metadata and user guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA = "microseg.frozen_checkpoint_registry.v1"


@dataclass(frozen=True)
class FrozenCheckpointRecord:
    """Metadata record for one frozen model checkpoint entry."""

    model_id: str
    model_nickname: str
    model_type: str
    framework: str
    input_size: str
    input_dimensions: str
    checkpoint_path_hint: str
    application_remarks: str
    short_description: str = ""
    detailed_description: str = ""
    classes: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenCheckpointRecord:
        """Create and validate record from untrusted dictionary payload."""

        required = [
            "model_id",
            "model_nickname",
            "model_type",
            "framework",
            "input_size",
            "input_dimensions",
            "checkpoint_path_hint",
            "application_remarks",
        ]
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise ValueError(f"missing required checkpoint metadata fields: {', '.join(sorted(missing))}")

        cls_payload = dict(payload)
        classes = cls_payload.get("classes", [])
        if not isinstance(classes, list):
            raise ValueError("classes must be a list of objects")
        cls_payload["classes"] = tuple(dict(item) for item in classes)
        return cls(**cls_payload)


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root by scanning parent folders."""

    cur = (start or Path(__file__)).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "frozen_checkpoints").exists() and (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("could not locate repository root containing frozen_checkpoints/")


def registry_path(start: Path | None = None) -> Path:
    """Return canonical model registry JSON path."""

    return find_repo_root(start) / "frozen_checkpoints" / "model_registry.json"


def load_frozen_checkpoint_records(path: str | Path | None = None) -> list[FrozenCheckpointRecord]:
    """Load frozen-checkpoint metadata records from JSON registry."""

    if path:
        reg_path = Path(path)
    else:
        try:
            reg_path = registry_path()
        except FileNotFoundError:
            return []
    if not reg_path.exists():
        return []

    payload = json.loads(reg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen checkpoint registry payload must be an object")
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError(
            f"unsupported frozen checkpoint registry schema: {payload.get('schema_version')!r}; "
            f"expected {REGISTRY_SCHEMA!r}"
        )

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("frozen checkpoint registry 'models' must be a list")
    return [FrozenCheckpointRecord.from_dict(dict(item)) for item in models]


def frozen_checkpoint_map(path: str | Path | None = None) -> dict[str, FrozenCheckpointRecord]:
    """Load registry records keyed by model identifier."""

    out: dict[str, FrozenCheckpointRecord] = {}
    for rec in load_frozen_checkpoint_records(path):
        out[rec.model_id] = rec
    return out
