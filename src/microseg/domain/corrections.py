"""Data contracts for correction sessions and exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CorrectionAction:
    """Represents one user correction action applied to a mask."""

    action_type: Literal["brush", "polygon"]
    mode: Literal["add", "erase"]
    params: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = ""


@dataclass
class CorrectionSessionReport:
    """Summary metadata for a correction session."""

    initial_foreground_pixels: int
    current_foreground_pixels: int
    actions_applied: int
    undo_depth: int
    redo_depth: int


@dataclass(frozen=True)
class CorrectionExportRecord:
    """Schema contract for one corrected sample export."""

    schema_version: str
    sample_id: str
    source_image_path: str
    model_id: str
    model_name: str
    run_id: str
    created_utc: str
    annotator: str
    notes: str
    files: dict[str, str]
    metrics: dict[str, float | int]
