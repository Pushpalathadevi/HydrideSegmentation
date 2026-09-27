"""Result contract ``microseg.hci_result.v1`` for the HCI analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RESULT_SCHEMA_VERSION


def _clean(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats into JSON-safe values."""

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if not math.isfinite(f) else f
    return value


@dataclass
class ContinuityClusterResult:
    """One single-linkage cluster at the report distance ``δ0``."""

    cluster_id: str
    members: list[int]
    area: float
    coverage_radial: float
    coverage_circumferential: float
    feret: float
    weight: dict[str, float]
    touches_border: bool
    bridges: list[dict[str, Any]]
    topology: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "cluster_id": self.cluster_id,
                "members": self.members,
                "area": self.area,
                "extent": {
                    "radial_coverage": self.coverage_radial,
                    "circumferential_coverage": self.coverage_circumferential,
                    "feret": self.feret,
                },
                "weight": self.weight,
                "touches_border": self.touches_border,
                "bridges": self.bridges,
                "topology": self.topology,
            }
        )


@dataclass
class ContinuityAnalysisResult:
    """Full HCI result.

    ``specimen``, ``curve`` and ``clusters`` are the scientific payload.
    ``arrays`` holds images for overlays and is excluded from ``to_dict``.
    """

    status: str
    status_reason: str | None
    unit: str
    specimen: dict[str, Any]
    curve: dict[str, Any]
    clusters: list[ContinuityClusterResult]
    parameters: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]
    quality_flags: list[str]
    warnings: list[str]
    arrays: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def headline(self) -> dict[str, float | None]:
        return dict(self.specimen.get("HCI", {})) if self.ok else {"radial": None, "circumferential": None, "isotropic": None}

    def to_dict(self, *, include_clusters: bool = True) -> dict[str, Any]:
        return _clean(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "formulation_id": self.parameters.get("formulation_id"),
                "status": self.status,
                "status_reason": self.status_reason,
                "unit": self.unit,
                "specimen": self.specimen,
                "curve": self.curve,
                "clusters": [c.to_dict() for c in self.clusters] if include_clusters else [],
                "n_clusters_reported": len(self.clusters),
                "parameters": self.parameters,
                "calibration": self.calibration,
                "provenance": self.provenance,
                "quality_flags": self.quality_flags,
                "warnings": self.warnings,
            }
        )
