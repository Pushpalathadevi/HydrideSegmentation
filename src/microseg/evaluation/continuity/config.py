"""Configuration for the Hydride Connectivity Index (HCI) analysis.

All geometric thresholds are expressed in physical units (µm, µm²) when the
pixel size is known, and fall back to documented pixel values otherwise. The
bridging range ``max_bridge_distance`` defaults to ``"auto"``: one fifth of the
length of the smallest hydride that survives mask clean-up (owner decision,
2026-09-27). Because that rule is a ratio of two lengths measured on the same
image, it is independent of magnification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

FORMULATION_ID = "hci.v1"
RESULT_SCHEMA_VERSION = "microseg.hci_result.v1"
DIRECTIONS = ("radial", "circumferential", "isotropic")


@dataclass(frozen=True)
class ContinuityAnalysisConfig:
    """Parameters of the HCI analysis (see ``docs/hci_specification.md`` §7).

    Parameters
    ----------
    foreground_class_indices:
        Class indices treated as hydride when an indexed mask is supplied.
        Boolean masks are used as-is; for other integer masks any listed index
        is foreground.
    min_feature_area_um2, min_feature_area_px:
        Components smaller than this are removed before analysis. The µm²
        value is used when the pixel size is known, the pixel value otherwise.
    max_hole_area_um2, max_hole_area_px:
        Interior holes smaller than this are filled.
    resolution_floor_px:
        Lower bound (pixels) on both clean-up thresholds: features and holes
        smaller than 2 × 2 pixels are not resolvable, whatever the µm value.
    max_bridge_distance:
        Upper limit ``δ_max`` of the integral that defines the HCI, in µm
        (or pixels when uncalibrated), or ``"auto"``.
    auto_bridge_fraction:
        With ``"auto"``, ``δ_max`` equals this fraction of the reference
        hydride length chosen by ``auto_size_statistic``.
    auto_size_statistic, auto_size_percentile:
        Which accepted-hydride length the ``"auto"`` rule uses: ``"min"`` (the
        smallest accepted hydride, the owner's default) or ``"percentile"``
        (the given percentile of the maximum Feret lengths). A percentile is
        less sensitive to a single small fragment.
    report_distance:
        Association distance ``δ0`` for the reported cluster table,
        ``C_u(δ0)`` and ``ξ_u``; ``"auto"`` uses ``δ_max``.
    reference_length_radial_um, reference_length_circumferential_um:
        Reference lengths ``L_u``. ``None`` uses the field extent. Set the
        radial value to the wall thickness for through-wall assessment.
    path_enabled, path_hydride_cost:
        Companion path-continuity descriptor and its hydride cost ``ε``
        (0 = pure matrix ligament, 0.02 = RHCP toughness ratio).
    topology_enabled:
        Companion skeleton-topology descriptors.
    curve_points:
        Number of points of the reported ``C_u(δ)`` grid over ``[0, δ_max]``
        (the HCI itself is integrated exactly, not on this grid).
    max_pixels:
        Refuse larger masks rather than exhausting memory.
    """

    formulation_id: str = FORMULATION_ID
    foreground_class_indices: tuple[int, ...] = (1,)
    min_feature_area_um2: float = 5.0
    min_feature_area_px: float = 20.0
    max_hole_area_um2: float = 5.0
    max_hole_area_px: float = 20.0
    resolution_floor_px: float = 4.0
    max_bridge_distance: float | str = "auto"
    auto_bridge_fraction: float = 0.2
    auto_size_statistic: str = "min"
    auto_size_percentile: float = 10.0
    report_distance: float | str = "auto"
    reference_length_radial_um: float | None = None
    reference_length_circumferential_um: float | None = None
    path_enabled: bool = True
    path_hydride_cost: float = 0.0
    topology_enabled: bool = True
    curve_points: int = 41
    max_pixels: int = 64_000_000

    def __post_init__(self) -> None:
        if self.formulation_id != FORMULATION_ID:
            raise ValueError(f"unsupported formulation_id {self.formulation_id!r}; expected {FORMULATION_ID!r}")
        for name in ("min_feature_area_um2", "min_feature_area_px", "max_hole_area_um2", "max_hole_area_px", "resolution_floor_px"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("max_bridge_distance", "report_distance"):
            value = getattr(self, name)
            if isinstance(value, str):
                if value != "auto":
                    raise ValueError(f"{name} must be a non-negative number or 'auto'")
            elif float(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        if isinstance(self.max_bridge_distance, (int, float)) and float(self.max_bridge_distance) == 0:
            raise ValueError("max_bridge_distance must be > 0 (the HCI integrates over [0, max_bridge_distance])")
        if not 0 < float(self.auto_bridge_fraction) <= 5:
            raise ValueError("auto_bridge_fraction must be in (0, 5]")
        if self.auto_size_statistic not in ("min", "percentile"):
            raise ValueError("auto_size_statistic must be 'min' or 'percentile'")
        if not 0 <= float(self.auto_size_percentile) <= 100:
            raise ValueError("auto_size_percentile must be in [0, 100]")
        if not 0 <= float(self.path_hydride_cost) <= 1:
            raise ValueError("path_hydride_cost must be in [0, 1]")
        if int(self.curve_points) < 2:
            raise ValueError("curve_points must be >= 2")
        for name in ("reference_length_radial_um", "reference_length_circumferential_um"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} must be > 0 or None")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ContinuityAnalysisConfig":
        """Build from a YAML/JSON mapping, rejecting unknown keys and coercing types."""

        payload = dict(payload or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown HCI config keys: {unknown}")
        data: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "foreground_class_indices":
                data[key] = tuple(int(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
            elif key in ("max_bridge_distance", "report_distance"):
                data[key] = "auto" if value is None or str(value).strip().lower() == "auto" else float(value)
            elif key in ("reference_length_radial_um", "reference_length_circumferential_um"):
                data[key] = None if value in (None, "", "none", "null") else float(value)
            elif key in ("path_enabled", "topology_enabled"):
                data[key] = value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
            elif key in ("curve_points", "max_pixels"):
                data[key] = int(value)
            elif key in ("formulation_id", "auto_size_statistic"):
                data[key] = str(value)
            else:
                data[key] = float(value)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["foreground_class_indices"] = list(self.foreground_class_indices)
        return out


@dataclass
class Calibration:
    """Spatial calibration of a mask.

    ``pixel_size_um`` of ``None`` means uncalibrated: every length is then
    reported in pixels and results are marked not comparable across
    magnifications.
    """

    pixel_size_um: float | None = None
    source: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.pixel_size_um is not None and self.pixel_size_um > 0

    @property
    def unit(self) -> str:
        return "um" if self.calibrated else "px"

    @property
    def scale(self) -> float:
        """Length of one pixel in the reporting unit."""

        return float(self.pixel_size_um) if self.calibrated else 1.0
