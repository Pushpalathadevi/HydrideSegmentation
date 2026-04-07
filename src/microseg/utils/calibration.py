"""Spatial-calibration helpers for GUI and reporting workflows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class SpatialCalibration:
    """Pixel-to-physical calibration metadata.

    Parameters
    ----------
    microns_per_pixel:
        Scalar micron-per-pixel ratio used for physical-unit conversion.
    source:
        Calibration source label (`manual_line`, `tiff_metadata`, `dpi_metadata`).
    notes:
        Optional human-readable note for provenance.
    x_microns_per_pixel:
        Optional x-axis micron-per-pixel ratio.
    y_microns_per_pixel:
        Optional y-axis micron-per-pixel ratio.
    """

    microns_per_pixel: float
    source: str
    notes: str = ""
    x_microns_per_pixel: float | None = None
    y_microns_per_pixel: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly dictionary payload."""

        return {
            "microns_per_pixel": float(self.microns_per_pixel),
            "source": str(self.source),
            "notes": str(self.notes),
            "x_microns_per_pixel": None if self.x_microns_per_pixel is None else float(self.x_microns_per_pixel),
            "y_microns_per_pixel": None if self.y_microns_per_pixel is None else float(self.y_microns_per_pixel),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SpatialCalibration":
        """Parse calibration payload from dictionary."""

        return cls(
            microns_per_pixel=float(payload.get("microns_per_pixel", 0.0)),
            source=str(payload.get("source", "unknown")),
            notes=str(payload.get("notes", "")),
            x_microns_per_pixel=(
                None
                if payload.get("x_microns_per_pixel") is None
                else float(payload.get("x_microns_per_pixel"))
            ),
            y_microns_per_pixel=(
                None
                if payload.get("y_microns_per_pixel") is None
                else float(payload.get("y_microns_per_pixel"))
            ),
        )


_UNIT_TO_MICRONS = {
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "mm": 1000.0,
    "nm": 0.001,
}


def convert_known_length_to_microns(length_value: float, length_unit: str) -> float:
    """Convert a known physical length to microns.

    Parameters
    ----------
    length_value:
        Numeric known length.
    length_unit:
        Length unit (`um`, `mm`, `nm`).

    Returns
    -------
    float
        Length in microns.
    """

    unit_key = str(length_unit).strip().lower()
    if unit_key not in _UNIT_TO_MICRONS:
        raise ValueError(f"Unsupported length unit: {length_unit}")
    value = float(length_value)
    if value <= 0:
        raise ValueError("Known length must be > 0")
    return value * _UNIT_TO_MICRONS[unit_key]


def calibration_from_manual_line(
    pixel_distance: float,
    known_length_value: float,
    known_length_unit: str = "um",
) -> SpatialCalibration:
    """Build calibration from a manually drawn reference line.

    Parameters
    ----------
    pixel_distance:
        Pixel distance of the drawn line.
    known_length_value:
        User-specified physical length corresponding to the line.
    known_length_unit:
        Unit for `known_length_value` (`um`, `mm`, `nm`).

    Returns
    -------
    SpatialCalibration
        Manual line calibration object.
    """

    px = float(pixel_distance)
    if px <= 0:
        raise ValueError("Pixel distance must be > 0 for calibration")
    known_um = convert_known_length_to_microns(known_length_value, known_length_unit)
    um_per_px = known_um / px
    note = f"known_length={known_length_value:g} {known_length_unit} over {px:.3f} px"
    return SpatialCalibration(
        microns_per_pixel=float(um_per_px),
        source="manual_line",
        notes=note,
        x_microns_per_pixel=float(um_per_px),
        y_microns_per_pixel=float(um_per_px),
    )


def _to_float_ratio(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if val > 0 else None
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0:
                return None
            val = num_f / den_f
            return val if val > 0 else None
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        parts = [p.strip() for p in text.split("/", 1)]
        if len(parts) == 2:
            try:
                num_f = float(parts[0])
                den_f = float(parts[1])
                if den_f == 0:
                    return None
                val = num_f / den_f
                return val if val > 0 else None
            except Exception:
                return None
    try:
        val = float(text)
        return val if val > 0 else None
    except Exception:
        return None


def _description_to_um_per_px(description: str) -> float | None:
    text = str(description or "")
    if not text:
        return None
    patterns = [
        r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?:um|µm|micron(?:s)?)\s*(?:/|per)\s*(?:px|pixel)",
        r"(?:microns?_per_pixel|pixel_size_um)\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group("value"))
            if val > 0:
                return val
        except Exception:
            continue
    return None


def metadata_calibration_from_image(image_path: str | Path) -> SpatialCalibration | None:
    """Try to derive micron-per-pixel calibration from image metadata.

    TIFF-specific tags are checked first, then generic DPI metadata.

    Parameters
    ----------
    image_path:
        Input image path.

    Returns
    -------
    SpatialCalibration | None
        Parsed calibration when metadata is usable; otherwise ``None``.
    """

    path = Path(image_path)
    if not path.exists():
        return None

    try:
        with Image.open(path) as img:
            tag_v2 = getattr(img, "tag_v2", None)
            if tag_v2 is not None:
                # TIFF baseline tags:
                # 282=XResolution, 283=YResolution, 296=ResolutionUnit, 270=ImageDescription
                x_res = _to_float_ratio(tag_v2.get(282))
                y_res = _to_float_ratio(tag_v2.get(283))
                unit = tag_v2.get(296)
                desc = str(tag_v2.get(270, ""))

                if not desc:
                    desc = str(img.info.get("description", ""))
                desc_um_per_px = _description_to_um_per_px(desc)
                if desc_um_per_px is not None:
                    return SpatialCalibration(
                        microns_per_pixel=float(desc_um_per_px),
                        source="tiff_metadata",
                        notes="parsed from image description",
                        x_microns_per_pixel=float(desc_um_per_px),
                        y_microns_per_pixel=float(desc_um_per_px),
                    )

                unit_um = None
                unit_int = int(unit) if unit is not None else 1
                if unit_int == 2:  # inch
                    unit_um = 25400.0
                elif unit_int == 3:  # centimeter
                    unit_um = 10000.0

                if unit_um and (x_res or y_res):
                    x_um_per_px = (unit_um / x_res) if x_res else None
                    y_um_per_px = (unit_um / y_res) if y_res else None
                    vals = [v for v in [x_um_per_px, y_um_per_px] if v is not None and v > 0]
                    if vals:
                        mean_um_per_px = float(sum(vals) / len(vals))
                        return SpatialCalibration(
                            microns_per_pixel=mean_um_per_px,
                            source="tiff_metadata",
                            notes=f"resolution_unit={unit_int}",
                            x_microns_per_pixel=x_um_per_px,
                            y_microns_per_pixel=y_um_per_px,
                        )

            dpi = img.info.get("dpi")
            if isinstance(dpi, tuple) and len(dpi) >= 1:
                x_dpi = _to_float_ratio(dpi[0])
                y_dpi = _to_float_ratio(dpi[1] if len(dpi) > 1 else dpi[0])
                vals = []
                if x_dpi:
                    vals.append(25400.0 / x_dpi)
                if y_dpi:
                    vals.append(25400.0 / y_dpi)
                if vals:
                    mean_um = float(sum(vals) / len(vals))
                    return SpatialCalibration(
                        microns_per_pixel=mean_um,
                        source="dpi_metadata",
                        notes="derived from dpi metadata (inch assumption)",
                        x_microns_per_pixel=(25400.0 / x_dpi) if x_dpi else None,
                        y_microns_per_pixel=(25400.0 / y_dpi) if y_dpi else None,
                    )
    except Exception:
        return None
    return None

