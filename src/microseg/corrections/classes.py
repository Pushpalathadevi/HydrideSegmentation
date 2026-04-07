"""Class-index palette contracts and mask colorization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SegmentationClass:
    """One semantic class in an indexed segmentation mask."""

    index: int
    name: str
    color_rgb: tuple[int, int, int]
    description: str = ""


@dataclass(frozen=True)
class SegmentationClassMap:
    """Validated class map for indexed mask editing and export."""

    classes: tuple[SegmentationClass, ...]

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("class map must include at least one class")
        indexes = [c.index for c in self.classes]
        if len(set(indexes)) != len(indexes):
            raise ValueError("class indexes must be unique")
        if 0 not in indexes:
            raise ValueError("class map must contain background index 0")
        for c in self.classes:
            if c.index < 0 or c.index > 255:
                raise ValueError(f"invalid class index: {c.index}")
            if len(c.color_rgb) != 3:
                raise ValueError(f"invalid color for class {c.name}")
            for ch in c.color_rgb:
                if ch < 0 or ch > 255:
                    raise ValueError(f"color channel out of range for class {c.name}")

    def class_for_index(self, index: int) -> SegmentationClass:
        """Return class metadata for given index."""

        for c in self.classes:
            if c.index == index:
                return c
        raise KeyError(f"unknown class index: {index}")

    def indexes(self) -> list[int]:
        """Return sorted class indexes."""

        return sorted(c.index for c in self.classes)

    def as_dict(self) -> list[dict[str, object]]:
        """Serialize class map to JSON-compatible list."""

        return [
            {
                "index": c.index,
                "name": c.name,
                "color_rgb": list(c.color_rgb),
                "description": c.description,
            }
            for c in self.classes
        ]

    @classmethod
    def from_dict(cls, payload: list[dict[str, object]]) -> SegmentationClassMap:
        """Deserialize class map from JSON-compatible list."""

        classes: list[SegmentationClass] = []
        for raw in payload:
            color = raw.get("color_rgb", [0, 0, 0])
            color_t = tuple(int(v) for v in color)
            classes.append(
                SegmentationClass(
                    index=int(raw["index"]),
                    name=str(raw.get("name", f"class_{raw['index']}")),
                    color_rgb=(color_t[0], color_t[1], color_t[2]),
                    description=str(raw.get("description", "")),
                )
            )
        return cls(tuple(classes))


DEFAULT_CLASS_MAP = SegmentationClassMap(
    classes=(
        SegmentationClass(index=0, name="background", color_rgb=(0, 0, 0), description="Background"),
        SegmentationClass(index=1, name="feature", color_rgb=(255, 0, 0), description="Foreground feature"),
    )
)

REPO_DEFAULT_CLASS_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "segmentation_classes.json"


def _hex_to_rgb(text: str) -> tuple[int, int, int]:
    value = str(text).strip()
    if not value.startswith("#") or len(value) != 7:
        raise ValueError(f"invalid color_hex value: {text!r}")
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError as exc:
        raise ValueError(f"invalid color_hex value: {text!r}") from exc


def _parse_color_rgb(raw: object, *, context: str) -> tuple[int, int, int]:
    if isinstance(raw, str):
        return _hex_to_rgb(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        vals = tuple(int(v) for v in raw)
        for v in vals:
            if v < 0 or v > 255:
                raise ValueError(f"color channel out of range for {context}: {vals}")
        return (vals[0], vals[1], vals[2])
    raise ValueError(f"invalid color specification for {context}: {raw!r}")


def _class_map_from_payload(payload: object) -> SegmentationClassMap:
    if isinstance(payload, dict):
        classes_raw = payload.get("classes", [])
    else:
        classes_raw = payload
    if not isinstance(classes_raw, list):
        raise ValueError("class map JSON must contain a 'classes' list")
    classes: list[SegmentationClass] = []
    for idx, raw in enumerate(classes_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"class entry at index {idx} must be an object")
        class_index = int(raw["index"])
        class_name = str(raw.get("name", f"class_{class_index}")).strip() or f"class_{class_index}"
        if "color_rgb" in raw:
            color_rgb = _parse_color_rgb(raw.get("color_rgb"), context=f"class[{class_index}].color_rgb")
        elif "color_hex" in raw:
            color_rgb = _parse_color_rgb(raw.get("color_hex"), context=f"class[{class_index}].color_hex")
        else:
            raise ValueError(f"class[{class_index}] must include color_rgb or color_hex")
        classes.append(
            SegmentationClass(
                index=class_index,
                name=class_name,
                color_rgb=color_rgb,
                description=str(raw.get("description", "")),
            )
        )
    return SegmentationClassMap(tuple(classes))


def load_class_map(path: str | Path) -> SegmentationClassMap:
    """Load segmentation class map from JSON file."""

    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists() or not cfg_path.is_file():
        raise FileNotFoundError(f"class map config not found: {cfg_path}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _class_map_from_payload(payload)


def resolve_class_map(class_map_path: str | Path | None = None) -> tuple[SegmentationClassMap, str]:
    """Resolve class map using explicit override, env override, repo default, then builtin fallback."""

    explicit = str(class_map_path).strip() if class_map_path is not None else ""
    if explicit:
        resolved = load_class_map(explicit)
        return resolved, f"explicit:{Path(explicit).expanduser().resolve()}"

    import os

    env_path = str(os.getenv("MICROSEG_CLASS_MAP_PATH", "")).strip()
    if env_path:
        resolved = load_class_map(env_path)
        return resolved, f"env:MICROSEG_CLASS_MAP_PATH:{Path(env_path).expanduser().resolve()}"

    if REPO_DEFAULT_CLASS_MAP_PATH.exists() and REPO_DEFAULT_CLASS_MAP_PATH.is_file():
        resolved = load_class_map(REPO_DEFAULT_CLASS_MAP_PATH)
        return resolved, f"repo_default:{REPO_DEFAULT_CLASS_MAP_PATH.resolve()}"

    return DEFAULT_CLASS_MAP, "builtin_default"


def class_map_to_colormap(class_map: SegmentationClassMap) -> dict[str, list[int]]:
    """Convert class map into index->RGB mapping payload."""

    colormap: dict[str, list[int]] = {}
    used_rgb: set[tuple[int, int, int]] = set()
    for cls in class_map.classes:
        if cls.color_rgb in used_rgb:
            raise ValueError(f"duplicate class color in class map: {cls.color_rgb}")
        used_rgb.add(cls.color_rgb)
        colormap[str(int(cls.index))] = [int(cls.color_rgb[0]), int(cls.color_rgb[1]), int(cls.color_rgb[2])]
    return colormap


def to_index_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask array to uint8 indexed mask.

    Notes
    -----
    If values are only {0, 255}, this is converted to {0, 1}.
    """

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    uniq = set(np.unique(arr).tolist())
    if uniq.issubset({0, 255}):
        return (arr > 0).astype(np.uint8)
    return arr


def binary_remapped_foreground_values(
    mask: np.ndarray,
    *,
    mode: Literal["off", "two_value_zero_background", "nonzero_foreground"] = "off",
) -> tuple[int, ...]:
    """Return foreground values that will be remapped to class 1 by binary normalization mode."""

    idx = to_index_mask(mask)
    if mode == "off":
        return tuple()
    if mode == "two_value_zero_background":
        unique_vals = np.unique(idx)
        if unique_vals.size == 2 and int(unique_vals.min()) == 0 and int(unique_vals.max()) != 1:
            return (int(unique_vals.max()),)
        return tuple()
    if mode == "nonzero_foreground":
        values = sorted(int(v) for v in np.unique(idx).tolist() if int(v) not in {0, 1})
        return tuple(values)
    raise ValueError(f"unsupported binary mask normalization mode: {mode}")


def normalize_binary_index_mask(
    mask: np.ndarray,
    *,
    mode: Literal["off", "two_value_zero_background", "nonzero_foreground"] = "off",
) -> np.ndarray:
    """Normalize binary masks with configurable auto-discovery behavior.

    Parameters
    ----------
    mask:
        Input 2D mask image array.
    mode:
        Binary normalization mode:
        - ``off``: keep indexed values unchanged except canonical ``{0,255}`` -> ``{0,1}`` via
          :func:`to_index_mask`.
        - ``two_value_zero_background``: if mask has exactly two unique values and one is
          background ``0``, map all non-zero pixels to class ``1``.
        - ``nonzero_foreground``: map every non-zero indexed value to class ``1``.

    Returns
    -------
    np.ndarray
        Normalized uint8 indexed mask.
    """

    idx = to_index_mask(mask)
    if mode == "off":
        return idx
    if mode == "two_value_zero_background":
        unique_vals = np.unique(idx)
        if unique_vals.size == 2 and int(unique_vals.min()) == 0 and int(unique_vals.max()) != 1:
            return (idx > 0).astype(np.uint8)
        return idx
    if mode == "nonzero_foreground":
        return (idx > 0).astype(np.uint8)
    raise ValueError(f"unsupported binary mask normalization mode: {mode}")


def colorize_index_mask(mask: np.ndarray, class_map: SegmentationClassMap) -> np.ndarray:
    """Convert indexed mask to RGB using class map colors."""

    idx = to_index_mask(mask)
    out = np.zeros((idx.shape[0], idx.shape[1], 3), dtype=np.uint8)
    for cls in class_map.classes:
        out[idx == cls.index] = np.array(cls.color_rgb, dtype=np.uint8)
    return out
