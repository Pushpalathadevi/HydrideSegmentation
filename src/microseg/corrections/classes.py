"""Class-index palette contracts and mask colorization utilities."""

from __future__ import annotations

from dataclasses import dataclass

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


def colorize_index_mask(mask: np.ndarray, class_map: SegmentationClassMap) -> np.ndarray:
    """Convert indexed mask to RGB using class map colors."""

    idx = to_index_mask(mask)
    out = np.zeros((idx.shape[0], idx.shape[1], 3), dtype=np.uint8)
    for cls in class_map.classes:
        out[idx == cls.index] = np.array(cls.color_rgb, dtype=np.uint8)
    return out
