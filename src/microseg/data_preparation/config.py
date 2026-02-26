"""Configuration models for segmentation dataset preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class MorphologyConfig:
    """Optional morphology cleanup settings."""

    open_kernel: int = 0
    close_kernel: int = 0
    remove_small_components: int = 0
    fill_holes: bool = False


@dataclass
class DebugConfig:
    """Debug inspection and subset processing settings."""

    enabled: bool = False
    limit_pairs: int = 100
    inspection_limit: int = 8
    show_plots: bool = False
    minimal_exports: bool = True
    skip_sanity_checks: bool = True
    draw_contours: bool = False


@dataclass
class DatasetPrepConfig:
    """Resolved configuration for data preparation pipeline."""

    input_dir: str
    output_dir: str
    styles: list[str] = field(default_factory=lambda: ["oxford", "mado"])
    train_pct: float = 0.8
    val_pct: float = 0.1
    seed: int = 42
    dry_run: bool = False
    strict_pairing: bool = True
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    mask_extensions: list[str] = field(default_factory=lambda: [".png", ".tif", ".tiff", ".jpg", ".jpeg"])
    mask_name_patterns: list[str] = field(default_factory=lambda: ["{stem}.png", "{stem}_mask.png", "{stem}.tif", "{stem}.tiff"])
    binarization_mode: Literal["nonzero", "threshold", "value_equals", "otsu", "percentile"] = "nonzero"
    threshold: int = 127
    threshold_strict: bool = False
    foreground_values: list[int] = field(default_factory=lambda: [255])
    percentile: float = 90.0
    invert_mask: bool = False
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    target_size: tuple[int, int] = (512, 512)
    resize_policy: Literal["letterbox_pad", "center_crop", "stretch", "keep_aspect_no_pad"] = "letterbox_pad"
    image_pad_mode: Literal["constant", "edge", "reflect"] = "constant"
    image_pad_value: int = 0
    image_interpolation: Literal["linear", "area", "cubic"] = "area"
    image_ext: str = ".png"
    mask_ext: str = ".png"
    debug_ext: str = ".png"
    mask_foreground_value: int = 255
    path_mode: Literal["absolute", "relative"] = "relative"
    skip_sanity: bool = False
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetPrepConfig":
        data = dict(raw)
        morphology = MorphologyConfig(**data.pop("morphology", {}))
        debug = DebugConfig(**data.pop("debug", {}))
        cfg = cls(**data)
        cfg.morphology = morphology
        cfg.debug = debug
        return cfg

    @classmethod
    def from_yaml_or_default(cls, config_path: str | None, fallback: dict[str, Any]) -> "DatasetPrepConfig":
        if config_path:
            path = Path(config_path)
            if path.exists():
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return cls.from_dict({**fallback, **loaded})
        return cls.from_dict(fallback)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_size"] = list(self.target_size)
        return result
