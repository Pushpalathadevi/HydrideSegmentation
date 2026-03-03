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
    max_val_examples: int | None = None
    max_test_examples: int | None = None
    seed: int = 42
    dry_run: bool = False
    strict_pairing: bool = True
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    mask_extensions: list[str] = field(default_factory=lambda: [".png", ".tif", ".tiff", ".jpg", ".jpeg"])
    mask_name_patterns: list[str] = field(default_factory=lambda: ["{stem}.png", "{stem}_mask.png", "{stem}.tif", "{stem}.tiff"])
    binarization_mode: Literal["nonzero", "threshold", "value_equals", "otsu", "percentile"] = "nonzero"
    rgb_mask_mode: bool = False
    mask_r_min: int = 200
    mask_g_max: int = 60
    mask_b_max: int = 60
    enforce_gb_thresholds: bool = True
    allow_red_dominance_fallback: bool = True
    mask_red_min_fallback: int = 16
    mask_red_dominance_margin: int = 8
    mask_red_dominance_ratio: float = 1.5
    auto_otsu_for_noisy_grayscale: bool = True
    noisy_grayscale_low_max: int = 5
    noisy_grayscale_high_min: int = 200
    noisy_grayscale_min_extreme_ratio: float = 0.98
    threshold: int = 127
    threshold_strict: bool = False
    foreground_values: list[int] = field(default_factory=lambda: [255])
    percentile: float = 90.0
    invert_mask: bool = False
    empty_mask_action: Literal["warn", "error"] = "warn"
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    target_size: tuple[int, int] = (512, 512)
    resize_policy: Literal["letterbox_pad", "center_crop", "stretch", "keep_aspect_no_pad", "short_side_to_target_crop"] = "letterbox_pad"
    crop_mode_train: Literal["center", "random"] = "random"
    crop_mode_eval: Literal["center", "random"] = "center"
    progress_log_interval: int = 20
    qa_report_name: str = "dataset_qa_report.json"
    html_report_name: str = "dataset_qa_report.html"
    debug_sample_count: int = 0
    image_pad_mode: Literal["constant", "edge", "reflect"] = "constant"
    image_pad_value: int = 0
    image_interpolation: Literal["linear", "area", "cubic"] = "area"
    image_ext: str = ".png"
    mask_ext: str = ".png"
    debug_ext: str = ".png"
    mask_foreground_value: int = 255
    expected_raw_binary_values: list[int] = field(default_factory=lambda: [0, 255])
    path_mode: Literal["absolute", "relative"] = "relative"
    skip_sanity: bool = False
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetPrepConfig":
        data = dict(raw)
        if "val_max_examples" in data and "max_val_examples" not in data:
            data["max_val_examples"] = data.get("val_max_examples")
        if "test_max_examples" in data and "max_test_examples" not in data:
            data["max_test_examples"] = data.get("test_max_examples")
        for key in ("max_val_examples", "max_test_examples"):
            raw_value = data.get(key)
            if raw_value is None or raw_value == "":
                data[key] = None
                continue
            value = int(raw_value)
            if value < 0:
                raise ValueError(f"{key} must be >= 0")
            data[key] = value
        if isinstance(data.get("target_size"), int):
            size = int(data["target_size"])
            data["target_size"] = (size, size)
        elif isinstance(data.get("target_size"), list):
            target = data["target_size"]
            if len(target) == 1:
                data["target_size"] = (int(target[0]), int(target[0]))
            elif len(target) == 2:
                data["target_size"] = (int(target[0]), int(target[1]))
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
