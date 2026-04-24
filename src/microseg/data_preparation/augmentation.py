"""Deterministic dataset-preparation augmentation primitives and runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from src.microseg.data_preparation.exporters import write_image


AugmentationScope = Literal["image_only", "paired_geometry"]
AugmentationStage = Literal["pre_resize", "post_resize"]


@dataclass(frozen=True)
class AugmentationDebugConfig:
    """Debug artifact controls for augmented sample inspection."""

    enabled: bool = False
    max_samples: int = 8


@dataclass(frozen=True)
class AugmentationOperationConfig:
    """Configuration for one augmentation operation."""

    name: str
    enabled: bool = True
    probability: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentationConfig:
    """Dataset-level augmentation controls."""

    enabled: bool = False
    seed: int = 42
    stage: AugmentationStage = "post_resize"
    apply_splits: tuple[str, ...] = ("train",)
    variants_per_sample: int = 1
    operations: tuple[AugmentationOperationConfig, ...] = ()
    debug: AugmentationDebugConfig = field(default_factory=AugmentationDebugConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-safe configuration payload."""

        payload = asdict(self)
        payload["apply_splits"] = list(self.apply_splits)
        payload["operations"] = [
            {
                "name": op.name,
                "enabled": bool(op.enabled),
                "probability": float(op.probability),
                "parameters": dict(op.parameters),
            }
            for op in self.operations
        ]
        return payload


@dataclass(frozen=True)
class OperationApplicationRecord:
    """Recorded execution details for one applied augmentation operation."""

    name: str
    scope: AugmentationScope
    probability: float
    parameters: dict[str, Any]


@dataclass(frozen=True)
class AugmentedVariantRecord:
    """One generated augmented variant plus machine-readable provenance."""

    variant_index: int
    sample_seed: int
    requested_stage: AugmentationStage
    resolved_stage: str
    split: str
    source_name: str
    applied_operations: tuple[OperationApplicationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""

        return {
            "variant_index": int(self.variant_index),
            "sample_seed": int(self.sample_seed),
            "requested_stage": str(self.requested_stage),
            "resolved_stage": str(self.resolved_stage),
            "split": str(self.split),
            "source_name": str(self.source_name),
            "applied_operations": [
                {
                    "name": op.name,
                    "scope": op.scope,
                    "probability": float(op.probability),
                    "parameters": _json_safe(op.parameters),
                }
                for op in self.applied_operations
            ],
        }


@dataclass(frozen=True)
class AugmentedVariant:
    """Augmented image/mask payload."""

    image: np.ndarray
    mask: np.ndarray
    metadata: AugmentedVariantRecord


class AugmentationOperation(Protocol):
    """Contract for one deterministic augmentation strategy."""

    name: str
    scope: AugmentationScope

    def apply(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        rng: random.Random,
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply the configured operation and return new arrays plus metadata."""


def _as_int_pair(value: Any, *, field_name: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo = int(value[0])
        hi = int(value[1])
    else:
        raise ValueError(f"{field_name} must be a 2-item list/tuple")
    if hi < lo:
        raise ValueError(f"{field_name} max must be >= min")
    return lo, hi


def _is_two_item_range(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _sample_int_or_range(value: Any, *, field_name: str, rng: random.Random, default: int) -> tuple[int, Any]:
    """Return fixed integer or sampled inclusive integer range value plus config echo."""

    raw = default if value is None else value
    if _is_two_item_range(raw):
        lo, hi = _as_int_pair(raw, field_name=field_name)
        return int(rng.randint(lo, hi)), [int(lo), int(hi)]
    try:
        return int(raw), int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer or 2-item integer range") from exc


def _sample_float_or_range(
    value: Any,
    *,
    field_name: str,
    rng: random.Random,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, Any]:
    """Return fixed float or sampled inclusive float range value plus config echo."""

    raw = default if value is None else value
    if _is_two_item_range(raw):
        try:
            lo = float(raw[0])
            hi = float(raw[1])
        except Exception as exc:
            raise ValueError(f"{field_name} must be a number or 2-item numeric range") from exc
        if hi < lo:
            raise ValueError(f"{field_name} max must be >= min")
        value_f = float(rng.uniform(lo, hi))
        config_echo: Any = [float(lo), float(hi)]
    else:
        try:
            value_f = float(raw)
        except Exception as exc:
            raise ValueError(f"{field_name} must be a number or 2-item numeric range") from exc
        config_echo = float(raw)
    if minimum is not None and value_f < float(minimum):
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value_f > float(maximum):
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value_f, config_echo


def _sample_odd_kernel(
    parameters: dict[str, Any],
    *,
    rng: random.Random,
    default: int = 9,
) -> tuple[int, dict[str, Any]]:
    """Sample an odd blur kernel from scalar ``kernel_size`` or legacy range."""

    source = "kernel_size"
    raw = parameters.get("kernel_size", None)
    if raw is None:
        raw = parameters.get("kernel_size_range", [3, default])
        source = "kernel_size_range"

    if _is_two_item_range(raw):
        lo, hi = _as_int_pair(raw, field_name=f"blur.{source}")
        odds = [value for value in range(lo, hi + 1) if value % 2 == 1 and value > 0]
        if not odds:
            raise ValueError(f"blur.{source} must include at least one positive odd integer")
        sampled = int(rng.choice(odds))
        configured: Any = [int(lo), int(hi)]
    else:
        try:
            sampled = int(raw)
        except Exception as exc:
            raise ValueError(f"blur.{source} must be an odd integer or 2-item integer range") from exc
        if sampled <= 0 or sampled % 2 == 0:
            raise ValueError(f"blur.{source} scalar must be a positive odd integer")
        configured = int(sampled)
    return sampled, {"source": source, "configured": configured}


def _sample_peripheral_center(
    width: int,
    height: int,
    min_ratio: float,
    rng: random.Random,
) -> tuple[int, int]:
    center_x = width / 2.0
    center_y = height / 2.0
    max_dist = math.sqrt(center_x**2 + center_y**2)
    min_dist = max(0.0, float(min_ratio)) * max_dist

    for _ in range(256):
        x = rng.randint(0, max(0, width - 1))
        y = rng.randint(0, max(0, height - 1))
        dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if dist >= min_dist:
            return x, y
    return max(0, width - 1), max(0, height - 1)


class ShadowAugmentation:
    """Localized subtractive shadow field."""

    name = "shadow"
    scope: AugmentationScope = "image_only"

    def apply(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        rng: random.Random,
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        radius, radius_config = _sample_float_or_range(
            parameters.get("radius"),
            field_name="shadow.radius",
            rng=rng,
            default=150.0,
            minimum=1.0,
        )
        sigma, sigma_config = _sample_float_or_range(
            parameters.get("sigma"),
            field_name="shadow.sigma",
            rng=rng,
            default=500.0,
            minimum=1.0,
        )
        intensity_lo, intensity_hi = _as_int_pair(
            parameters.get("intensity_range", [40, 50]),
            field_name="shadow.intensity_range",
        )
        count_lo, count_hi = _as_int_pair(
            parameters.get("count_range", [1, 3]),
            field_name="shadow.count_range",
        )

        image_f = image.astype(np.float32)
        height, width = image.shape[:2]
        x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        total_shadow = np.zeros((height, width), dtype=np.float32)
        count = rng.randint(count_lo, count_hi)
        entries: list[dict[str, Any]] = []

        for _ in range(count):
            sx = rng.randint(0, max(0, width - 1))
            sy = rng.randint(0, max(0, height - 1))
            intensity = rng.randint(intensity_lo, intensity_hi)

            dist_sq = (x_grid - float(sx)) ** 2 + (y_grid - float(sy)) ** 2
            gaussian = np.exp(-dist_sq / max(1.0, 2.0 * sigma**2))
            radial = np.exp(-dist_sq / max(1.0, 2.0 * radius**2))
            field = gaussian * radial
            peak = float(field.max()) if field.size else 0.0
            if peak > 0.0:
                field = field / peak
            total_shadow += intensity * field
            entries.append(
                {
                    "center": {"x": int(sx), "y": int(sy)},
                    "intensity": int(intensity),
                }
            )

        if image_f.ndim == 3:
            image_f = image_f - total_shadow[:, :, None]
        else:
            image_f = image_f - total_shadow
        out = np.clip(image_f, 0, 255).astype(np.uint8)
        return out, mask, {
            "shadow_count": int(count),
            "radius": float(radius),
            "sigma": float(sigma),
            "configured": {
                "radius": radius_config,
                "sigma": sigma_config,
            },
            "shadows": entries,
        }


class BlurAugmentation:
    """Localized peripheral Gaussian blur field."""

    name = "blur"
    scope: AugmentationScope = "image_only"

    def apply(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        rng: random.Random,
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sigma, sigma_config = _sample_float_or_range(
            parameters.get("sigma"),
            field_name="blur.sigma",
            rng=rng,
            default=120.0,
            minimum=1.0,
        )
        count_lo, count_hi = _as_int_pair(
            parameters.get("count_range", [1, 3]),
            field_name="blur.count_range",
        )
        min_center_distance_ratio, min_center_distance_config = _sample_float_or_range(
            parameters.get("min_center_distance_ratio"),
            field_name="blur.min_center_distance_ratio",
            rng=rng,
            default=0.4,
            minimum=0.0,
            maximum=1.0,
        )

        current = image.astype(np.float32)
        height, width = image.shape[:2]
        x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        count = rng.randint(count_lo, count_hi)
        entries: list[dict[str, Any]] = []

        for _ in range(count):
            bx, by = _sample_peripheral_center(width, height, min_center_distance_ratio, rng)
            ksize, kernel_metadata = _sample_odd_kernel(parameters, rng=rng)
            blurred = cv2.GaussianBlur(current, (ksize, ksize), 0)
            dist_sq = (x_grid - float(bx)) ** 2 + (y_grid - float(by)) ** 2
            mask_field = np.exp(-dist_sq / max(1.0, 2.0 * sigma**2))
            peak = float(mask_field.max()) if mask_field.size else 0.0
            if peak > 0.0:
                mask_field = mask_field / peak
            if current.ndim == 3:
                current = current * (1.0 - mask_field[:, :, None]) + blurred * mask_field[:, :, None]
            else:
                current = current * (1.0 - mask_field) + blurred * mask_field
            entries.append(
                {
                    "center": {"x": int(bx), "y": int(by)},
                    "kernel_size": int(ksize),
                    "kernel": kernel_metadata,
                }
            )

        out = np.clip(current, 0, 255).astype(np.uint8)
        return out, mask, {
            "blur_count": int(count),
            "sigma": float(sigma),
            "min_center_distance_ratio": float(min_center_distance_ratio),
            "configured": {
                "sigma": sigma_config,
                "min_center_distance_ratio": min_center_distance_config,
            },
            "blurs": entries,
        }


DEFAULT_AUGMENTATION_REGISTRY: dict[str, AugmentationOperation] = {
    "shadow": ShadowAugmentation(),
    "blur": BlurAugmentation(),
}


def parse_augmentation_config(raw: Any, *, default_seed: int) -> AugmentationConfig:
    """Normalize YAML/JSON augmentation config into dataclasses."""

    if raw in (None, False, "", {}):
        return AugmentationConfig(enabled=False, seed=int(default_seed))
    if raw is True:
        return AugmentationConfig(enabled=True, seed=int(default_seed))
    if not isinstance(raw, dict):
        raise ValueError("augmentation config must be a mapping")

    enabled = bool(raw.get("enabled", True))
    seed = int(raw.get("seed", default_seed))
    stage = str(raw.get("stage", "post_resize")).strip().lower()
    if stage not in {"pre_resize", "post_resize"}:
        raise ValueError("augmentation.stage must be 'pre_resize' or 'post_resize'")

    apply_splits_raw = raw.get("apply_splits", ["train"])
    if isinstance(apply_splits_raw, str):
        apply_splits = tuple(part.strip() for part in apply_splits_raw.split(",") if part.strip())
    else:
        apply_splits = tuple(str(part).strip() for part in list(apply_splits_raw or []) if str(part).strip())
    if not apply_splits:
        apply_splits = ("train",)

    variants_per_sample = int(raw.get("variants_per_sample", 1))
    if variants_per_sample < 0:
        raise ValueError("augmentation.variants_per_sample must be >= 0")

    ops_raw = raw.get("operations", [])
    if ops_raw is None:
        ops_raw = []
    if not isinstance(ops_raw, list):
        raise ValueError("augmentation.operations must be a list")
    operations: list[AugmentationOperationConfig] = []
    for idx, item in enumerate(ops_raw):
        if not isinstance(item, dict):
            raise ValueError(f"augmentation.operations[{idx}] must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"augmentation.operations[{idx}].name is required")
        probability = float(item.get("probability", 1.0))
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"augmentation.operations[{idx}].probability must be within [0,1]")
        parameters = item.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(f"augmentation.operations[{idx}].parameters must be a mapping")
        operations.append(
            AugmentationOperationConfig(
                name=name,
                enabled=bool(item.get("enabled", True)),
                probability=probability,
                parameters={str(k): v for k, v in parameters.items()},
            )
        )

    debug_raw = raw.get("debug", {})
    if debug_raw is None:
        debug_raw = {}
    if not isinstance(debug_raw, dict):
        raise ValueError("augmentation.debug must be a mapping")
    debug = AugmentationDebugConfig(
        enabled=bool(debug_raw.get("enabled", False)),
        max_samples=int(debug_raw.get("max_samples", 8)),
    )
    if debug.max_samples < 0:
        raise ValueError("augmentation.debug.max_samples must be >= 0")

    return AugmentationConfig(
        enabled=enabled,
        seed=seed,
        stage=stage,
        apply_splits=apply_splits,
        variants_per_sample=variants_per_sample,
        operations=tuple(operations),
        debug=debug,
    )


class AugmentationRunner:
    """Compose configured augmentations with deterministic per-sample seeding."""

    def __init__(
        self,
        config: AugmentationConfig,
        *,
        registry: dict[str, AugmentationOperation] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or DEFAULT_AUGMENTATION_REGISTRY
        unknown = sorted(
            {
                op.name
                for op in config.operations
                if op.enabled and op.name not in self.registry
            }
        )
        if unknown:
            raise ValueError(f"unknown augmentation operations: {', '.join(unknown)}")

    def enabled_for_split(self, split: str) -> bool:
        """Return whether augmentation should run for the given split."""

        return bool(self.config.enabled and split in set(self.config.apply_splits))

    def generate_variants(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        split: str,
        source_name: str,
        resolved_stage: str,
    ) -> list[AugmentedVariant]:
        """Generate augmented variants for one sample."""

        if not self.enabled_for_split(split):
            return []
        if self.config.variants_per_sample <= 0:
            return []
        active_ops = [op for op in self.config.operations if op.enabled]
        if not active_ops:
            return []

        variants: list[AugmentedVariant] = []
        for variant_index in range(1, int(self.config.variants_per_sample) + 1):
            sample_seed = _stable_seed(
                int(self.config.seed),
                split,
                source_name,
                str(variant_index),
            )
            rng = random.Random(sample_seed)
            current_image = image.copy()
            current_mask = mask.copy()
            applied: list[OperationApplicationRecord] = []
            for op_cfg in active_ops:
                if rng.random() > float(op_cfg.probability):
                    continue
                operation = self.registry[op_cfg.name]
                current_image, current_mask, metadata = operation.apply(
                    image=current_image,
                    mask=current_mask,
                    rng=rng,
                    parameters=dict(op_cfg.parameters),
                )
                applied.append(
                    OperationApplicationRecord(
                        name=operation.name,
                        scope=operation.scope,
                        probability=float(op_cfg.probability),
                        parameters=_json_safe(metadata),
                    )
                )
            if not applied:
                continue
            variants.append(
                AugmentedVariant(
                    image=current_image,
                    mask=current_mask,
                    metadata=AugmentedVariantRecord(
                        variant_index=variant_index,
                        sample_seed=sample_seed,
                        requested_stage=self.config.stage,
                        resolved_stage=resolved_stage,
                        split=split,
                        source_name=source_name,
                        applied_operations=tuple(applied),
                    ),
                )
            )
        return variants


class AugmentationDebugWriter:
    """Write before/after visual debug artifacts for augmented samples."""

    def write(
        self,
        *,
        debug_root: Path,
        split: str,
        stem: str,
        base_image: np.ndarray,
        augmented_image: np.ndarray,
        mask: np.ndarray,
        metadata: AugmentedVariantRecord,
        ext: str = ".png",
    ) -> None:
        sample_root = debug_root / split / stem
        sample_root.mkdir(parents=True, exist_ok=True)

        before_overlay = self._overlay(base_image, mask)
        after_overlay = self._overlay(augmented_image, mask)
        difference = cv2.absdiff(
            _ensure_bgr(base_image),
            _ensure_bgr(augmented_image),
        )
        panel = self._panel(
            before=_ensure_bgr(base_image),
            after=_ensure_bgr(augmented_image),
            before_overlay=before_overlay,
            after_overlay=after_overlay,
            difference=difference,
            mask=mask,
        )

        write_image(sample_root / f"{stem}_before{ext}", _ensure_bgr(base_image))
        write_image(sample_root / f"{stem}_after{ext}", _ensure_bgr(augmented_image))
        write_image(sample_root / f"{stem}_before_overlay{ext}", before_overlay)
        write_image(sample_root / f"{stem}_after_overlay{ext}", after_overlay)
        write_image(sample_root / f"{stem}_difference{ext}", difference)
        write_image(sample_root / f"{stem}_mask{ext}", _mask_display(mask))
        write_image(sample_root / f"{stem}_panel{ext}", panel)
        (sample_root / f"{stem}_metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = _ensure_bgr(image)
        mask_arr = mask
        if mask_arr.shape[:2] != base.shape[:2]:
            mask_arr = cv2.resize(
                _mask_display(mask_arr),
                (base.shape[1], base.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        highlight = np.zeros_like(base)
        highlight[:, :, 2] = (mask_arr > 0).astype(np.uint8) * 255
        return cv2.addWeighted(base, 0.75, highlight, 0.25, 0)

    @staticmethod
    def _panel(
        *,
        before: np.ndarray,
        after: np.ndarray,
        before_overlay: np.ndarray,
        after_overlay: np.ndarray,
        difference: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        mask_vis = cv2.cvtColor(_mask_display(mask), cv2.COLOR_GRAY2BGR)
        tiles = [before, after, before_overlay, after_overlay, difference, mask_vis]
        target_h = max(tile.shape[0] for tile in tiles)
        target_w = max(tile.shape[1] for tile in tiles)
        resized = [_resize_tile(tile, target_w, target_h) for tile in tiles]
        top = np.concatenate(resized[:3], axis=1)
        bottom = np.concatenate(resized[3:], axis=1)
        return np.concatenate([top, bottom], axis=0)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resize_tile(tile: np.ndarray, width: int, height: int) -> np.ndarray:
    if tile.shape[0] == height and tile.shape[1] == width:
        return tile
    return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _mask_display(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[2] >= 3:
        return cv2.cvtColor(_ensure_bgr(mask), cv2.COLOR_BGR2GRAY)
    arr = mask.astype(np.uint8)
    if arr.max(initial=0) <= 1:
        return arr * 255
    return arr


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
