"""Mask binarization and optional morphology cleanup."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from src.microseg.data_preparation.config import DatasetPrepConfig


class MaskBinarizer:
    """Convert mask arrays to uint8 binary masks with configurable modes."""

    def __init__(self, cfg: DatasetPrepConfig) -> None:
        self.cfg = cfg

    def apply(self, raw_mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if self.cfg.rgb_mask_mode:
            return self._apply_rgb_threshold(raw_mask)
        return self._apply_grayscale(raw_mask)

    def _apply_grayscale(
        self,
        raw_mask: np.ndarray,
        *,
        extra_warnings: list[str] | None = None,
        mode_override: str | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        gray = self._to_gray(raw_mask)
        unique_raw = sorted(np.unique(gray).astype(int).tolist())
        expected = sorted({int(v) for v in self.cfg.expected_raw_binary_values})
        if self._is_binary_like_values(unique_raw):
            expected = sorted({*expected, 1})
        non_binary_map = ~np.isin(gray, expected)
        non_binary_count = int(np.count_nonzero(non_binary_map))
        non_binary_ratio = float(non_binary_count / gray.size) if gray.size else 0.0

        warnings: list[str] = list(extra_warnings or [])
        stats: dict[str, object] = {
            "mode": mode_override or f"grayscale_{self.cfg.binarization_mode}",
            "unique_raw_values": unique_raw,
            "expected_raw_binary_values": expected,
            "non_binary_pixel_count": non_binary_count,
            "non_binary_pixel_ratio": non_binary_ratio,
            "warnings": warnings,
        }
        if non_binary_count > 0:
            non_binary_values = sorted(np.unique(gray[non_binary_map]).astype(int).tolist())
            stats["non_binary_values"] = non_binary_values
            stats["non_binary_value_count"] = len(non_binary_values)
            stats["warnings"].append(
                "raw mask contains non-binary values outside expected "
                f"{expected}: values={non_binary_values[:20]}{'...' if len(non_binary_values) > 20 else ''} "
                f"pixels={non_binary_count}/{gray.size} ({non_binary_ratio:.4%})"
            )

        mode = self.cfg.binarization_mode
        if self._should_use_auto_otsu(gray, unique_raw):
            threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            stats["mode"] = f"{mode_override}_auto_otsu" if mode_override else "grayscale_auto_otsu"
            stats["auto_otsu_applied"] = True
            stats["otsu_threshold"] = int(threshold)
            stats["warnings"].append(
                "near-binary grayscale mask with noisy values detected; auto-otsu thresholding applied"
            )
            binary = gray >= int(threshold)
        elif mode == "nonzero":
            binary = gray > 0
        elif mode == "threshold":
            binary = gray > self.cfg.threshold if self.cfg.threshold_strict else gray >= self.cfg.threshold
        elif mode == "value_equals":
            values = set(int(v) for v in self.cfg.foreground_values)
            binary = np.isin(gray, list(values))
        elif mode == "otsu":
            threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            stats["otsu_threshold"] = int(threshold)
            binary = gray >= int(threshold)
        elif mode == "percentile":
            threshold = float(np.percentile(gray, self.cfg.percentile))
            stats["percentile_threshold"] = threshold
            binary = gray >= threshold
        else:
            raise ValueError(f"unsupported binarization mode: {mode}")

        if self.cfg.invert_mask:
            binary = ~binary

        cleaned = self._morphology(binary.astype(np.uint8))
        stats["unique_binary_values"] = sorted(np.unique(cleaned).astype(int).tolist())
        stats["fg_pixel_count"] = int(cleaned.sum())
        stats["fg_ratio"] = float(cleaned.mean())
        return cleaned.astype(np.uint8), stats

    def _apply_rgb_threshold(self, raw_mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        warnings: list[str] = []
        if raw_mask.ndim == 2:
            unique_raw = sorted(np.unique(raw_mask).astype(int).tolist())
            if self._is_binary_like_values(unique_raw):
                binary = raw_mask > 0
                if self.cfg.invert_mask:
                    binary = ~binary
                cleaned = self._morphology(binary.astype(np.uint8))
                stats: dict[str, object] = {
                    "mode": "grayscale_binary_passthrough",
                    "warnings": [
                        "mask is grayscale while rgb_mask_mode=true; detected binary-like values and mapped non-zero pixels to foreground"
                    ],
                    "raw_mask_channels": 1,
                    "unique_raw_values": unique_raw,
                    "unique_binary_values": sorted(np.unique(cleaned).astype(int).tolist()),
                    "fg_pixel_count": int(cleaned.sum()),
                    "fg_ratio": float(cleaned.mean()),
                }
                return cleaned.astype(np.uint8), stats
            warnings.append("mask is grayscale while rgb_mask_mode=true; applying grayscale binarization mode")
            binary, stats = self._apply_grayscale(raw_mask, extra_warnings=warnings, mode_override="grayscale_fallback")
            stats["raw_mask_channels"] = 1
            return binary, stats
        elif raw_mask.ndim == 3 and raw_mask.shape[2] >= 3:
            blue = raw_mask[:, :, 0]
            green = raw_mask[:, :, 1]
            red = raw_mask[:, :, 2]
        else:
            raise ValueError(f"unsupported mask shape for RGB binarization: {raw_mask.shape}")

        red_unique_values = sorted(np.unique(red).astype(int).tolist())[:256]
        green_unique_values = sorted(np.unique(green).astype(int).tolist())[:256]
        blue_unique_values = sorted(np.unique(blue).astype(int).tolist())[:256]

        red_i = red.astype(np.int16)
        green_i = green.astype(np.int16)
        blue_i = blue.astype(np.int16)

        red_sel = red_i >= int(self.cfg.mask_r_min)
        if self.cfg.enforce_gb_thresholds:
            rgb_threshold_binary = red_sel & (green_i <= int(self.cfg.mask_g_max)) & (blue_i <= int(self.cfg.mask_b_max))
        else:
            rgb_threshold_binary = red_sel

        dominance_binary = (
            (red_i >= int(self.cfg.mask_red_min_fallback))
            & ((red_i - np.maximum(green_i, blue_i)) >= int(self.cfg.mask_red_dominance_margin))
            & (red_i.astype(np.float32) >= (green_i.astype(np.float32) + 1.0) * float(self.cfg.mask_red_dominance_ratio))
            & (red_i.astype(np.float32) >= (blue_i.astype(np.float32) + 1.0) * float(self.cfg.mask_red_dominance_ratio))
        )
        binary = rgb_threshold_binary
        if self.cfg.allow_red_dominance_fallback:
            binary = binary | dominance_binary
        if self.cfg.invert_mask:
            binary = ~binary

        cleaned = self._morphology(binary.astype(np.uint8))
        stats: dict[str, object] = {
            "mode": "rgb_threshold",
            "thresholds": {
                "mask_r_min": int(self.cfg.mask_r_min),
                "mask_g_max": int(self.cfg.mask_g_max),
                "mask_b_max": int(self.cfg.mask_b_max),
                "enforce_gb_thresholds": bool(self.cfg.enforce_gb_thresholds),
                "allow_red_dominance_fallback": bool(self.cfg.allow_red_dominance_fallback),
                "mask_red_min_fallback": int(self.cfg.mask_red_min_fallback),
                "mask_red_dominance_margin": int(self.cfg.mask_red_dominance_margin),
                "mask_red_dominance_ratio": float(self.cfg.mask_red_dominance_ratio),
            },
            "warnings": warnings,
            "raw_mask_channels": 1 if raw_mask.ndim == 2 else int(raw_mask.shape[2]),
            "red_unique_values": red_unique_values,
            "green_unique_values": green_unique_values,
            "blue_unique_values": blue_unique_values,
            "rgb_threshold_fg_pixel_count": int(np.count_nonzero(rgb_threshold_binary)),
            "red_dominance_fg_pixel_count": int(np.count_nonzero(dominance_binary)),
            "unique_binary_values": sorted(np.unique(cleaned).astype(int).tolist()),
            "fg_pixel_count": int(cleaned.sum()),
            "fg_ratio": float(cleaned.mean()),
        }
        return cleaned.astype(np.uint8), stats

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        result = binary.copy()
        if self.cfg.morphology.open_kernel > 0:
            kernel = np.ones((self.cfg.morphology.open_kernel, self.cfg.morphology.open_kernel), dtype=np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        if self.cfg.morphology.close_kernel > 0:
            kernel = np.ones((self.cfg.morphology.close_kernel, self.cfg.morphology.close_kernel), dtype=np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        if self.cfg.morphology.remove_small_components > 0:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
            filtered = np.zeros_like(result)
            for component in range(1, count):
                area = stats[component, cv2.CC_STAT_AREA]
                if int(area) >= self.cfg.morphology.remove_small_components:
                    filtered[labels == component] = 1
            result = filtered
        if self.cfg.morphology.fill_holes:
            result = ndimage.binary_fill_holes(result.astype(bool)).astype(np.uint8)
        return (result > 0).astype(np.uint8)

    @staticmethod
    def _to_gray(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask

    @staticmethod
    def _is_binary_like_values(values: list[int]) -> bool:
        return set(int(v) for v in values).issubset({0, 1, 255})

    def _should_use_auto_otsu(self, gray: np.ndarray, unique_raw: list[int]) -> bool:
        if not bool(self.cfg.auto_otsu_for_noisy_grayscale):
            return False
        if self._is_binary_like_values(unique_raw):
            return False
        if gray.size == 0:
            return False
        low_max = int(self.cfg.noisy_grayscale_low_max)
        high_min = int(self.cfg.noisy_grayscale_high_min)
        if high_min <= low_max:
            return False
        extreme = (gray <= low_max) | (gray >= high_min)
        extreme_ratio = float(np.count_nonzero(extreme) / gray.size)
        if extreme_ratio < float(self.cfg.noisy_grayscale_min_extreme_ratio):
            return False
        # Require that values exist in both bands to avoid collapsing all-zero/all-high masks.
        has_low = bool(np.any(gray <= low_max))
        has_high = bool(np.any(gray >= high_min))
        return has_low and has_high
