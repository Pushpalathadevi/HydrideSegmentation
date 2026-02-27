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
        gray = self._to_gray(raw_mask)
        unique_raw = sorted(np.unique(gray).astype(int).tolist())
        expected = sorted({int(v) for v in self.cfg.expected_raw_binary_values})
        non_binary_map = ~np.isin(gray, expected)
        non_binary_count = int(np.count_nonzero(non_binary_map))
        non_binary_ratio = float(non_binary_count / gray.size) if gray.size else 0.0

        stats: dict[str, object] = {
            "unique_raw_values": unique_raw,
            "expected_raw_binary_values": expected,
            "non_binary_pixel_count": non_binary_count,
            "non_binary_pixel_ratio": non_binary_ratio,
            "warnings": [],
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
        if mode == "nonzero":
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
            warnings.append("mask is grayscale but rgb_mask_mode=true; falling back to grayscale threshold on red/min channel")
            red = raw_mask
            green = np.zeros_like(raw_mask)
            blue = np.zeros_like(raw_mask)
        elif raw_mask.ndim == 3 and raw_mask.shape[2] >= 3:
            blue = raw_mask[:, :, 0]
            green = raw_mask[:, :, 1]
            red = raw_mask[:, :, 2]
        else:
            raise ValueError(f"unsupported mask shape for RGB binarization: {raw_mask.shape}")

        red_sel = red >= int(self.cfg.mask_r_min)
        if self.cfg.enforce_gb_thresholds:
            binary = red_sel & (green <= int(self.cfg.mask_g_max)) & (blue <= int(self.cfg.mask_b_max))
        else:
            binary = red_sel
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
            },
            "warnings": warnings,
            "raw_mask_channels": 1 if raw_mask.ndim == 2 else int(raw_mask.shape[2]),
            "red_unique_values": sorted(np.unique(red).astype(int).tolist())[:256],
            "green_unique_values": sorted(np.unique(green).astype(int).tolist())[:256],
            "blue_unique_values": sorted(np.unique(blue).astype(int).tolist())[:256],
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
            return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask
