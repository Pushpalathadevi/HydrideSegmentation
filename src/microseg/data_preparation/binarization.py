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
        gray = self._to_gray(raw_mask)
        stats: dict[str, object] = {
            "unique_raw_values": sorted(np.unique(gray).astype(int).tolist()),
            "warnings": [],
        }

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
