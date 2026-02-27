"""Image and mask resizing with multiple aspect-ratio policies."""

from __future__ import annotations

import random

import cv2
import numpy as np

from src.microseg.data_preparation.config import DatasetPrepConfig


_INTERP_MAP = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}
_BORDER_MAP = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
}


class Resizer:
    """Resize transform helper for images and binary masks."""

    def __init__(self, cfg: DatasetPrepConfig) -> None:
        self.cfg = cfg

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        split: str | None = None,
        sample_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        warnings: list[str] = []
        h, w = image.shape[:2]
        target_h, target_w = self.cfg.target_size
        policy = self.cfg.resize_policy

        if policy == "stretch":
            warnings.append("stretch policy may distort aspect ratio")
            out_img = cv2.resize(image, (target_w, target_h), interpolation=_INTERP_MAP[self.cfg.image_interpolation])
            out_msk = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return out_img, (out_msk > 0).astype(np.uint8), warnings

        if policy == "keep_aspect_no_pad":
            scale = min(target_w / w, target_h / h)
            if abs(scale - 1.0) > 1e-6:
                warnings.append("keep_aspect_no_pad does not guarantee exact target shape")
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            out_img = cv2.resize(image, (new_w, new_h), interpolation=_INTERP_MAP[self.cfg.image_interpolation])
            out_msk = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return out_img, (out_msk > 0).astype(np.uint8), warnings

        if policy == "center_crop":
            scale = max(target_w / w, target_h / h)
            scaled_w = max(1, int(round(w * scale)))
            scaled_h = max(1, int(round(h * scale)))
            scaled_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=_INTERP_MAP[self.cfg.image_interpolation])
            scaled_msk = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            start_x = max(0, (scaled_w - target_w) // 2)
            start_y = max(0, (scaled_h - target_h) // 2)
            return (
                scaled_img[start_y:start_y + target_h, start_x:start_x + target_w],
                (scaled_msk[start_y:start_y + target_h, start_x:start_x + target_w] > 0).astype(np.uint8),
                warnings,
            )

        if policy == "short_side_to_target_crop":
            scale = max(target_w / w, target_h / h)
            scaled_w = max(1, int(round(w * scale)))
            scaled_h = max(1, int(round(h * scale)))
            scaled_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=_INTERP_MAP[self.cfg.image_interpolation])
            scaled_msk = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            crop_mode = self._resolve_crop_mode(split)
            start_x, start_y = self._resolve_crop_origin(
                scaled_w=scaled_w,
                scaled_h=scaled_h,
                target_w=target_w,
                target_h=target_h,
                crop_mode=crop_mode,
                sample_seed=sample_seed,
            )
            return (
                scaled_img[start_y:start_y + target_h, start_x:start_x + target_w],
                (scaled_msk[start_y:start_y + target_h, start_x:start_x + target_w] > 0).astype(np.uint8),
                warnings,
            )

        if policy == "letterbox_pad":
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=_INTERP_MAP[self.cfg.image_interpolation])
            resized_msk = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            out_img = cv2.copyMakeBorder(
                resized_img,
                top,
                bottom,
                left,
                right,
                borderType=_BORDER_MAP[self.cfg.image_pad_mode],
                value=self.cfg.image_pad_value,
            )
            out_msk = cv2.copyMakeBorder(
                (resized_msk > 0).astype(np.uint8),
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            return out_img, out_msk.astype(np.uint8), warnings

        raise ValueError(f"unsupported resize policy: {policy}")

    def _resolve_crop_mode(self, split: str | None) -> str:
        if split == "train":
            return self.cfg.crop_mode_train
        if split in {"val", "test"}:
            return self.cfg.crop_mode_eval
        return self.cfg.crop_mode_eval

    @staticmethod
    def _resolve_crop_origin(
        *,
        scaled_w: int,
        scaled_h: int,
        target_w: int,
        target_h: int,
        crop_mode: str,
        sample_seed: int | None,
    ) -> tuple[int, int]:
        max_x = max(0, scaled_w - target_w)
        max_y = max(0, scaled_h - target_h)
        if crop_mode == "center" or (max_x == 0 and max_y == 0):
            return max_x // 2, max_y // 2
        if crop_mode == "random":
            rng = random.Random(sample_seed)
            return rng.randint(0, max_x), rng.randint(0, max_y)
        raise ValueError(f"unsupported crop mode: {crop_mode}")
