"""Input size policies for segmentation image/mask tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

InputPolicyName = Literal["resize", "letterbox", "random_crop", "center_crop"]


@dataclass(frozen=True)
class InputPolicyConfig:
    """Configuration for deterministic image/mask shape handling.

    Parameters
    ----------
    input_hw:
        Target ``(height, width)`` after policy application.
    input_policy:
        Transformation strategy applied to both image and mask.
    keep_aspect:
        Keep aspect ratio when using ``letterbox``.
    pad_value_image:
        Fill value used for image padding.
    pad_value_mask:
        Fill value used for mask padding.
    image_interpolation:
        Interpolation for image resizing.
    require_divisible_by:
        If > 1, pad output shape to next multiple of this stride.
    """

    input_hw: tuple[int, int] = (512, 512)
    input_policy: InputPolicyName = "random_crop"
    keep_aspect: bool = True
    pad_value_image: float = 0.0
    pad_value_mask: int = 0
    image_interpolation: Literal["bilinear", "bicubic", "nearest"] = "bilinear"
    require_divisible_by: int = 32


def _resize_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_hw: tuple[int, int],
    image_interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = "bilinear" if image_interpolation not in {"bilinear", "bicubic", "nearest"} else image_interpolation
    image_4d = image.unsqueeze(0)
    mask_4d = mask.unsqueeze(0).to(torch.float32)
    resized_img = F.interpolate(
        image_4d,
        size=out_hw,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    ).squeeze(0)
    resized_mask = F.interpolate(mask_4d, size=out_hw, mode="nearest").squeeze(0)
    return resized_img, resized_mask.to(mask.dtype)


def _pad_to_size(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_hw: tuple[int, int],
    pad_value_image: float,
    pad_value_mask: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[-2:]
    out_h, out_w = out_hw
    if h >= out_h and w >= out_w:
        return image, mask
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image = F.pad(image, (left, right, top, bottom), value=float(pad_value_image))
    mask = F.pad(mask, (left, right, top, bottom), value=float(pad_value_mask)).to(mask.dtype)
    return image, mask


def _center_crop_pair(image: torch.Tensor, mask: torch.Tensor, *, out_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[-2:]
    out_h, out_w = out_hw
    if h < out_h or w < out_w:
        image, mask = _pad_to_size(image, mask, out_hw=out_hw, pad_value_image=0.0, pad_value_mask=0)
        h, w = image.shape[-2:]
    top = max(0, (h - out_h) // 2)
    left = max(0, (w - out_w) // 2)
    return image[:, top : top + out_h, left : left + out_w], mask[:, top : top + out_h, left : left + out_w]


def _random_crop_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_hw: tuple[int, int],
    generator: torch.Generator | None,
    pad_value_image: float,
    pad_value_mask: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_h, out_w = out_hw
    image, mask = _pad_to_size(
        image,
        mask,
        out_hw=out_hw,
        pad_value_image=pad_value_image,
        pad_value_mask=pad_value_mask,
    )
    h, w = image.shape[-2:]
    max_top = max(0, h - out_h)
    max_left = max(0, w - out_w)
    if max_top == 0:
        top = 0
    else:
        top = int(torch.randint(0, max_top + 1, (1,), generator=generator).item())
    if max_left == 0:
        left = 0
    else:
        left = int(torch.randint(0, max_left + 1, (1,), generator=generator).item())
    return image[:, top : top + out_h, left : left + out_w], mask[:, top : top + out_h, left : left + out_w]


def _letterbox_pair(image: torch.Tensor, mask: torch.Tensor, *, cfg: InputPolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    out_h, out_w = cfg.input_hw
    h, w = image.shape[-2:]
    if not cfg.keep_aspect:
        return _resize_pair(image, mask, out_hw=cfg.input_hw, image_interpolation=cfg.image_interpolation)
    scale = min(out_h / max(1, h), out_w / max(1, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    image, mask = _resize_pair(image, mask, out_hw=(new_h, new_w), image_interpolation=cfg.image_interpolation)
    image, mask = _pad_to_size(
        image,
        mask,
        out_hw=cfg.input_hw,
        pad_value_image=cfg.pad_value_image,
        pad_value_mask=cfg.pad_value_mask,
    )
    return _center_crop_pair(image, mask, out_hw=cfg.input_hw)


def _pad_to_multiple(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    multiple: int,
    pad_value_image: float,
    pad_value_mask: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if multiple <= 1:
        return image, mask
    h, w = image.shape[-2:]
    out_h = ((h + multiple - 1) // multiple) * multiple
    out_w = ((w + multiple - 1) // multiple) * multiple
    return _pad_to_size(
        image,
        mask,
        out_hw=(out_h, out_w),
        pad_value_image=pad_value_image,
        pad_value_mask=pad_value_mask,
    )


def apply_input_policy(
    image: torch.Tensor,
    mask: torch.Tensor,
    cfg: InputPolicyConfig,
    *,
    rng: torch.Generator | None = None,
    is_train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured input transform policy to image and mask tensors."""
    if image.ndim != 3:
        raise ValueError(f"expected image shape [C,H,W], got {tuple(image.shape)}")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"expected mask shape [H,W] or [1,H,W], got {tuple(mask.shape)}")

    policy = str(cfg.input_policy).strip().lower()
    if policy == "resize":
        image, mask = _resize_pair(image, mask, out_hw=cfg.input_hw, image_interpolation=cfg.image_interpolation)
    elif policy == "letterbox":
        image, mask = _letterbox_pair(image, mask, cfg=cfg)
    elif policy == "center_crop":
        image, mask = _center_crop_pair(image, mask, out_hw=cfg.input_hw)
    elif policy == "random_crop":
        if is_train:
            image, mask = _random_crop_pair(
                image,
                mask,
                out_hw=cfg.input_hw,
                generator=rng,
                pad_value_image=cfg.pad_value_image,
                pad_value_mask=cfg.pad_value_mask,
            )
        else:
            image, mask = _center_crop_pair(image, mask, out_hw=cfg.input_hw)
    else:
        raise ValueError(f"unsupported input_policy={cfg.input_policy!r}")

    image, mask = _pad_to_multiple(
        image,
        mask,
        multiple=max(1, int(cfg.require_divisible_by)),
        pad_value_image=cfg.pad_value_image,
        pad_value_mask=cfg.pad_value_mask,
    )
    return image, mask.to(mask.dtype)
