"""DataLoader collate functions for segmentation tensors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def pad_to_max_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad image/mask items in a batch to the maximum H/W and stack.

    Parameters
    ----------
    batch:
        Sequence of ``(image, mask)`` where image is ``[C,H,W]`` and mask is ``[H,W]`` or ``[1,H,W]``.
    """

    if not batch:
        raise ValueError("empty batch")
    max_h = max(int(item[0].shape[-2]) for item in batch)
    max_w = max(int(item[0].shape[-1]) for item in batch)

    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for image, mask in batch:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        pad_h = max_h - int(image.shape[-2])
        pad_w = max_w - int(image.shape[-1])
        image_pad = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), value=0).to(mask.dtype)
        images.append(image_pad)
        masks.append(mask_pad)

    return torch.stack(images, dim=0), torch.stack(masks, dim=0)


def resolve_collate_fn(name: str) -> Any:
    """Resolve collate function by policy name."""

    mode = str(name).strip().lower() or "default"
    if mode == "default":
        return None
    if mode == "pad_to_max":
        return pad_to_max_collate
    raise ValueError(f"unsupported dataloader_collate={name!r}; expected default|pad_to_max")
