"""Input-size policy and collate behavior tests for mixed-resolution segmentation data."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.microseg.data.collate import pad_to_max_collate
from src.microseg.data.transforms import InputPolicyConfig, apply_input_policy


class _RawMixedSizeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            (torch.zeros((3, 1920, 2560), dtype=torch.float32), torch.zeros((1, 1920, 2560), dtype=torch.float32)),
            (torch.zeros((3, 320, 320), dtype=torch.float32), torch.zeros((1, 320, 320), dtype=torch.float32)),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class _PolicyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cfg: InputPolicyConfig) -> None:
        self.cfg = cfg
        mask_a = torch.zeros((1, 1920, 2560), dtype=torch.long)
        mask_a[:, :, 200:800] = 1
        mask_b = torch.zeros((1, 320, 320), dtype=torch.long)
        mask_b[:, 40:120, 40:120] = 1
        self.samples = [
            (torch.rand((3, 1920, 2560), dtype=torch.float32), mask_a),
            (torch.rand((3, 320, 320), dtype=torch.float32), mask_b),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.samples[idx]
        out = apply_input_policy(image, mask, self.cfg, rng=torch.Generator().manual_seed(123 + idx), is_train=True)
        return out


def test_mixed_size_default_collate_reproduces_stack_failure() -> None:
    loader = DataLoader(_RawMixedSizeDataset(), batch_size=2, shuffle=False)
    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        _ = next(iter(loader))


def test_resize_policy_allows_stacking_to_fixed_hw() -> None:
    cfg = InputPolicyConfig(input_hw=(512, 512), input_policy="resize", require_divisible_by=1)
    loader = DataLoader(_PolicyDataset(cfg), batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    assert tuple(images.shape) == (2, 3, 512, 512)
    assert tuple(masks.shape) == (2, 1, 512, 512)


def test_letterbox_preserves_mask_labels_and_applies_padding() -> None:
    image = torch.ones((3, 200, 400), dtype=torch.float32)
    mask = torch.zeros((1, 200, 400), dtype=torch.long)
    mask[:, 60:140, 120:260] = 1
    cfg = InputPolicyConfig(
        input_hw=(512, 512),
        input_policy="letterbox",
        keep_aspect=True,
        pad_value_image=0.25,
        pad_value_mask=0,
        require_divisible_by=1,
    )
    out_img, out_mask = apply_input_policy(image, mask, cfg, is_train=False)
    assert tuple(out_img.shape) == (3, 512, 512)
    assert tuple(out_mask.shape) == (1, 512, 512)
    assert torch.all((out_mask == 0) | (out_mask == 1))
    assert torch.isclose(out_img[:, 0, 0].mean(), torch.tensor(0.25), atol=1e-6)


def test_require_divisible_by_pads_to_next_multiple() -> None:
    image = torch.zeros((3, 513, 513), dtype=torch.float32)
    mask = torch.ones((1, 513, 513), dtype=torch.long)
    cfg = InputPolicyConfig(input_hw=(513, 513), input_policy="resize", require_divisible_by=32, pad_value_mask=0)
    out_img, out_mask = apply_input_policy(image, mask, cfg, is_train=False)
    assert tuple(out_img.shape) == (3, 544, 544)
    assert tuple(out_mask.shape) == (1, 544, 544)
    assert int(out_mask[:, -1, -1].item()) == 0


def test_pad_to_max_collate_fallback() -> None:
    loader = DataLoader(_RawMixedSizeDataset(), batch_size=2, collate_fn=pad_to_max_collate, shuffle=False)
    images, masks = next(iter(loader))
    assert tuple(images.shape) == (2, 3, 1920, 2560)
    assert tuple(masks.shape) == (2, 1, 1920, 2560)
