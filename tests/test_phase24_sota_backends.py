"""Phase 24 tests for newly integrated SOTA segmentation backends."""

from __future__ import annotations

import pytest

from src.microseg.training.unet_binary import _build_binary_model


@pytest.mark.parametrize(
    "architecture",
    [
        "smp_deeplabv3plus_resnet101",
        "smp_unetplusplus_resnet101",
        "smp_pspnet_resnet101",
        "smp_fpn_resnet101",
    ],
)
def test_phase24_new_smp_backends_forward_shape(architecture: str) -> None:
    pytest.importorskip("segmentation_models_pytorch")
    import torch

    model = _build_binary_model(
        architecture=architecture,
        base_channels=16,
        transformer_depth=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        segformer_patch_size=4,
        pretrained_bundle=None,
        pretrained_strict=False,
        pretrained_ignore_mismatched_sizes=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert tuple(int(v) for v in out.shape) == (1, 1, 64, 64)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_phase24_hf_upernet_swin_large_forward_shape() -> None:
    pytest.importorskip("transformers")
    import torch

    model = _build_binary_model(
        architecture="hf_upernet_swin_large",
        base_channels=16,
        transformer_depth=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        segformer_patch_size=4,
        pretrained_bundle=None,
        pretrained_strict=False,
        pretrained_ignore_mismatched_sizes=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 128, 128))
    assert tuple(int(v) for v in out.shape) == (1, 1, 128, 128)
