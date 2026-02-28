"""Download and stage local pretrained-weight bundles for air-gapped transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "pre_trained_weights"
REGISTRY_SCHEMA = "microseg.pretrained_weights_registry.v1"
BUNDLE_SCHEMA = "microseg.pretrained_bundle.v1"

SEGFORMER_CITATION = (
    "Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. "
    "\"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.\" "
    "NeurIPS 2021."
)
UPERNET_CITATION = (
    "Xiao, T., Liu, Y., Zhou, B., Jiang, Y., and Sun, J. "
    "\"Unified Perceptual Parsing for Scene Understanding.\" ECCV 2018."
)
SWIN_CITATION = (
    "Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. "
    "\"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.\" ICCV 2021."
)
UNET_CITATION = (
    "Ronneberger, O., Fischer, P., and Brox, T. "
    "\"U-Net: Convolutional Networks for Biomedical Image Segmentation.\" MICCAI 2015."
)
UNETPP_CITATION = (
    "Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., and Liang, J. "
    "\"UNet++: A Nested U-Net Architecture for Medical Image Segmentation.\" 2018."
)
DEEPLABV3PLUS_CITATION = (
    "Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., and Adam, H. "
    "\"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.\" ECCV 2018."
)
PSPNET_CITATION = (
    "Zhao, H., Shi, J., Qi, X., Wang, X., and Jia, J. "
    "\"Pyramid Scene Parsing Network.\" CVPR 2017."
)
FPN_CITATION = (
    "Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. "
    "\"Feature Pyramid Networks for Object Detection.\" CVPR 2017."
)
RESNET_CITATION = (
    "He, K., Zhang, X., Ren, S., and Sun, J. "
    "\"Deep Residual Learning for Image Recognition.\" CVPR 2016."
)
VIT_CITATION = (
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "
    "\"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.\" ICLR 2021."
)

HF_SEGFORMER_TARGETS: dict[str, dict[str, str]] = {
    "hf_segformer_b0": {
        "model_id": "hf_segformer_b0_ade20k",
        "architecture": "hf_segformer_b0",
        "repo_id": "nvidia/segformer-b0-finetuned-ade-512-512",
    },
    "hf_segformer_b2": {
        "model_id": "hf_segformer_b2_ade20k",
        "architecture": "hf_segformer_b2",
        "repo_id": "nvidia/segformer-b2-finetuned-ade-512-512",
    },
    "hf_segformer_b5": {
        "model_id": "hf_segformer_b5_ade20k",
        "architecture": "hf_segformer_b5",
        "repo_id": "nvidia/segformer-b5-finetuned-ade-640-640",
    },
}

HF_UPERNET_TARGETS: dict[str, dict[str, str]] = {
    "hf_upernet_swin_large": {
        "model_id": "hf_upernet_swin_large_ade20k",
        "architecture": "hf_upernet_swin_large",
        "repo_id": "openmmlab/upernet-swin-large",
    }
}

SMP_SEGMENTATION_TARGETS: dict[str, dict[str, str]] = {
    "smp_unet_resnet18": {
        "model_id": "smp_unet_resnet18_imagenet",
        "architecture": "smp_unet_resnet18",
        "decoder_name": "unet",
        "encoder_name": "resnet18",
    },
    "smp_deeplabv3plus_resnet101": {
        "model_id": "smp_deeplabv3plus_resnet101_imagenet",
        "architecture": "smp_deeplabv3plus_resnet101",
        "decoder_name": "deeplabv3plus",
        "encoder_name": "resnet101",
    },
    "smp_unetplusplus_resnet101": {
        "model_id": "smp_unetplusplus_resnet101_imagenet",
        "architecture": "smp_unetplusplus_resnet101",
        "decoder_name": "unetplusplus",
        "encoder_name": "resnet101",
    },
    "smp_pspnet_resnet101": {
        "model_id": "smp_pspnet_resnet101_imagenet",
        "architecture": "smp_pspnet_resnet101",
        "decoder_name": "pspnet",
        "encoder_name": "resnet101",
    },
    "smp_fpn_resnet101": {
        "model_id": "smp_fpn_resnet101_imagenet",
        "architecture": "smp_fpn_resnet101",
        "decoder_name": "fpn",
        "encoder_name": "resnet101",
    },
}

INTERNAL_TRANSFORMER_BOOTSTRAP_TARGETS: dict[str, dict[str, str]] = {
    "transunet_tiny": {
        "model_id": "transunet_tiny_vit_tiny_patch16_imagenet",
        "architecture": "transunet_tiny",
        "source_model": "vit_tiny_patch16_224",
    },
    "segformer_mini": {
        "model_id": "segformer_mini_vit_tiny_patch16_imagenet",
        "architecture": "segformer_mini",
        "source_model": "vit_tiny_patch16_224",
    },
}

UNET_BINARY_BOOTSTRAP_TARGETS: dict[str, dict[str, str]] = {
    "unet_binary": {
        "model_id": "unet_binary_resnet18_imagenet_partial",
        "architecture": "unet_binary",
        "source_model": "resnet18",
    }
}

ALL_TARGETS = tuple(
    sorted(
        (
            *UNET_BINARY_BOOTSTRAP_TARGETS.keys(),
            *HF_SEGFORMER_TARGETS.keys(),
            *HF_UPERNET_TARGETS.keys(),
            *SMP_SEGMENTATION_TARGETS.keys(),
            *INTERNAL_TRANSFORMER_BOOTSTRAP_TARGETS.keys(),
        )
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _collect_files(bundle_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        files.append(
            {
                "path": rel,
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return files


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_hf_segformer_bundle(spec: dict[str, str], bundle_root: Path, *, force: bool) -> dict[str, Any]:
    from huggingface_hub import model_info, snapshot_download

    model_id = str(spec["model_id"])
    architecture = str(spec["architecture"])
    repo_id = str(spec["repo_id"])
    bundle_dir = bundle_root / model_id
    model_dir = bundle_dir / "hf_model"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    info = model_info(repo_id)
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=info.sha,
            ignore_patterns=["*.h5", "*.msgpack"],
        )
    )
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(snapshot_path, model_dir)

    metadata = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": _utc_now(),
        "model_id": model_id,
        "architecture": architecture,
        "framework": "transformers",
        "weights_format": "hf_model_dir",
        "source": f"huggingface:{repo_id}",
        "source_url": f"https://huggingface.co/{repo_id}",
        "source_revision": info.sha,
        "license": "See upstream model card license metadata",
        "citation_key": "xie2021segformer",
        "citation": SEGFORMER_CITATION,
        "citation_url": "https://arxiv.org/abs/2105.15203",
        "notes": (
            "Local snapshot for offline transfer-learning init. "
            "Training pipeline uses local_files_only=True and num_labels=2."
        ),
    }
    _write_json(metadata_path, metadata)
    files = _collect_files(bundle_dir)
    _write_json(bundle_dir / "SHA256SUMS.json", {"schema_version": "microseg.sha256.v1", "files": files})

    return {
        "model_id": model_id,
        "architecture": architecture,
        "framework": "transformers",
        "source": f"huggingface:{repo_id}",
        "source_url": f"https://huggingface.co/{repo_id}",
        "source_revision": str(info.sha),
        "bundle_dir": model_id,
        "weights_path": "hf_model",
        "weights_format": "hf_model_dir",
        "metadata_path": "metadata.json",
        "license": "See upstream model card license metadata",
        "citation_key": "xie2021segformer",
        "citation": SEGFORMER_CITATION,
        "citation_url": "https://arxiv.org/abs/2105.15203",
        "notes": f"{architecture} pretrained segmentation snapshot for offline fine-tuning.",
        "files": files,
    }


def _download_hf_upernet_bundle(spec: dict[str, str], bundle_root: Path, *, force: bool) -> dict[str, Any]:
    from huggingface_hub import model_info, snapshot_download

    model_id = str(spec["model_id"])
    architecture = str(spec["architecture"])
    repo_id = str(spec["repo_id"])
    bundle_dir = bundle_root / model_id
    model_dir = bundle_dir / "hf_model"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    info = model_info(repo_id)
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=info.sha,
            ignore_patterns=["*.h5", "*.msgpack"],
        )
    )
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(snapshot_path, model_dir)

    metadata = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": _utc_now(),
        "model_id": model_id,
        "architecture": architecture,
        "framework": "transformers",
        "weights_format": "hf_model_dir",
        "source": f"huggingface:{repo_id}",
        "source_url": f"https://huggingface.co/{repo_id}",
        "source_revision": info.sha,
        "license": "See upstream model card license metadata",
        "citation_key": "xiao2018upernet;liu2021swin",
        "citation": f"{UPERNET_CITATION} {SWIN_CITATION}",
        "citation_url": "https://arxiv.org/abs/1807.10221;https://arxiv.org/abs/2103.14030",
        "notes": (
            "Local snapshot for offline transfer-learning init. "
            "Training pipeline uses local_files_only=True and num_labels=2."
        ),
    }
    _write_json(metadata_path, metadata)
    files = _collect_files(bundle_dir)
    _write_json(bundle_dir / "SHA256SUMS.json", {"schema_version": "microseg.sha256.v1", "files": files})

    return {
        "model_id": model_id,
        "architecture": architecture,
        "framework": "transformers",
        "source": f"huggingface:{repo_id}",
        "source_url": f"https://huggingface.co/{repo_id}",
        "source_revision": str(info.sha),
        "bundle_dir": model_id,
        "weights_path": "hf_model",
        "weights_format": "hf_model_dir",
        "metadata_path": "metadata.json",
        "license": "See upstream model card license metadata",
        "citation_key": "xiao2018upernet;liu2021swin",
        "citation": f"{UPERNET_CITATION} {SWIN_CITATION}",
        "citation_url": "https://arxiv.org/abs/1807.10221;https://arxiv.org/abs/2103.14030",
        "notes": f"{architecture} pretrained segmentation snapshot for offline fine-tuning.",
        "files": files,
    }


def _build_smp_segmentation_bundle(spec: dict[str, str], bundle_root: Path, *, force: bool) -> dict[str, Any]:
    import torch

    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise RuntimeError(
            "segmentation_models_pytorch is required to materialize U-Net pretrained bundles. "
            "Install with `pip install segmentation-models-pytorch`."
        ) from exc
    try:
        import timm  # type: ignore

        timm_version = getattr(timm, "__version__", "unknown")
    except Exception:
        timm_version = "unknown"

    model_id = str(spec["model_id"])
    architecture = str(spec["architecture"])
    decoder_name = str(spec.get("decoder_name", "unet")).strip().lower()
    encoder_name = str(spec["encoder_name"])
    bundle_dir = bundle_root / model_id
    weights_dir = bundle_dir / "weights"
    weights_path = weights_dir / f"{model_id}_state_dict.pt"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_factory_by_decoder = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "pspnet": smp.PSPNet,
        "fpn": smp.FPN,
    }
    model_factory = model_factory_by_decoder.get(decoder_name)
    if model_factory is None:
        raise ValueError(
            f"unsupported SMP decoder_name={decoder_name!r}; expected one of: {', '.join(sorted(model_factory_by_decoder))}"
        )
    model = model_factory(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    torch.save(model.state_dict(), weights_path)

    source = (
        "segmentation_models_pytorch:"
        f"{model_factory.__name__}({encoder_name}, encoder_weights=imagenet)"
    )
    citation_by_decoder: dict[str, tuple[str, str, str]] = {
        "unet": (
            "ronneberger2015unet;he2016resnet",
            f"{UNET_CITATION} {RESNET_CITATION}",
            "https://arxiv.org/abs/1505.04597;https://arxiv.org/abs/1512.03385",
        ),
        "unetplusplus": (
            "zhou2018unetplusplus;he2016resnet",
            f"{UNETPP_CITATION} {RESNET_CITATION}",
            "https://arxiv.org/abs/1807.10165;https://arxiv.org/abs/1512.03385",
        ),
        "deeplabv3plus": (
            "chen2018encoderdecoder;he2016resnet",
            f"{DEEPLABV3PLUS_CITATION} {RESNET_CITATION}",
            "https://arxiv.org/abs/1802.02611;https://arxiv.org/abs/1512.03385",
        ),
        "pspnet": (
            "zhao2017pspnet;he2016resnet",
            f"{PSPNET_CITATION} {RESNET_CITATION}",
            "https://arxiv.org/abs/1612.01105;https://arxiv.org/abs/1512.03385",
        ),
        "fpn": (
            "lin2017fpn;he2016resnet",
            f"{FPN_CITATION} {RESNET_CITATION}",
            "https://arxiv.org/abs/1612.03144;https://arxiv.org/abs/1512.03385",
        ),
    }
    citation_key, citation_text, citation_url = citation_by_decoder[decoder_name]
    source_revision = (
        f"smp={getattr(smp, '__version__', 'unknown')};"
        f"torch={getattr(torch, '__version__', 'unknown')};"
        f"timm={timm_version}"
    )
    metadata = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": _utc_now(),
        "model_id": model_id,
        "architecture": architecture,
        "framework": "segmentation_models_pytorch",
        "weights_format": "torch_state_dict",
        "source": source,
        "source_url": "https://github.com/qubvel-org/segmentation_models.pytorch",
        "source_revision": source_revision,
        "license": "See upstream package licenses",
        "citation_key": citation_key,
        "citation": citation_text,
        "citation_url": citation_url,
        "notes": (
            f"Full {model_factory.__name__} state dict initialized from ImageNet-pretrained {encoder_name} encoder. "
            f"Suitable as local transfer-learning bootstrap for {architecture} backend."
        ),
    }
    _write_json(metadata_path, metadata)
    files = _collect_files(bundle_dir)
    _write_json(bundle_dir / "SHA256SUMS.json", {"schema_version": "microseg.sha256.v1", "files": files})

    return {
        "model_id": model_id,
        "architecture": architecture,
        "framework": "segmentation_models_pytorch",
        "source": source,
        "source_url": "https://github.com/qubvel-org/segmentation_models.pytorch",
        "source_revision": source_revision,
        "bundle_dir": model_id,
        "weights_path": f"weights/{model_id}_state_dict.pt",
        "weights_format": "torch_state_dict",
        "metadata_path": "metadata.json",
        "license": "See upstream package licenses",
        "citation_key": citation_key,
        "citation": citation_text,
        "citation_url": citation_url,
        "notes": f"SMP {model_factory.__name__} {encoder_name} ImageNet-initialized state dict for offline fine-tuning.",
        "files": files,
    }


def _build_unet_binary_resnet18_bootstrap_bundle(
    spec: dict[str, str],
    bundle_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    import timm
    import torch

    from src.microseg.training.unet_binary import _build_binary_model

    model_id = str(spec["model_id"])
    architecture = str(spec["architecture"])
    source_model = str(spec["source_model"])
    bundle_dir = bundle_root / model_id
    weights_dir = bundle_dir / "weights"
    weights_path = weights_dir / f"{model_id}_state_dict.pt"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    target_model = _build_binary_model(
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
    state = target_model.state_dict()
    resnet = timm.create_model(source_model, pretrained=True)
    src = resnet.state_dict()

    # Encoder conv/bn warm-start from ResNet18 with channel slicing and kernel resizing.
    state["inc.0.weight"] = _fit_tensor(src["conv1.weight"], state["inc.0.weight"])
    state["inc.1.weight"] = _fit_tensor(src["bn1.weight"], state["inc.1.weight"])
    state["inc.1.bias"] = _fit_tensor(src["bn1.bias"], state["inc.1.bias"])
    state["inc.1.running_mean"] = _fit_tensor(src["bn1.running_mean"], state["inc.1.running_mean"])
    state["inc.1.running_var"] = _fit_tensor(src["bn1.running_var"], state["inc.1.running_var"])
    state["inc.3.weight"] = _fit_tensor(src["layer1.0.conv1.weight"], state["inc.3.weight"])
    state["inc.4.weight"] = _fit_tensor(src["layer1.0.bn1.weight"], state["inc.4.weight"])
    state["inc.4.bias"] = _fit_tensor(src["layer1.0.bn1.bias"], state["inc.4.bias"])
    state["inc.4.running_mean"] = _fit_tensor(src["layer1.0.bn1.running_mean"], state["inc.4.running_mean"])
    state["inc.4.running_var"] = _fit_tensor(src["layer1.0.bn1.running_var"], state["inc.4.running_var"])

    state["down1.1.0.weight"] = _fit_tensor(src["layer1.0.conv2.weight"], state["down1.1.0.weight"])
    state["down1.1.1.weight"] = _fit_tensor(src["layer1.0.bn2.weight"], state["down1.1.1.weight"])
    state["down1.1.1.bias"] = _fit_tensor(src["layer1.0.bn2.bias"], state["down1.1.1.bias"])
    state["down1.1.1.running_mean"] = _fit_tensor(src["layer1.0.bn2.running_mean"], state["down1.1.1.running_mean"])
    state["down1.1.1.running_var"] = _fit_tensor(src["layer1.0.bn2.running_var"], state["down1.1.1.running_var"])
    state["down1.1.3.weight"] = _fit_tensor(src["layer1.1.conv1.weight"], state["down1.1.3.weight"])
    state["down1.1.4.weight"] = _fit_tensor(src["layer1.1.bn1.weight"], state["down1.1.4.weight"])
    state["down1.1.4.bias"] = _fit_tensor(src["layer1.1.bn1.bias"], state["down1.1.4.bias"])
    state["down1.1.4.running_mean"] = _fit_tensor(src["layer1.1.bn1.running_mean"], state["down1.1.4.running_mean"])
    state["down1.1.4.running_var"] = _fit_tensor(src["layer1.1.bn1.running_var"], state["down1.1.4.running_var"])

    state["down2.1.0.weight"] = _fit_tensor(src["layer2.0.conv1.weight"], state["down2.1.0.weight"])
    state["down2.1.1.weight"] = _fit_tensor(src["layer2.0.bn1.weight"], state["down2.1.1.weight"])
    state["down2.1.1.bias"] = _fit_tensor(src["layer2.0.bn1.bias"], state["down2.1.1.bias"])
    state["down2.1.1.running_mean"] = _fit_tensor(src["layer2.0.bn1.running_mean"], state["down2.1.1.running_mean"])
    state["down2.1.1.running_var"] = _fit_tensor(src["layer2.0.bn1.running_var"], state["down2.1.1.running_var"])
    state["down2.1.3.weight"] = _fit_tensor(src["layer2.0.conv2.weight"], state["down2.1.3.weight"])
    state["down2.1.4.weight"] = _fit_tensor(src["layer2.0.bn2.weight"], state["down2.1.4.weight"])
    state["down2.1.4.bias"] = _fit_tensor(src["layer2.0.bn2.bias"], state["down2.1.4.bias"])
    state["down2.1.4.running_mean"] = _fit_tensor(src["layer2.0.bn2.running_mean"], state["down2.1.4.running_mean"])
    state["down2.1.4.running_var"] = _fit_tensor(src["layer2.0.bn2.running_var"], state["down2.1.4.running_var"])

    torch.save({"model_state_dict": state}, weights_path)

    source = f"timm:{source_model}"
    source_revision = f"timm={getattr(timm, '__version__', 'unknown')};torch={getattr(torch, '__version__', 'unknown')}"
    metadata = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": _utc_now(),
        "model_id": model_id,
        "architecture": architecture,
        "framework": "torch",
        "weights_format": "torch_state_dict",
        "source": source,
        "source_url": "https://github.com/huggingface/pytorch-image-models",
        "source_revision": source_revision,
        "license": "See upstream timm model card/license terms",
        "citation_key": "he2016resnet;ronneberger2015unet",
        "citation": f"{RESNET_CITATION} {UNET_CITATION}",
        "citation_url": "https://arxiv.org/abs/1512.03385;https://arxiv.org/abs/1505.04597",
        "notes": (
            "Partial warm-start bundle: maps ResNet18 convolution and normalization weights into internal "
            "unet_binary encoder tensors using channel slicing and kernel resizing. Decoder/head remain default init."
        ),
        "creators": "Ross Wightman (timm) and original ResNet authors",
        "bootstrap_mapping": {
            "source_model": source_model,
            "target_architecture": architecture,
            "mapped_components": ["inc", "down1", "down2"],
            "unmapped_components": ["up1_t", "up1_c", "up2_t", "up2_c", "out"],
            "target_defaults": {"model_base_channels": 16},
        },
    }
    _write_json(metadata_path, metadata)
    files = _collect_files(bundle_dir)
    _write_json(bundle_dir / "SHA256SUMS.json", {"schema_version": "microseg.sha256.v1", "files": files})

    return {
        "model_id": model_id,
        "architecture": architecture,
        "framework": "torch",
        "source": source,
        "source_url": "https://github.com/huggingface/pytorch-image-models",
        "source_revision": source_revision,
        "bundle_dir": model_id,
        "weights_path": f"weights/{model_id}_state_dict.pt",
        "weights_format": "torch_state_dict",
        "metadata_path": "metadata.json",
        "license": "See upstream timm model card/license terms",
        "citation_key": "he2016resnet;ronneberger2015unet",
        "citation": f"{RESNET_CITATION} {UNET_CITATION}",
        "citation_url": "https://arxiv.org/abs/1512.03385;https://arxiv.org/abs/1505.04597",
        "notes": "unet_binary partial warm-start bundle derived from ResNet18 features for offline transfer learning.",
        "files": files,
    }


def _resize_conv_kernel(src, target_hw: tuple[int, int]):
    import torch.nn.functional as F

    if src.ndim != 4:
        raise ValueError("conv kernel resize expects 4D tensor")
    out_ch, in_ch, h, w = src.shape
    if (h, w) == target_hw:
        return src
    flat = src.reshape(out_ch * in_ch, 1, h, w)
    resized = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
    return resized.reshape(out_ch, in_ch, target_hw[0], target_hw[1])


def _fit_tensor(src, target):
    import torch

    out = src.detach().clone()
    if out.ndim == 4 and target.ndim == 4:
        out = _resize_conv_kernel(out, (int(target.shape[-2]), int(target.shape[-1])))
    slices: list[slice] = []
    for dim, want in enumerate(target.shape):
        have = int(out.shape[dim])
        if have >= int(want):
            slices.append(slice(0, int(want)))
        else:
            slices.append(slice(0, have))
    cropped = out[tuple(slices)]
    if tuple(int(v) for v in cropped.shape) == tuple(int(v) for v in target.shape):
        return cropped.to(dtype=target.dtype)
    padded = torch.zeros_like(target)
    copy_slices = tuple(slice(0, int(v)) for v in cropped.shape)
    padded[copy_slices] = cropped.to(dtype=target.dtype)
    return padded


def _map_vit_block_to_transformer(state: dict[str, Any], vit_state: dict[str, Any], *, layer_idx: int, prefix: str) -> None:
    qkv_w = vit_state[f"blocks.{layer_idx}.attn.qkv.weight"]
    qkv_b = vit_state[f"blocks.{layer_idx}.attn.qkv.bias"]
    proj_w = vit_state[f"blocks.{layer_idx}.attn.proj.weight"]
    proj_b = vit_state[f"blocks.{layer_idx}.attn.proj.bias"]
    fc1_w = vit_state[f"blocks.{layer_idx}.mlp.fc1.weight"]
    fc1_b = vit_state[f"blocks.{layer_idx}.mlp.fc1.bias"]
    fc2_w = vit_state[f"blocks.{layer_idx}.mlp.fc2.weight"]
    fc2_b = vit_state[f"blocks.{layer_idx}.mlp.fc2.bias"]
    n1_w = vit_state[f"blocks.{layer_idx}.norm1.weight"]
    n1_b = vit_state[f"blocks.{layer_idx}.norm1.bias"]
    n2_w = vit_state[f"blocks.{layer_idx}.norm2.weight"]
    n2_b = vit_state[f"blocks.{layer_idx}.norm2.bias"]

    key = f"{prefix}.self_attn.in_proj_weight"
    state[key] = _fit_tensor(qkv_w, state[key])
    key = f"{prefix}.self_attn.in_proj_bias"
    state[key] = _fit_tensor(qkv_b, state[key])
    key = f"{prefix}.self_attn.out_proj.weight"
    state[key] = _fit_tensor(proj_w, state[key])
    key = f"{prefix}.self_attn.out_proj.bias"
    state[key] = _fit_tensor(proj_b, state[key])
    key = f"{prefix}.linear1.weight"
    state[key] = _fit_tensor(fc1_w, state[key])
    key = f"{prefix}.linear1.bias"
    state[key] = _fit_tensor(fc1_b, state[key])
    key = f"{prefix}.linear2.weight"
    state[key] = _fit_tensor(fc2_w, state[key])
    key = f"{prefix}.linear2.bias"
    state[key] = _fit_tensor(fc2_b, state[key])
    key = f"{prefix}.norm1.weight"
    state[key] = _fit_tensor(n1_w, state[key])
    key = f"{prefix}.norm1.bias"
    state[key] = _fit_tensor(n1_b, state[key])
    key = f"{prefix}.norm2.weight"
    state[key] = _fit_tensor(n2_w, state[key])
    key = f"{prefix}.norm2.bias"
    state[key] = _fit_tensor(n2_b, state[key])


def _build_internal_transformer_bootstrap_bundle(
    spec: dict[str, str],
    bundle_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    import torch
    import timm

    from src.microseg.training.unet_binary import _build_binary_model

    model_id = str(spec["model_id"])
    architecture = str(spec["architecture"])
    source_model = str(spec["source_model"])
    bundle_dir = bundle_root / model_id
    weights_dir = bundle_dir / "weights"
    weights_path = weights_dir / f"{model_id}_state_dict.pt"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    target_model = _build_binary_model(
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
    state = target_model.state_dict()
    vit = timm.create_model(source_model, pretrained=True)
    vit_state = vit.state_dict()

    if architecture == "segformer_mini":
        state["patch_embed.weight"] = _fit_tensor(vit_state["patch_embed.proj.weight"], state["patch_embed.weight"])
        if "patch_embed.bias" in state:
            state["patch_embed.bias"] = _fit_tensor(vit_state["patch_embed.proj.bias"], state["patch_embed.bias"])
        if "patch_norm.weight" in state:
            state["patch_norm.weight"] = _fit_tensor(vit_state["norm.weight"], state["patch_norm.weight"])
        if "patch_norm.bias" in state:
            state["patch_norm.bias"] = _fit_tensor(vit_state["norm.bias"], state["patch_norm.bias"])
        for idx in range(2):
            _map_vit_block_to_transformer(state, vit_state, layer_idx=idx, prefix=f"transformer.layers.{idx}")
    elif architecture == "transunet_tiny":
        for idx in range(2):
            _map_vit_block_to_transformer(state, vit_state, layer_idx=idx, prefix=f"transformer.layers.{idx}")
    else:
        raise ValueError(f"unsupported internal transformer bootstrap architecture: {architecture}")

    torch.save({"model_state_dict": state}, weights_path)

    source = f"timm:{source_model}"
    source_revision = f"timm={getattr(timm, '__version__', 'unknown')};torch={getattr(torch, '__version__', 'unknown')}"
    metadata = {
        "schema_version": BUNDLE_SCHEMA,
        "created_utc": _utc_now(),
        "model_id": model_id,
        "architecture": architecture,
        "framework": "torch",
        "weights_format": "torch_state_dict",
        "source": source,
        "source_url": f"https://huggingface.co/timm/{source_model}.augreg_in21k_ft_in1k",
        "source_revision": source_revision,
        "license": "See upstream timm model card/license terms",
        "citation_key": "dosovitskiy2020vit",
        "citation": VIT_CITATION,
        "citation_url": "https://arxiv.org/abs/2010.11929",
        "notes": (
            "Partial warm-start bundle: maps ViT-tiny patch16 weights into internal architecture tensor shapes "
            "using channel slicing and kernel resizing. Transformer layers are initialized from first two ViT blocks; "
            "remaining layers stay at default init where unmatched."
        ),
        "creators": "Ross Wightman (timm) and original ViT authors",
        "bootstrap_mapping": {
            "source_model": source_model,
            "target_architecture": architecture,
            "mapped_components": (
                ["transformer.layers.0", "transformer.layers.1", "patch_embed (segformer_mini only)"]
                if architecture == "segformer_mini"
                else ["transformer.layers.0", "transformer.layers.1"]
            ),
            "target_defaults": {
                "model_base_channels": 16,
                "transformer_depth": 2,
                "transformer_num_heads": 4,
                "transformer_mlp_ratio": 2.0,
                "segformer_patch_size": 4,
            },
        },
    }
    _write_json(metadata_path, metadata)
    files = _collect_files(bundle_dir)
    _write_json(bundle_dir / "SHA256SUMS.json", {"schema_version": "microseg.sha256.v1", "files": files})

    return {
        "model_id": model_id,
        "architecture": architecture,
        "framework": "torch",
        "source": source,
        "source_url": f"https://huggingface.co/timm/{source_model}.augreg_in21k_ft_in1k",
        "source_revision": source_revision,
        "bundle_dir": model_id,
        "weights_path": f"weights/{model_id}_state_dict.pt",
        "weights_format": "torch_state_dict",
        "metadata_path": "metadata.json",
        "license": "See upstream timm model card/license terms",
        "citation_key": "dosovitskiy2020vit",
        "citation": VIT_CITATION,
        "citation_url": "https://arxiv.org/abs/2010.11929",
        "notes": (
            f"{architecture} partial warm-start bundle derived from {source_model} "
            "for offline transfer-learning experiments."
        ),
        "files": files,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/stage pretrained bundles for air-gapped transfer.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output root for pretrained bundles and registry.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="all",
        help=(
            "Comma-separated targets or 'all'. Supported: "
            + ",".join(ALL_TARGETS)
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing bundle folders")
    return parser


def _resolve_requested_targets(raw: str) -> tuple[str, ...]:
    items = tuple(sorted({item.strip() for item in str(raw).split(",") if item.strip()}))
    if not items or items == ("all",):
        return ALL_TARGETS
    unknown = [name for name in items if name not in ALL_TARGETS]
    if unknown:
        raise ValueError(
            f"unknown --targets entries: {', '.join(sorted(unknown))}; supported: {', '.join(ALL_TARGETS)}"
        )
    return items


def main() -> None:
    args = _build_parser().parse_args()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    requested = _resolve_requested_targets(str(args.targets))
    records: list[dict[str, Any]] = []
    for target in requested:
        if target in UNET_BINARY_BOOTSTRAP_TARGETS:
            records.append(
                _build_unet_binary_resnet18_bootstrap_bundle(
                    UNET_BINARY_BOOTSTRAP_TARGETS[target],
                    out_root,
                    force=bool(args.force),
                )
            )
            continue
        if target in HF_SEGFORMER_TARGETS:
            records.append(_download_hf_segformer_bundle(HF_SEGFORMER_TARGETS[target], out_root, force=bool(args.force)))
            continue
        if target in HF_UPERNET_TARGETS:
            records.append(_download_hf_upernet_bundle(HF_UPERNET_TARGETS[target], out_root, force=bool(args.force)))
            continue
        if target in SMP_SEGMENTATION_TARGETS:
            records.append(
                _build_smp_segmentation_bundle(SMP_SEGMENTATION_TARGETS[target], out_root, force=bool(args.force))
            )
            continue
        if target in INTERNAL_TRANSFORMER_BOOTSTRAP_TARGETS:
            records.append(
                _build_internal_transformer_bootstrap_bundle(
                    INTERNAL_TRANSFORMER_BOOTSTRAP_TARGETS[target],
                    out_root,
                    force=bool(args.force),
                )
            )
            continue
        raise ValueError(f"unsupported target: {target}")

    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "updated_utc": _utc_now(),
        "models": records,
    }
    _write_json(out_root / "registry.json", registry)

    print(f"pretrained root: {out_root}")
    print(f"registry: {out_root / 'registry.json'}")
    print(f"targets: {', '.join(requested)}")
    for rec in records:
        print(
            f"- {rec['model_id']} | arch={rec['architecture']} | format={rec['weights_format']} "
            f"| bundle={rec['bundle_dir']}"
        )


if __name__ == "__main__":
    main()
