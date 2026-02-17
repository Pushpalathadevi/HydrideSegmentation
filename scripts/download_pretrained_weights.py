"""Download and stage local pretrained-weight bundles for air-gapped transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "pre_trained_weights"
REGISTRY_SCHEMA = "microseg.pretrained_weights_registry.v1"
BUNDLE_SCHEMA = "microseg.pretrained_bundle.v1"

SEGFORMER_CITATION = (
    "Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. "
    "\"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.\" "
    "NeurIPS 2021."
)
UNET_CITATION = (
    "Ronneberger, O., Fischer, P., and Brox, T. "
    "\"U-Net: Convolutional Networks for Biomedical Image Segmentation.\" MICCAI 2015."
)
RESNET_CITATION = (
    "He, K., Zhang, X., Ren, S., and Sun, J. "
    "\"Deep Residual Learning for Image Recognition.\" CVPR 2016."
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

SMP_UNET_TARGETS: dict[str, dict[str, str]] = {
    "smp_unet_resnet18": {
        "model_id": "smp_unet_resnet18_imagenet",
        "architecture": "smp_unet_resnet18",
        "encoder_name": "resnet18",
    },
}

ALL_TARGETS = tuple(sorted((*HF_SEGFORMER_TARGETS.keys(), *SMP_UNET_TARGETS.keys())))


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


def _build_smp_unet_bundle(spec: dict[str, str], bundle_root: Path, *, force: bool) -> dict[str, Any]:
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
    encoder_name = str(spec["encoder_name"])
    bundle_dir = bundle_root / model_id
    weights_dir = bundle_dir / "weights"
    weights_path = weights_dir / f"{model_id}_state_dict.pt"
    metadata_path = bundle_dir / "metadata.json"

    if bundle_dir.exists() and force:
        shutil.rmtree(bundle_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    torch.save(model.state_dict(), weights_path)

    source = f"segmentation_models_pytorch:Unet({encoder_name}, encoder_weights=imagenet)"
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
        "citation_key": "ronneberger2015unet;he2016resnet",
        "citation": f"{UNET_CITATION} {RESNET_CITATION}",
        "citation_url": "https://arxiv.org/abs/1505.04597;https://arxiv.org/abs/1512.03385",
        "notes": (
            f"Full U-Net state dict initialized from ImageNet-pretrained {encoder_name} encoder. "
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
        "citation_key": "ronneberger2015unet;he2016resnet",
        "citation": f"{UNET_CITATION} {RESNET_CITATION}",
        "citation_url": "https://arxiv.org/abs/1505.04597;https://arxiv.org/abs/1512.03385",
        "notes": f"SMP U-Net {encoder_name} ImageNet-initialized state dict for offline fine-tuning.",
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
        if target in HF_SEGFORMER_TARGETS:
            records.append(_download_hf_segformer_bundle(HF_SEGFORMER_TARGETS[target], out_root, force=bool(args.force)))
            continue
        if target in SMP_UNET_TARGETS:
            records.append(_build_smp_unet_bundle(SMP_UNET_TARGETS[target], out_root, force=bool(args.force)))
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
