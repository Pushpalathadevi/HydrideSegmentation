"""Unified architecture-aware loading for trainable segmentation backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.plugins import find_repo_root, frozen_checkpoint_map
from src.microseg.training.unet_binary import load_unet_binary_model, predict_unet_binary_mask

_SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "unet_binary",
    "smp_unet_resnet18",
    "smp_unetplusplus_resnet101",
    "smp_deeplabv3plus_resnet101",
    "smp_pspnet_resnet50",
    "smp_fpn_resnet34",
    "transunet_tiny",
    "segformer_mini",
    "hf_segformer_b0",
    "hf_segformer_b2",
    "hf_segformer_b5",
    "hf_upernet_swin_large",
)


@dataclass(frozen=True)
class InferenceModelReference:
    """Resolved model metadata for an inference-capable trained artifact."""

    reference_id: str
    display_name: str
    source: str
    checkpoint_path: str
    architecture: str
    backend_label: str
    run_dir: str = ""
    manifest_path: str = ""


def supported_trainable_architectures() -> tuple[str, ...]:
    """Return architecture identifiers supported by training/inference loader."""

    return _SUPPORTED_ARCHITECTURES


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _architecture_from_payload(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        arch = str(payload.get("model_architecture", "")).strip().lower()
        if arch:
            return arch
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            nested = str(cfg.get("model_architecture", "")).strip().lower()
            if nested:
                return nested
    return ""


def _backend_from_payload(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        backend = str(payload.get("backend", "")).strip().lower()
        if backend:
            return backend
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            nested = str(cfg.get("backend_label", cfg.get("backend", ""))).strip().lower()
            if nested:
                return nested
    return ""


def load_reference_from_run_dir(run_dir: str | Path) -> InferenceModelReference:
    """Resolve model reference from a training run directory."""

    root = Path(run_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")

    report = _read_json(root / "report.json")
    if str(report.get("status", "")).strip().lower() not in {"ok", "success", "completed"}:
        raise ValueError(f"run is not inference-eligible (status={report.get('status', 'unknown')!r}): {root}")

    manifest = _read_json(root / "training_manifest.json")
    resolved_cfg = _read_json(root / "resolved_config.json")

    model_path = str(
        report.get("model_path")
        or manifest.get("model_path")
        or resolved_cfg.get("model_path")
        or ""
    ).strip()
    if not model_path:
        raise ValueError(f"missing model_path in run metadata: {root}")

    candidate = Path(model_path)
    ckpt_path = (candidate if candidate.is_absolute() else (root / candidate)).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint declared by run metadata does not exist: {ckpt_path}")

    architecture = _architecture_from_payload(report, manifest, resolved_cfg)
    if not architecture:
        raise ValueError(f"missing model_architecture metadata in {root}")
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture for inference: {architecture}")

    backend = _backend_from_payload(report, manifest, resolved_cfg) or architecture
    return InferenceModelReference(
        reference_id=f"run::{root.name}",
        display_name=f"Run: {root.name} ({architecture})",
        source="run_dir",
        checkpoint_path=str(ckpt_path),
        architecture=architecture,
        backend_label=backend,
        run_dir=str(root),
        manifest_path=str(root / "training_manifest.json") if (root / "training_manifest.json").exists() else "",
    )


def load_reference_from_registry(model_id: str) -> InferenceModelReference:
    """Resolve model reference from frozen-checkpoint registry entry."""

    records = frozen_checkpoint_map()
    rec = records.get(model_id)
    if rec is None:
        raise KeyError(f"unknown frozen checkpoint model_id: {model_id}")

    hint = str(rec.checkpoint_path_hint or "").strip()
    if not hint:
        raise ValueError(f"registry model has empty checkpoint_path_hint: {model_id}")

    path = Path(hint)
    if not path.is_absolute():
        path = (find_repo_root(Path(__file__)) / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"registry checkpoint path does not exist: {path}")

    architecture = str(rec.model_type or "").strip().lower()
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(f"registry model has unsupported architecture {architecture!r}: {model_id}")

    return InferenceModelReference(
        reference_id=f"registry::{model_id}",
        display_name=f"Registry: {rec.model_nickname} ({architecture})",
        source="registry",
        checkpoint_path=str(path),
        architecture=architecture,
        backend_label=architecture,
    )


def discover_inference_references(
    *,
    runs_root: str | Path | None = None,
    include_registry: bool = True,
) -> tuple[list[InferenceModelReference], list[str]]:
    """Discover inference-capable models from run folders and optionally registry."""

    refs: list[InferenceModelReference] = []
    warnings: list[str] = []

    if runs_root is None:
        try:
            runs_root = find_repo_root(Path(__file__)) / "outputs" / "runs"
        except Exception:
            runs_root = Path("outputs/runs")

    root = Path(runs_root)
    if root.exists() and root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            try:
                refs.append(load_reference_from_run_dir(child))
            except Exception as exc:
                warnings.append(f"skip run {child.name}: {exc}")

    if include_registry:
        try:
            for model_id in sorted(frozen_checkpoint_map().keys()):
                try:
                    refs.append(load_reference_from_registry(model_id))
                except Exception as exc:
                    warnings.append(f"skip registry {model_id}: {exc}")
        except Exception as exc:
            warnings.append(f"failed to read frozen registry: {exc}")

    return refs, warnings


def run_reference_inference(
    image_path: str | Path,
    reference: InferenceModelReference,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run segmentation with a resolved model reference."""

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    bundle = load_unet_binary_model(
        reference.checkpoint_path,
        enable_gpu=enable_gpu,
        device_policy=device_policy,
    )
    pred = predict_unet_binary_mask(image, bundle).astype(np.uint8) * 255
    return image, pred, {
        "reference_id": reference.reference_id,
        "architecture": reference.architecture,
        "backend": reference.backend_label,
        "source": reference.source,
        "checkpoint_path": reference.checkpoint_path,
        "device": bundle.get("device", "cpu"),
    }
