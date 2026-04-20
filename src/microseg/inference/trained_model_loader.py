"""Unified architecture-aware loading for trainable segmentation backends."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.microseg.plugins import find_repo_root, frozen_checkpoint_map
from src.microseg.training.unet_binary import load_unet_binary_model, predict_unet_binary_mask
from .gui_preprocessing import (
    GuiInferencePreprocessConfig,
    coerce_gui_inference_preprocess_config,
    load_original_inference_image,
    prepare_gui_inference_input,
    rescale_image_to_original,
    rescale_mask_to_original,
)

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


@dataclass(frozen=True)
class ModelWarmLoadStatus:
    """Background warm-load status for one resolved inference model."""

    reference_id: str
    display_name: str
    status: str
    message: str
    checkpoint_path: str = ""
    load_duration_seconds: float = 0.0
    cache_hit: bool = False


_BUNDLE_CACHE_LOCK = threading.Lock()
_BUNDLE_CACHE: dict[tuple[str, bool, str], dict[str, Any]] = {}
_LOGGER = logging.getLogger(__name__)


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


def _bundle_cache_key(reference: InferenceModelReference, *, enable_gpu: bool, device_policy: str) -> tuple[str, bool, str]:
    return (str(reference.checkpoint_path), bool(enable_gpu), str(device_policy).strip().lower() or "cpu")


def get_or_load_reference_bundle(
    reference: InferenceModelReference,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
    use_cache: bool = True,
) -> tuple[dict[str, Any], bool, float]:
    """Load a trained-model bundle, optionally reusing the process-local cache."""

    key = _bundle_cache_key(reference, enable_gpu=enable_gpu, device_policy=device_policy)
    if use_cache:
        with _BUNDLE_CACHE_LOCK:
            cached = _BUNDLE_CACHE.get(key)
        if cached is not None:
            return cached, True, 0.0

    started = time.perf_counter()
    bundle = load_unet_binary_model(
        reference.checkpoint_path,
        enable_gpu=enable_gpu,
        device_policy=device_policy,
    )
    load_seconds = max(0.0, time.perf_counter() - started)
    if not use_cache:
        return bundle, False, load_seconds

    with _BUNDLE_CACHE_LOCK:
        cached = _BUNDLE_CACHE.get(key)
        if cached is None:
            _BUNDLE_CACHE[key] = bundle
            return bundle, False, load_seconds
        return cached, True, 0.0


def warm_load_reference_bundle(
    reference: InferenceModelReference,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
) -> ModelWarmLoadStatus:
    """Warm-load and cache a reference bundle for later GUI inference reuse."""

    bundle, cache_hit, load_seconds = get_or_load_reference_bundle(
        reference,
        enable_gpu=enable_gpu,
        device_policy=device_policy,
        use_cache=True,
    )
    device = str(bundle.get("device", "cpu"))
    if cache_hit:
        message = f"Model ready from cache on {device}."
    else:
        message = f"Model loaded on {device} in {load_seconds:.2f}s."
    return ModelWarmLoadStatus(
        reference_id=reference.reference_id,
        display_name=reference.display_name,
        status="ready",
        message=message,
        checkpoint_path=reference.checkpoint_path,
        load_duration_seconds=float(load_seconds),
        cache_hit=bool(cache_hit),
    )


def run_reference_inference(
    image_path: str | Path,
    reference: InferenceModelReference,
    *,
    enable_gpu: bool = False,
    device_policy: str = "cpu",
    preprocess_config: GuiInferencePreprocessConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run segmentation with a resolved model reference."""

    image_load_started = time.perf_counter()
    if preprocess_config is None:
        image, _original_channels = load_original_inference_image(image_path)
        image_load_seconds = max(0.0, time.perf_counter() - image_load_started)
        preprocess_seconds = 0.0
        model_input = image if image.ndim == 3 else np.repeat(image[:, :, None], 3, axis=2)
        preprocessing_manifest: dict[str, Any] = {
            "applied": False,
            "original_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "preprocessed_size": {"width": int(model_input.shape[1]), "height": int(model_input.shape[0])},
            "original_channel_count": int(1 if image.ndim == 2 else image.shape[2]),
            "preprocessed_channel_count": int(1 if image.ndim == 2 else image.shape[2]),
            "model_input_channel_count": int(model_input.shape[2]),
            "channel_duplicated": bool(image.ndim == 2),
            "resize": {
                "policy": "none",
                "target_long_side": None,
                "scale": 1.0,
            },
            "contrast": {"enabled": False, "mode": "disabled", "parameters": {}},
            "rescaled_to_original": False,
        }
        display_image = model_input if image.ndim == 2 else image.astype(np.uint8, copy=True)
        _LOGGER.info(
            "GUI_PREPROCESS | image=%s applied=false original=%dx%d preprocessed=%dx%d resize_policy=none scale=1.0000 contrast=disabled channels=%d->%d duplicated=%s",
            image_path,
            int(image.shape[1]),
            int(image.shape[0]),
            int(model_input.shape[1]),
            int(model_input.shape[0]),
            int(1 if image.ndim == 2 else image.shape[2]),
            int(model_input.shape[2]),
            bool(image.ndim == 2),
        )
    else:
        preprocess_started = image_load_started
        prepared = prepare_gui_inference_input(
            image_path,
            coerce_gui_inference_preprocess_config(preprocess_config),
        )
        preprocess_elapsed = max(0.0, time.perf_counter() - preprocess_started)
        image_load_seconds = 0.0
        preprocess_seconds = preprocess_elapsed
        model_input = prepared.model_ready_image
        display_image = rescale_image_to_original(prepared.processed_image, prepared.original_size)
        preprocessing_manifest = dict(prepared.metadata)
        preprocessing_manifest["applied"] = True
        preprocessing_manifest["rescaled_to_original"] = True
        resize_meta = preprocessing_manifest.get("resize", {})
        contrast_meta = preprocessing_manifest.get("contrast", {})
        _LOGGER.info(
            "GUI_PREPROCESS | image=%s applied=true original=%dx%d resized=%dx%d target_long_side=%s scale=%.4f contrast_mode=%s contrast_parameters=%s channels=%d->%d duplicated=%s",
            image_path,
            int(prepared.original_size[0]),
            int(prepared.original_size[1]),
            int(prepared.preprocessed_size[0]),
            int(prepared.preprocessed_size[1]),
            resize_meta.get("target_long_side"),
            float(prepared.resize_scale),
            contrast_meta.get("mode", "disabled"),
            json.dumps(contrast_meta.get("parameters", {}), sort_keys=True),
            int(prepared.original_channel_count),
            int(prepared.output_channel_count),
            bool(prepared.channel_duplicated),
        )

    bundle, cache_hit, bundle_load_seconds = get_or_load_reference_bundle(
        reference,
        enable_gpu=enable_gpu,
        device_policy=device_policy,
    )
    forward_started = time.perf_counter()
    pred = predict_unet_binary_mask(model_input.astype(np.uint8), bundle).astype(np.uint8) * 255
    forward_seconds = max(0.0, time.perf_counter() - forward_started)
    postprocess_started = time.perf_counter()
    if preprocess_config is not None:
        pred = rescale_mask_to_original(
            pred,
            (
                int(preprocessing_manifest["original_size"]["width"]),
                int(preprocessing_manifest["original_size"]["height"]),
            ),
        )
        _LOGGER.info(
            "GUI_POSTPROCESS | image=%s mask_rescaled_to_original=true output=%dx%d",
            image_path,
            int(pred.shape[1]),
            int(pred.shape[0]),
        )
    postprocess_seconds = max(0.0, time.perf_counter() - postprocess_started)
    return display_image.astype(np.uint8, copy=True), pred, {
        "reference_id": reference.reference_id,
        "architecture": reference.architecture,
        "backend": reference.backend_label,
        "source": reference.source,
        "checkpoint_path": reference.checkpoint_path,
        "device": bundle.get("device", "cpu"),
        "preprocessing": preprocessing_manifest,
        "cache": {
            "bundle_cache_hit": bool(cache_hit),
            "cache_key": {
                "checkpoint_path": reference.checkpoint_path,
                "enable_gpu": bool(enable_gpu),
                "device_policy": str(device_policy),
            },
        },
        "timing": {
            "image_load_seconds": float(image_load_seconds),
            "preprocess_seconds": float(preprocess_seconds),
            "bundle_lookup_or_load_seconds": float(bundle_load_seconds),
            "forward_pass_seconds": float(forward_seconds),
            "postprocess_seconds": float(postprocess_seconds),
        },
    }
