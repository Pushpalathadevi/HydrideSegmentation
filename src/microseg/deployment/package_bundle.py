"""Deployment package contract helpers for model handoff and smoke validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.microseg.training import (
    load_pixel_classifier,
    load_torch_pixel_classifier,
    load_unet_binary_model,
    predict_index_mask,
    predict_index_mask_torch,
    predict_unet_binary_mask,
)


DEPLOYMENT_PACKAGE_SCHEMA = "microseg.deployment_package.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "deployment_package") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_any(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _manifest_file_rows(package_dir: Path, *, skip_names: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip = set(skip_names or set())
    for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(package_dir).as_posix()
        if Path(rel).name in skip:
            continue
        rows.append(
            {
                "path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _extract_runtime_hints(model_path: Path, resolved_cfg: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    suffix = model_path.suffix.lower()
    if model_path.is_dir():
        hints["model_format"] = "directory"
    elif suffix in {".pt", ".pth", ".ckpt"}:
        hints["model_format"] = "torch_checkpoint"
    elif suffix == ".joblib":
        hints["model_format"] = "sklearn_joblib"
    else:
        hints["model_format"] = "unknown"
    for key in [
        "backend",
        "model_architecture",
        "binary_mask_normalization",
        "input_hw",
        "input_policy",
        "val_input_policy",
        "keep_aspect",
        "image_interpolation",
        "mask_interpolation",
        "require_divisible_by",
    ]:
        if key in resolved_cfg:
            hints[key] = resolved_cfg[key]
    return hints


def _predict_from_artifact(
    image_rgb: np.ndarray,
    model_artifact: Path,
    *,
    enable_gpu: bool,
    device_policy: str,
) -> np.ndarray:
    if model_artifact.is_file() and model_artifact.suffix.lower() == ".joblib":
        model = load_pixel_classifier(model_artifact)
        return predict_index_mask(image_rgb, model)

    if model_artifact.is_file() and model_artifact.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        try:
            bundle = load_unet_binary_model(
                model_artifact,
                enable_gpu=bool(enable_gpu),
                device_policy=str(device_policy),
            )
            pred = predict_unet_binary_mask(image_rgb, bundle).astype(np.uint8)
            return pred
        except Exception:
            bundle = load_torch_pixel_classifier(
                model_artifact,
                enable_gpu=bool(enable_gpu),
                device_policy=str(device_policy),
            )
            return predict_index_mask_torch(image_rgb, bundle)

    if model_artifact.is_dir():
        raise ValueError(
            "directory model artifacts are not directly inferable without a model loader contract; "
            "package a concrete checkpoint file for smoke inference"
        )
    raise ValueError(f"unsupported model artifact for deployment smoke: {model_artifact}")


def build_predictor_from_artifact(
    model_artifact: str | Path,
    *,
    enable_gpu: bool,
    device_policy: str,
):
    """Build a reusable predictor callable from a concrete model artifact path."""

    artifact = Path(model_artifact).resolve()
    if artifact.is_file() and artifact.suffix.lower() == ".joblib":
        model = load_pixel_classifier(artifact)
        return lambda image_rgb: predict_index_mask(image_rgb, model)

    if artifact.is_file() and artifact.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        try:
            bundle = load_unet_binary_model(
                artifact,
                enable_gpu=bool(enable_gpu),
                device_policy=str(device_policy),
            )

            def _predict(image_rgb: np.ndarray) -> np.ndarray:
                return predict_unet_binary_mask(image_rgb, bundle).astype(np.uint8)

            return _predict
        except Exception:
            bundle = load_torch_pixel_classifier(
                artifact,
                enable_gpu=bool(enable_gpu),
                device_policy=str(device_policy),
            )
            return lambda image_rgb: predict_index_mask_torch(image_rgb, bundle)

    if artifact.is_dir():
        raise ValueError(
            "directory model artifacts are not directly inferable without a model loader contract; "
            "package a concrete checkpoint file for smoke inference"
        )
    raise ValueError(f"unsupported model artifact for deployment smoke: {artifact}")


def predict_from_artifact(
    image_rgb: np.ndarray,
    model_artifact: str | Path,
    *,
    enable_gpu: bool,
    device_policy: str,
) -> np.ndarray:
    """Public inference helper for deployment runtime tooling."""

    predictor = build_predictor_from_artifact(
        model_artifact,
        enable_gpu=bool(enable_gpu),
        device_policy=str(device_policy),
    )
    return np.asarray(predictor(image_rgb), dtype=np.uint8)


def resolve_model_artifact_from_package(
    package_dir: str | Path,
    *,
    verify_sha256: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Resolve deployment package manifest and concrete model artifact path."""

    validation = validate_deployment_package(package_dir, verify_sha256=verify_sha256)
    if not validation.ok:
        raise RuntimeError("deployment package validation failed: " + "; ".join(validation.errors[:5]))

    root = Path(package_dir).resolve()
    manifest = _read_json(root / "deployment_manifest.json")
    model_rel = str((manifest.get("model") or {}).get("artifact_rel_path", "")).strip()
    if not model_rel:
        raise ValueError("deployment manifest missing model.artifact_rel_path")
    model_artifact = (root / model_rel).resolve()
    if not model_artifact.exists():
        raise FileNotFoundError(f"deployment model artifact missing: {model_artifact}")
    return manifest, model_artifact


@dataclass(frozen=True)
class DeploymentPackageConfig:
    """Input contract for deployment package creation."""

    model_path: str
    output_dir: str = "outputs/deployments"
    package_name: str = ""
    resolved_config_path: str = ""
    training_report_path: str = ""
    evaluation_report_path: str = ""
    extra_paths: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class DeploymentPackageResult:
    """Result payload from deployment package creation."""

    schema_version: str
    created_utc: str
    package_dir: str
    manifest_path: str
    model_artifact_path: str
    copied_files: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DeploymentPackageValidationReport:
    """Validation report for deployment package integrity."""

    schema_version: str
    created_utc: str
    package_dir: str
    manifest_path: str
    ok: bool
    file_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DeploymentSmokeConfig:
    """Runtime smoke-check configuration for a deployment package."""

    package_dir: str
    image_path: str
    output_dir: str = "outputs/deployments/smoke"
    enable_gpu: bool = False
    device_policy: str = "cpu"


@dataclass
class DeploymentSmokeResult:
    """Runtime smoke-check result payload."""

    schema_version: str
    created_utc: str
    package_dir: str
    image_path: str
    output_mask_path: str
    report_path: str
    runtime_seconds: float
    ok: bool
    warnings: list[str] = field(default_factory=list)


def create_deployment_package(config: DeploymentPackageConfig) -> DeploymentPackageResult:
    """Create a deployment package with manifest and artifact checksums."""

    model_path = Path(config.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    package_root = Path(config.output_dir).resolve()
    package_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = _safe_name(config.package_name or model_path.stem, fallback="model")
    package_dir = package_root / f"{base_name}_{timestamp}"
    package_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    model_dst = package_dir / "artifacts" / "model"
    _copy_any(model_path, model_dst)

    cfg_dst = package_dir / "config"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    resolved_cfg: dict[str, Any] = {}
    if str(config.resolved_config_path).strip():
        resolved_cfg_path = Path(config.resolved_config_path).resolve()
        if resolved_cfg_path.exists():
            _copy_any(resolved_cfg_path, cfg_dst / "resolved_config.json")
            resolved_cfg = _read_json(cfg_dst / "resolved_config.json")
        else:
            warnings.append(f"resolved_config_path missing: {resolved_cfg_path}")

    reports_dst = package_dir / "reports"
    reports_dst.mkdir(parents=True, exist_ok=True)
    for src_raw, dst_name in [
        (config.training_report_path, "training_report.json"),
        (config.evaluation_report_path, "evaluation_report.json"),
    ]:
        src_text = str(src_raw).strip()
        if not src_text:
            continue
        src = Path(src_text).resolve()
        if src.exists():
            _copy_any(src, reports_dst / dst_name)
        else:
            warnings.append(f"report path missing: {src}")

    extras_dst = package_dir / "extras"
    extras_dst.mkdir(parents=True, exist_ok=True)
    for idx, raw in enumerate(config.extra_paths, start=1):
        src = Path(str(raw)).resolve()
        if not src.exists():
            warnings.append(f"extra path missing: {src}")
            continue
        name = f"{idx:03d}_{_safe_name(src.name, fallback='extra')}"
        _copy_any(src, extras_dst / name)

    runtime_hints = _extract_runtime_hints(model_path, resolved_cfg)

    manifest_path = package_dir / "deployment_manifest.json"
    manifest = {
        "schema_version": DEPLOYMENT_PACKAGE_SCHEMA,
        "created_utc": _utc_now(),
        "notes": str(config.notes),
        "model": {
            "source_path": str(model_path),
            "artifact_rel_path": model_dst.relative_to(package_dir).as_posix(),
            "is_directory": bool(model_dst.is_dir()),
        },
        "runtime_hints": runtime_hints,
        "preprocess_contract": {
            "binary_mask_normalization": resolved_cfg.get("binary_mask_normalization", "off"),
            "input_hw": resolved_cfg.get("input_hw", [512, 512]),
            "input_policy": resolved_cfg.get("input_policy", "random_crop"),
            "keep_aspect": resolved_cfg.get("keep_aspect", True),
            "image_interpolation": resolved_cfg.get("image_interpolation", "bilinear"),
            "mask_interpolation": resolved_cfg.get("mask_interpolation", "nearest"),
            "require_divisible_by": resolved_cfg.get("require_divisible_by", 32),
        },
        "postprocess_contract": {
            "output_mask_type": "index",
            "binary_export_mapping": {"background": 0, "foreground": 255},
        },
        "files": [],
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = _manifest_file_rows(package_dir, skip_names={manifest_path.name})
    manifest["files"] = rows
    manifest["file_count"] = len(rows)
    manifest["total_size_bytes"] = int(sum(int(row.get("size_bytes", 0)) for row in rows))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return DeploymentPackageResult(
        schema_version="microseg.deployment_package_result.v1",
        created_utc=_utc_now(),
        package_dir=str(package_dir),
        manifest_path=str(manifest_path),
        model_artifact_path=str(model_dst),
        copied_files=int(len(rows)),
        warnings=warnings,
    )


def validate_deployment_package(
    package_dir: str | Path,
    *,
    verify_sha256: bool = True,
) -> DeploymentPackageValidationReport:
    """Validate deployment package manifest and on-disk file integrity."""

    root = Path(package_dir).resolve()
    manifest_path = root / "deployment_manifest.json"
    report = DeploymentPackageValidationReport(
        schema_version="microseg.deployment_package_validation.v1",
        created_utc=_utc_now(),
        package_dir=str(root),
        manifest_path=str(manifest_path),
        ok=False,
    )
    if not root.exists():
        report.errors.append(f"package dir does not exist: {root}")
        return report
    if not manifest_path.exists():
        report.errors.append(f"manifest missing: {manifest_path}")
        return report

    payload = _read_json(manifest_path)
    if payload.get("schema_version") != DEPLOYMENT_PACKAGE_SCHEMA:
        report.errors.append(
            f"unsupported deployment package schema: {payload.get('schema_version')!r}; "
            f"expected {DEPLOYMENT_PACKAGE_SCHEMA!r}"
        )
    model = payload.get("model", {})
    if not isinstance(model, dict):
        report.errors.append("manifest model block must be object")
    else:
        model_rel = str(model.get("artifact_rel_path", "")).strip()
        if not model_rel:
            report.errors.append("manifest model.artifact_rel_path is empty")
        else:
            model_path = root / model_rel
            if not model_path.exists():
                report.errors.append(f"model artifact missing: {model_path}")

    files = payload.get("files", [])
    if not isinstance(files, list):
        report.errors.append("manifest files must be a list")
        files = []
    report.file_count = len(files)
    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            report.errors.append(f"files[{idx}] must be object")
            continue
        rel = str(item.get("path", "")).strip()
        if not rel:
            report.errors.append(f"files[{idx}] missing path")
            continue
        path = (root / rel).resolve()
        if not path.exists() or not path.is_file():
            report.errors.append(f"listed file missing: {path}")
            continue
        size_expected = item.get("size_bytes")
        if size_expected not in (None, ""):
            try:
                expected_int = int(size_expected)
            except Exception:
                report.errors.append(f"files[{idx}] invalid size_bytes: {size_expected!r}")
            else:
                observed = int(path.stat().st_size)
                if observed != expected_int:
                    report.errors.append(
                        f"size mismatch for {rel}: expected {expected_int}, observed {observed}"
                    )
        if bool(verify_sha256):
            expected_sha = str(item.get("sha256", "")).strip().lower()
            if not expected_sha:
                report.errors.append(f"files[{idx}] missing sha256 for validation: {rel}")
            else:
                observed_sha = _sha256_file(path).lower()
                if observed_sha != expected_sha:
                    report.errors.append(
                        f"sha256 mismatch for {rel}: expected {expected_sha}, observed {observed_sha}"
                    )

    report.ok = len(report.errors) == 0
    return report


def run_deployment_smoke(config: DeploymentSmokeConfig) -> DeploymentSmokeResult:
    """Run one-image inference smoke test using a validated deployment package."""

    manifest, model_artifact = resolve_model_artifact_from_package(config.package_dir, verify_sha256=True)
    package_dir = Path(config.package_dir).resolve()
    image_path = Path(config.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"smoke image not found: {image_path}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    start = time.perf_counter()
    pred = predict_from_artifact(
        image_rgb,
        model_artifact,
        enable_gpu=bool(config.enable_gpu),
        device_policy=str(config.device_policy),
    )
    runtime_seconds = float(time.perf_counter() - start)
    pred_u8 = ((np.asarray(pred, dtype=np.uint8) > 0).astype(np.uint8) * 255)

    stem = _safe_name(image_path.stem, fallback="smoke")
    out_input = output_dir / f"{stem}_input.png"
    out_mask = output_dir / f"{stem}_pred_mask.png"
    out_overlay = output_dir / f"{stem}_overlay.png"
    Image.fromarray(image_rgb).save(out_input)
    Image.fromarray(pred_u8).save(out_mask)

    overlay = image_rgb.copy()
    fg = pred_u8 > 0
    overlay[fg, 0] = np.clip(0.65 * overlay[fg, 0] + 0.35 * 255.0, 0, 255).astype(np.uint8)
    overlay[fg, 1] = np.clip(0.65 * overlay[fg, 1], 0, 255).astype(np.uint8)
    overlay[fg, 2] = np.clip(0.65 * overlay[fg, 2], 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_overlay)

    smoke_report = {
        "schema_version": "microseg.deployment_smoke_report.v1",
        "created_utc": _utc_now(),
        "package_dir": str(package_dir),
        "image_path": str(image_path),
        "model_artifact_path": str(model_artifact),
        "runtime_seconds": runtime_seconds,
        "pred_foreground_fraction": float(np.mean(pred_u8 > 0)),
        "output": {
            "input_png": str(out_input),
            "pred_mask_png": str(out_mask),
            "overlay_png": str(out_overlay),
        },
        "environment": {
            "python_executable": os.environ.get("PYTHON_EXECUTABLE", ""),
            "pid": os.getpid(),
        },
    }
    report_path = output_dir / f"{stem}_deployment_smoke_report.json"
    report_path.write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")

    return DeploymentSmokeResult(
        schema_version="microseg.deployment_smoke_result.v1",
        created_utc=_utc_now(),
        package_dir=str(package_dir),
        image_path=str(image_path),
        output_mask_path=str(out_mask),
        report_path=str(report_path),
        runtime_seconds=runtime_seconds,
        ok=True,
        warnings=[],
    )
