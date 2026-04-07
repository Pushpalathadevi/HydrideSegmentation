"""Support bundle utilities for post-mortem and field diagnostics."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "bundle") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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


def _environment_fingerprint() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "microseg.compatibility_fingerprint.v1",
        "created_utc": _utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        },
    }
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        payload["pip_freeze"] = [line.strip() for line in pip_freeze.splitlines() if line.strip()]
    except Exception as exc:
        payload["pip_freeze_error"] = str(exc)

    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        payload["nvidia_smi"] = [line.strip() for line in nvidia_smi.splitlines() if line.strip()]
    except Exception as exc:
        payload["nvidia_smi_error"] = str(exc)
    return payload


@dataclass(frozen=True)
class SupportBundleConfig:
    """Configuration for collecting a support diagnostic bundle."""

    run_root: str
    output_dir: str = "outputs/support_bundles"
    bundle_name: str = ""
    include_paths: tuple[str, ...] = ()


@dataclass
class SupportBundleResult:
    """Result payload from support bundle collection."""

    schema_version: str
    created_utc: str
    run_root: str
    bundle_dir: str
    zip_path: str
    manifest_path: str
    included_count: int
    missing_paths: list[str] = field(default_factory=list)


def _candidate_artifacts(run_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for rel in [
        "summary.json",
        "summary.html",
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "benchmark_aggregate.csv",
        "benchmark_dashboard.html",
        "logs",
    ]:
        p = run_root / rel
        if p.exists():
            candidates.append(p)
    summary_json = run_root / "summary.json"
    payload = _read_json(summary_json)
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ["train_log", "eval_log", "run_events_log", "train_config", "eval_config", "train_dir", "eval_report"]:
                raw = str(row.get(key, "")).strip()
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = (run_root / p).resolve()
                if p.exists():
                    candidates.append(p)
    uniq: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def write_compatibility_matrix(output_path: str | Path) -> Path:
    """Write runtime compatibility/environment fingerprint JSON."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_environment_fingerprint(), indent=2), encoding="utf-8")
    return out


def create_support_bundle(config: SupportBundleConfig) -> SupportBundleResult:
    """Collect run artifacts, environment details, and produce zipped support bundle."""

    run_root = Path(config.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run_root does not exist: {run_root}")

    out_root = Path(config.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = _safe_name(config.bundle_name or run_root.name, fallback="support")
    bundle_dir = out_root / f"{label}_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    collected: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in _candidate_artifacts(run_root):
        rel = path.name if path.is_file() else path.name
        dst = bundle_dir / "artifacts" / rel
        try:
            _copy_path(path, dst)
            collected.append({"source": str(path), "dest": str(dst), "is_dir": bool(path.is_dir())})
        except Exception as exc:
            missing.append(f"{path} ({exc})")

    for raw in config.include_paths:
        src = Path(str(raw)).resolve()
        if not src.exists():
            missing.append(str(src))
            continue
        dst = bundle_dir / "extras" / _safe_name(src.name, fallback="extra")
        try:
            _copy_path(src, dst)
            collected.append({"source": str(src), "dest": str(dst), "is_dir": bool(src.is_dir())})
        except Exception as exc:
            missing.append(f"{src} ({exc})")

    env_path = bundle_dir / "environment_fingerprint.json"
    write_compatibility_matrix(env_path)

    manifest = {
        "schema_version": "microseg.support_bundle_manifest.v1",
        "created_utc": _utc_now(),
        "run_root": str(run_root),
        "included": collected,
        "missing_paths": missing,
        "environment_fingerprint": str(env_path),
    }
    manifest_path = bundle_dir / "support_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_base = out_root / f"{bundle_dir.name}"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=bundle_dir)

    return SupportBundleResult(
        schema_version="microseg.support_bundle_result.v1",
        created_utc=_utc_now(),
        run_root=str(run_root),
        bundle_dir=str(bundle_dir),
        zip_path=str(zip_path),
        manifest_path=str(manifest_path),
        included_count=len(collected),
        missing_paths=missing,
    )
