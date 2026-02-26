"""Run hydride benchmark suites and produce consolidated comparison dashboards."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.dataops import generate_dataset_split_manifest_from_splits
from src.microseg.io import resolve_config
from src.microseg.plugins import resolve_bundle_paths, resolve_pretrained_record, validate_pretrained_registry


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"suite config must be a mapping: {path}")
    return data


def _ensure_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_overrides(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    sep = "|" if "|" in text else ","
    return [part.strip() for part in text.split(sep) if part.strip()]


def _resolve_path(path_value: str, *, base: Path) -> Path:
    p = Path(str(path_value).strip())
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _resolve_train_config(train_config: str, train_overrides: list[str], *, repo_root: Path) -> dict[str, Any]:
    cfg_path = _resolve_path(train_config, base=repo_root)
    if not cfg_path.exists():
        raise FileNotFoundError(f"train config does not exist: {cfg_path}")
    payload = resolve_config(str(cfg_path), list(train_overrides))
    if not isinstance(payload, dict):
        raise ValueError(f"resolved train config must be a mapping: {cfg_path}")
    payload["_resolved_train_config_path"] = str(cfg_path)
    return payload


def _pretrained_preflight(
    *,
    train_config: str,
    train_overrides: list[str],
    repo_root: Path,
    validation_cache: dict[tuple[str, bool], Any],
) -> tuple[bool, dict[str, Any]]:
    try:
        cfg = _resolve_train_config(train_config, train_overrides, repo_root=repo_root)
    except Exception as exc:
        return (
            False,
            {
                "required": False,
                "reason": f"failed to resolve train config: {exc}",
                "actions": [
                    "fix the train config path or syntax in suite YAML before rerunning",
                ],
            },
        )

    mode = str(cfg.get("pretrained_init_mode", "scratch")).strip().lower()
    if mode in {"", "scratch", "none", "off"}:
        return True, {"required": False, "mode": mode or "scratch", "reason": ""}
    if mode not in {"local", "local_pretrained"}:
        return (
            False,
            {
                "required": True,
                "mode": mode,
                "reason": f"unsupported pretrained_init_mode={mode!r}; expected scratch/local",
                "actions": ["set pretrained_init_mode=scratch or pretrained_init_mode=local"],
            },
        )

    registry_cfg = str(cfg.get("pretrained_registry_path", "pre_trained_weights/registry.json")).strip()
    if not registry_cfg:
        registry_cfg = "pre_trained_weights/registry.json"
    registry_path = _resolve_path(registry_cfg, base=repo_root)
    model_id = str(cfg.get("pretrained_model_id", "")).strip()
    bundle_dir = str(cfg.get("pretrained_bundle_dir", "")).strip()
    verify_sha = bool(cfg.get("pretrained_verify_sha256", True))

    common_actions = [
        "on connected Linux machine: python scripts/download_pretrained_weights.py --targets all --force",
        f"on HPC/repo root: microseg-cli validate-pretrained --registry-path {registry_cfg} --strict",
        "confirm pre_trained_weights/ was copied and extracted at repo root before running suite",
    ]

    if model_id:
        cache_key = (str(registry_path), bool(verify_sha))
        report = validation_cache.get(cache_key)
        if report is None:
            report = validate_pretrained_registry(str(registry_path), verify_sha256=bool(verify_sha))
            validation_cache[cache_key] = report
        if not bool(getattr(report, "ok", False)):
            errors = list(getattr(report, "errors", []))
            details = "; ".join(errors[:5]) if errors else "registry validation failed"
            return (
                False,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": model_id,
                    "registry_path": str(registry_path),
                    "reason": f"pretrained registry invalid: {details}",
                    "actions": common_actions,
                },
            )
        try:
            rec = resolve_pretrained_record(model_id=model_id, registry_path=str(registry_path))
        except Exception as exc:
            return (
                False,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": model_id,
                    "registry_path": str(registry_path),
                    "reason": f"pretrained model_id missing in registry: {exc}",
                    "actions": common_actions,
                },
            )
        try:
            _bundle, weights_path, _metadata = resolve_bundle_paths(rec, registry_path=str(registry_path))
        except Exception as exc:
            return (
                False,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": model_id,
                    "registry_path": str(registry_path),
                    "reason": f"failed to resolve pretrained bundle paths: {exc}",
                    "actions": common_actions,
                },
            )
        weights_format = str(getattr(rec, "weights_format", "")).strip().lower()
        if weights_format == "hf_model_dir":
            ok_weights = weights_path.exists() and weights_path.is_dir()
            expected = "directory"
        else:
            ok_weights = weights_path.exists() and weights_path.is_file()
            expected = "file"
        if not ok_weights:
            return (
                False,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": model_id,
                    "registry_path": str(registry_path),
                    "weights_path": str(weights_path),
                    "reason": f"pretrained weights missing at {weights_path} (expected {expected})",
                    "actions": common_actions,
                },
            )
        return (
            True,
            {
                "required": True,
                "mode": mode,
                "model_id": model_id,
                "registry_path": str(registry_path),
                "weights_path": str(weights_path),
                "reason": "",
            },
        )

    if bundle_dir:
        bundle_path = _resolve_path(bundle_dir, base=repo_root)
        if bundle_path.exists():
            return (
                True,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": "",
                    "bundle_dir": str(bundle_path),
                    "reason": "",
                },
            )
        return (
            False,
            {
                "required": True,
                "mode": mode,
                "model_id": "",
                "bundle_dir": str(bundle_path),
                "reason": f"pretrained_bundle_dir does not exist: {bundle_path}",
                "actions": common_actions,
            },
        )

    return (
        False,
        {
            "required": True,
            "mode": mode,
            "reason": "pretrained_init_mode=local requires pretrained_model_id or pretrained_bundle_dir",
            "actions": [
                "set pretrained_model_id to a model in pre_trained_weights/registry.json",
                "or set pretrained_bundle_dir to an existing local bundle directory",
            ],
        },
    )


def _write_skip_log(
    *,
    log_path: Path,
    run_tag: str,
    reason: str,
    actions: list[str],
    details: dict[str, Any],
) -> None:
    lines = [
        f"[skip] run={run_tag}",
        f"[reason] {reason}",
    ]
    if details:
        lines.append("[details]")
        for key in sorted(details.keys()):
            if key in {"reason", "actions"}:
                continue
            lines.append(f"  - {key}: {details.get(key)}")
    if actions:
        lines.append("[actions]")
        for action in actions:
            lines.append(f"  - {action}")
    lines.append("")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_positive_seconds(value: object) -> float | None:
    try:
        sec = float(value)
    except Exception:
        return None
    if sec <= 0.0:
        return None
    return float(sec)


def _terminate_process(proc: subprocess.Popen[str], *, grace_seconds: float) -> None:
    if proc.poll() is not None:
        return

    sent_term = False
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            sent_term = True
        except Exception:
            sent_term = False
    if not sent_term:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=max(1.0, float(grace_seconds)))
        return
    except Exception:
        pass

    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    else:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    dry_run: bool,
    run_label: str,
    idle_timeout_seconds: float | None,
    wall_timeout_seconds: float | None,
    terminate_grace_seconds: float,
    poll_interval_seconds: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = " ".join(cmd)
    if dry_run:
        log_path.write_text("$ " + cmd_text + "\n[dry-run]\n", encoding="utf-8")
        print(f"[suite] {run_label} dry-run planned | log={log_path}")
        return 0

    print(f"[suite] {run_label} starting | log={log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + cmd_text + "\n\n")
        log_file.flush()

        popen_kwargs: dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)

        started = time.monotonic()
        last_activity = started
        try:
            last_size = log_path.stat().st_size
        except Exception:
            last_size = 0

        poll_seconds = max(0.1, float(poll_interval_seconds))
        grace_seconds = max(1.0, float(terminate_grace_seconds))

        timeout_reason = ""
        timeout_seconds = 0.0
        while True:
            rc = proc.poll()
            if rc is not None:
                print(f"[suite] {run_label} finished | rc={int(rc)} | log={log_path}")
                return int(rc)

            now = time.monotonic()
            try:
                size_now = log_path.stat().st_size
            except Exception:
                size_now = last_size
            if size_now > last_size:
                last_size = size_now
                last_activity = now

            if wall_timeout_seconds is not None and (now - started) > float(wall_timeout_seconds):
                timeout_reason = "wall_timeout"
                timeout_seconds = float(wall_timeout_seconds)
                break
            if idle_timeout_seconds is not None and (now - last_activity) > float(idle_timeout_seconds):
                timeout_reason = "idle_timeout"
                timeout_seconds = float(idle_timeout_seconds)
                break
            time.sleep(poll_seconds)

        log_file.write(
            f"\n[watchdog] {timeout_reason} triggered after {timeout_seconds:.1f}s; "
            "terminating subprocess and continuing suite.\n"
        )
        log_file.flush()
        _terminate_process(proc, grace_seconds=grace_seconds)
        print(
            f"[suite] {run_label} watchdog {timeout_reason} after {timeout_seconds:.1f}s | "
            f"returning rc=124 | log={log_path}"
        )
        return 124


def _resolve_model_path(train_dir: Path) -> Path:
    candidates = [
        train_dir / "best_checkpoint.pt",
        train_dir / "last_checkpoint.pt",
        train_dir / "torch_pixel_classifier.pt",
        train_dir / "pixel_classifier.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no model artifact found in {train_dir}")


def _safe_float(v: object) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


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




def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_int(v: object) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _mean_and_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = [float(v) for v in (_safe_float(r.get(key)) for r in rows) if v is not None]
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(sum(vals) / len(vals)), float(statistics.pstdev(vals))


def _bytes_to_mb(value: int | float | None) -> float:
    if value is None:
        return 0.0
    return float(value) / (1024.0 * 1024.0)


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def _extract_history_series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    vals: list[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        ep = _safe_int(item.get("epoch"))
        val = _safe_float(item.get(key))
        if ep is None or val is None:
            continue
        epochs.append(ep)
        vals.append(float(val))
    return epochs, vals


def _write_curve_plot(
    *,
    output_path: Path,
    title: str,
    x_vals: list[int],
    lines: list[tuple[str, list[float]]],
    y_label: str,
) -> None:
    if not x_vals:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, y_vals in lines:
        if not y_vals:
            continue
        points = min(len(x_vals), len(y_vals))
        if points <= 0:
            continue
        ax.plot(x_vals[:points], y_vals[:points], marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    if any(y for _, y in lines):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _checkpoint_state_dict(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    if model_path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        payload = torch.load(model_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("model_state_dict")
    if isinstance(state, dict):
        return state
    state = payload.get("state_dict")
    if isinstance(state, dict):
        return state
    return None


def _checkpoint_state_stats(model_path: Path) -> dict[str, Any]:
    state = _checkpoint_state_dict(model_path)
    if not isinstance(state, dict):
        return {}
    total = 0
    tensor_count = 0
    total_sum = 0.0
    total_sq = 0.0
    value_count = 0
    min_value: float | None = None
    max_value: float | None = None
    for val in state.values():
        try:
            count = int(val.numel())  # type: ignore[call-arg]
        except Exception:
            continue
        total += count
        try:
            arr = val.detach().to("cpu").reshape(-1)  # type: ignore[attr-defined]
        except Exception:
            arr = None
        if arr is None:
            continue
        try:
            import torch

            arr = arr.to(dtype=torch.float32)
        except Exception:
            continue
        if int(arr.numel()) <= 0:
            continue
        tensor_count += 1
        value_count += int(arr.numel())
        total_sum += float(arr.sum().item())
        total_sq += float((arr * arr).sum().item())
        vmin = float(arr.min().item())
        vmax = float(arr.max().item())
        min_value = vmin if min_value is None else min(min_value, vmin)
        max_value = vmax if max_value is None else max(max_value, vmax)
    out: dict[str, Any] = {}
    if total > 0:
        out["parameter_count"] = int(total)
    if value_count > 0:
        mean = total_sum / float(value_count)
        variance = max(0.0, (total_sq / float(value_count)) - (mean * mean))
        out.update(
            {
                "weight_tensor_count": int(tensor_count),
                "weight_value_count": int(value_count),
                "weight_mean": float(mean),
                "weight_std": float(math.sqrt(variance)),
                "weight_min": float(min_value if min_value is not None else 0.0),
                "weight_max": float(max_value if max_value is not None else 0.0),
            }
        )
    return out


def _read_training_metadata(train_dir: Path, run_tag: str, output_root: Path) -> dict[str, Any]:
    report = _read_json(train_dir / "report.json")
    history_raw = report.get("history", [])
    history = history_raw if isinstance(history_raw, list) else []
    progress = report.get("progress", {})
    if not isinstance(progress, dict):
        progress = {}

    epochs_loss, train_loss = _extract_history_series(history, "train_loss")
    _, val_loss = _extract_history_series(history, "val_loss")
    epochs_acc, train_acc = _extract_history_series(history, "train_accuracy")
    _, val_acc = _extract_history_series(history, "val_accuracy")
    _, train_iou = _extract_history_series(history, "train_iou")
    _, val_iou = _extract_history_series(history, "val_iou")
    _, epoch_runtime = _extract_history_series(history, "epoch_runtime_seconds")

    curves_dir = output_root / "curves"
    loss_curve = curves_dir / f"{run_tag}_loss_curve.png"
    acc_curve = curves_dir / f"{run_tag}_accuracy_curve.png"
    iou_curve = curves_dir / f"{run_tag}_iou_curve.png"

    _write_curve_plot(
        output_path=loss_curve,
        title=f"{run_tag} - Loss vs Epoch",
        x_vals=epochs_loss,
        lines=[("Train Loss", train_loss), ("Val Loss", val_loss)],
        y_label="Loss",
    )
    _write_curve_plot(
        output_path=acc_curve,
        title=f"{run_tag} - Accuracy vs Epoch",
        x_vals=epochs_acc,
        lines=[("Train Accuracy", train_acc), ("Val Accuracy", val_acc)],
        y_label="Accuracy",
    )
    _write_curve_plot(
        output_path=iou_curve,
        title=f"{run_tag} - IoU vs Epoch",
        x_vals=epochs_acc if epochs_acc else epochs_loss,
        lines=[("Train IoU", train_iou), ("Val IoU", val_iou)],
        y_label="IoU",
    )

    epoch_total = _safe_int(progress.get("epochs_total")) or len(history)
    epoch_done = _safe_int(progress.get("epochs_completed")) or len(history)

    avg_epoch_runtime = (sum(epoch_runtime) / len(epoch_runtime)) if epoch_runtime else _safe_float(report.get("runtime_seconds"))
    sample_metric_keys = [
        "pixel_accuracy",
        "macro_f1",
        "mean_iou",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
        "frequency_weighted_iou",
        "foreground_precision",
        "foreground_recall",
        "foreground_specificity",
        "foreground_iou",
        "foreground_dice",
        "false_positive_rate",
        "false_negative_rate",
        "matthews_corrcoef",
        "mask_area_fraction_abs_error",
        "hydride_count_abs_error",
        "hydride_size_wasserstein",
        "hydride_orientation_wasserstein",
    ]
    tracked_samples_raw = report.get("latest_tracked_samples", [])
    tracked_samples: list[dict[str, Any]] = []
    tracked_ious: list[float] = []
    if isinstance(tracked_samples_raw, list):
        for item in tracked_samples_raw:
            if not isinstance(item, dict):
                continue
            iou = _safe_float(item.get("iou"))
            if iou is not None:
                tracked_ious.append(float(iou))
            sample_name = str(item.get("sample_name", "")).strip()
            panel_rel = str(item.get("panel", "")).strip()
            pred_rel = str(item.get("pred", "")).strip()
            gt_rel = str(item.get("gt", "")).strip()
            sample_entry: dict[str, Any] = {
                "sample_name": sample_name,
                "iou": iou,
                "panel_png": str((train_dir / panel_rel).resolve()) if panel_rel else "",
                "pred_png": str((train_dir / pred_rel).resolve()) if pred_rel else "",
                "gt_png": str((train_dir / gt_rel).resolve()) if gt_rel else "",
            }
            for key in sample_metric_keys:
                value = _safe_float(item.get(key))
                if value is not None:
                    sample_entry[key] = float(value)
            tracked_samples.append(sample_entry)

    tracked_history_map: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        epoch_num = _safe_int(item.get("epoch"))
        if epoch_num is None:
            continue
        tracked_epoch = item.get("tracked_samples", [])
        if not isinstance(tracked_epoch, list):
            continue
        for sample in tracked_epoch:
            if not isinstance(sample, dict):
                continue
            name = str(sample.get("sample_name", "")).strip()
            iou = _safe_float(sample.get("iou"))
            panel_rel = str(sample.get("panel", "")).strip()
            if not name or iou is None:
                continue
            tracked_history_map.setdefault(name, []).append(
                {
                    "epoch": int(epoch_num),
                    "iou": float(iou),
                    "panel_png": str((train_dir / panel_rel).resolve()) if panel_rel else "",
                }
            )

    tracked_sample_evolution: list[dict[str, Any]] = []
    for sample_name in sorted(tracked_history_map.keys()):
        records = sorted(tracked_history_map[sample_name], key=lambda v: int(v.get("epoch", 0)))
        epochs = [int(r["epoch"]) for r in records if _safe_int(r.get("epoch")) is not None]
        ious = [float(r["iou"]) for r in records if _safe_float(r.get("iou")) is not None]
        if not epochs or not ious:
            continue
        evo_curve = curves_dir / f"{run_tag}_tracked_{_safe_name(sample_name)}_iou_curve.png"
        _write_curve_plot(
            output_path=evo_curve,
            title=f"{run_tag} - {sample_name} IoU vs Epoch",
            x_vals=epochs,
            lines=[("Tracked Sample IoU", ious)],
            y_label="IoU",
        )
        tracked_sample_evolution.append(
            {
                "sample_name": sample_name,
                "points": len(ious),
                "first_epoch": int(epochs[0]),
                "last_epoch": int(epochs[-1]),
                "first_iou": float(ious[0]),
                "last_iou": float(ious[-1]),
                "delta_iou": float(ious[-1] - ious[0]),
                "best_iou": float(max(ious)),
                "worst_iou": float(min(ious)),
                "latest_panel_png": str(records[-1].get("panel_png", "")),
                "iou_curve_png": str(evo_curve) if evo_curve.exists() else "",
            }
        )

    tracked_deltas = [
        float(item.get("delta_iou"))
        for item in tracked_sample_evolution
        if _safe_float(item.get("delta_iou")) is not None
    ]
    runtime_hardware = report.get("runtime_hardware", {})
    if not isinstance(runtime_hardware, dict):
        runtime_hardware = {}
    compute_effort = report.get("compute_effort", {})
    if not isinstance(compute_effort, dict):
        compute_effort = {}
    weight_stats = report.get("model_weight_statistics", {})
    if not isinstance(weight_stats, dict):
        weight_stats = {}
    runtime_device = str(report.get("device", "")).strip()
    runtime_device_reason = str(report.get("device_reason", "")).strip()

    return {
        "training_status": str(report.get("status", "")),
        "training_runtime_seconds": _safe_float(report.get("runtime_seconds")),
        "training_runtime_human": str(report.get("runtime_human", "")),
        "runtime_device": runtime_device,
        "runtime_device_reason": runtime_device_reason,
        "runtime_hardware": runtime_hardware,
        "training_epochs_total": epoch_total,
        "training_epochs_completed": epoch_done,
        "training_history_points": len(history),
        "best_val_loss_train": _safe_float(report.get("best_val_loss")),
        "last_train_loss": train_loss[-1] if train_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "last_train_accuracy": train_acc[-1] if train_acc else None,
        "last_val_accuracy": val_acc[-1] if val_acc else None,
        "last_train_iou": train_iou[-1] if train_iou else None,
        "last_val_iou": val_iou[-1] if val_iou else None,
        "avg_epoch_runtime_seconds": avg_epoch_runtime,
        "loss_curve_png": str(loss_curve) if loss_curve.exists() else "",
        "accuracy_curve_png": str(acc_curve) if acc_curve.exists() else "",
        "iou_curve_png": str(iou_curve) if iou_curve.exists() else "",
        "tracked_samples_count": len(tracked_samples),
        "tracked_samples_mean_iou": (sum(tracked_ious) / len(tracked_ious)) if tracked_ious else None,
        "tracked_samples_min_iou": min(tracked_ious) if tracked_ious else None,
        "tracked_samples_max_iou": max(tracked_ious) if tracked_ious else None,
        "tracked_samples": tracked_samples,
        "tracked_sample_evolution": tracked_sample_evolution,
        "tracked_sample_evolution_count": len(tracked_sample_evolution),
        "best_tracked_sample_delta_iou": max(tracked_deltas) if tracked_deltas else None,
        "worst_tracked_sample_delta_iou": min(tracked_deltas) if tracked_deltas else None,
        "model_parameter_count_report": _safe_int(report.get("model_parameter_count")),
        "model_trainable_parameter_count": _safe_int(report.get("model_trainable_parameter_count")),
        "model_checkpoint_size_bytes": _safe_int(report.get("model_checkpoint_size_bytes")),
        "model_checkpoint_size_mb": _safe_float(report.get("model_checkpoint_size_mb")),
        "model_weight_tensor_count": _safe_int(weight_stats.get("tensor_count")),
        "model_weight_value_count": _safe_int(weight_stats.get("value_count")),
        "model_weight_mean": _safe_float(weight_stats.get("mean")),
        "model_weight_std": _safe_float(weight_stats.get("std")),
        "model_weight_min": _safe_float(weight_stats.get("min")),
        "model_weight_max": _safe_float(weight_stats.get("max")),
        "compute_train_samples_processed": _safe_int(compute_effort.get("train_samples_processed")),
        "compute_val_samples_processed": _safe_int(compute_effort.get("val_samples_processed")),
        "compute_tracking_samples_processed": _safe_int(compute_effort.get("tracking_samples_processed")),
        "compute_estimated_forward_flops_per_sample": _safe_float(compute_effort.get("estimated_forward_flops_per_sample")),
        "compute_estimated_total_flops": _safe_float(compute_effort.get("estimated_total_flops")),
        "compute_estimated_total_tflops": _safe_float(compute_effort.get("estimated_total_tflops")),
        "compute_flops_estimate_method": str(compute_effort.get("flops_estimate_method", "")),
        "runtime_gpu_name": str(runtime_hardware.get("gpu_name", runtime_hardware.get("nvidia_smi_name", ""))),
        "runtime_gpu_peak_memory_allocated_mb": _safe_float(runtime_hardware.get("gpu_peak_memory_allocated_mb")),
        "runtime_gpu_total_memory_mb": _safe_float(
            runtime_hardware.get("gpu_total_memory_mb", runtime_hardware.get("nvidia_smi_memory_total_mb"))
        ),
        "runtime_gpu_utilization_pct": _safe_float(runtime_hardware.get("nvidia_smi_gpu_utilization_pct")),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        by_model.setdefault(model, []).append(row)

    out: list[dict[str, Any]] = []
    for model, items in sorted(by_model.items()):
        ok_items = [i for i in items if str(i.get("status", "")).strip().lower() == "ok"]
        fail_items = [i for i in items if i not in ok_items]
        mean_pa, std_pa = _mean_and_std(items, "pixel_accuracy")
        mean_f1, std_f1 = _mean_and_std(items, "macro_f1")
        mean_mi, std_mi = _mean_and_std(items, "mean_iou")
        mean_mp, std_mp = _mean_and_std(items, "macro_precision")
        mean_mr, std_mr = _mean_and_std(items, "macro_recall")
        mean_wf1, std_wf1 = _mean_and_std(items, "weighted_f1")
        mean_bacc, std_bacc = _mean_and_std(items, "balanced_accuracy")
        mean_fwiou, std_fwiou = _mean_and_std(items, "frequency_weighted_iou")
        mean_fg_dice, std_fg_dice = _mean_and_std(items, "foreground_dice")
        mean_fg_iou, std_fg_iou = _mean_and_std(items, "foreground_iou")
        mean_fg_precision, _ = _mean_and_std(items, "foreground_precision")
        mean_fg_recall, _ = _mean_and_std(items, "foreground_recall")
        mean_fg_specificity, _ = _mean_and_std(items, "foreground_specificity")
        mean_fpr, _ = _mean_and_std(items, "false_positive_rate")
        mean_fnr, _ = _mean_and_std(items, "false_negative_rate")
        mean_mcc, _ = _mean_and_std(items, "matthews_corrcoef")
        mean_eval_rt, std_eval_rt = _mean_and_std(items, "runtime_seconds")
        mean_train_rt, std_train_rt = _mean_and_std(items, "training_runtime_seconds")
        mean_total_rt, std_total_rt = _mean_and_std(items, "total_runtime_seconds")
        mean_params, _ = _mean_and_std(items, "model_parameter_count")
        mean_trainable_params, _ = _mean_and_std(items, "model_trainable_parameter_count")
        mean_ckpt_mb, _ = _mean_and_std(items, "model_artifact_size_mb")
        mean_weight_mean, _ = _mean_and_std(items, "model_weight_mean")
        mean_weight_std, _ = _mean_and_std(items, "model_weight_std")
        mean_weight_min, _ = _mean_and_std(items, "model_weight_min")
        mean_weight_max, _ = _mean_and_std(items, "model_weight_max")
        mean_total_tflops, _ = _mean_and_std(items, "compute_estimated_total_tflops")
        mean_gpu_peak_mb, _ = _mean_and_std(items, "runtime_gpu_peak_memory_allocated_mb")
        mean_train_loss, _ = _mean_and_std(items, "last_train_loss")
        mean_val_loss, _ = _mean_and_std(items, "last_val_loss")
        mean_train_acc, _ = _mean_and_std(items, "last_train_accuracy")
        mean_val_acc, _ = _mean_and_std(items, "last_val_accuracy")
        mean_train_iou, _ = _mean_and_std(items, "last_train_iou")
        mean_val_iou, _ = _mean_and_std(items, "last_val_iou")
        mean_tracked_iou, _ = _mean_and_std(items, "tracked_samples_mean_iou")
        mean_area_err, _ = _mean_and_std(items, "mask_area_fraction_abs_error")
        mean_count_err, _ = _mean_and_std(items, "hydride_count_abs_error")
        mean_size_w, _ = _mean_and_std(items, "hydride_size_wasserstein")
        mean_orient_w, _ = _mean_and_std(items, "hydride_orientation_wasserstein")
        mean_best_delta_iou, _ = _mean_and_std(items, "best_tracked_sample_delta_iou")
        mean_worst_delta_iou, _ = _mean_and_std(items, "worst_tracked_sample_delta_iou")

        iou_gaps: list[float] = []
        tracked_spans: list[float] = []
        for item in items:
            tr = _safe_float(item.get("last_train_iou"))
            va = _safe_float(item.get("last_val_iou"))
            if tr is not None and va is not None:
                iou_gaps.append(float(tr - va))
            tmax = _safe_float(item.get("tracked_samples_max_iou"))
            tmin = _safe_float(item.get("tracked_samples_min_iou"))
            if tmax is not None and tmin is not None:
                tracked_spans.append(float(tmax - tmin))
        mean_overfit_iou_gap = float(sum(iou_gaps) / len(iou_gaps)) if iou_gaps else 0.0
        mean_tracked_iou_span = float(sum(tracked_spans) / len(tracked_spans)) if tracked_spans else 0.0

        fg_quality = mean_fg_dice if mean_fg_dice > 0 else mean_mi
        quality_score = (
            0.40 * mean_mi
            + 0.25 * mean_f1
            + 0.20 * mean_wf1
            + 0.15 * fg_quality
        )
        runtime_base = mean_total_rt if mean_total_rt > 0 else mean_eval_rt
        efficiency_score = quality_score / (1.0 + math.log1p(max(0.0, runtime_base)))
        robustness_score = (len(ok_items) / len(items)) if items else 0.0
        out.append(
            {
                "model": model,
                "runs": len(items),
                "ok_runs": len(ok_items),
                "failed_runs": len(fail_items),
                "mean_pixel_accuracy": mean_pa,
                "std_pixel_accuracy": std_pa,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_mean_iou": mean_mi,
                "std_mean_iou": std_mi,
                "mean_macro_precision": mean_mp,
                "std_macro_precision": std_mp,
                "mean_macro_recall": mean_mr,
                "std_macro_recall": std_mr,
                "mean_weighted_f1": mean_wf1,
                "std_weighted_f1": std_wf1,
                "mean_balanced_accuracy": mean_bacc,
                "std_balanced_accuracy": std_bacc,
                "mean_frequency_weighted_iou": mean_fwiou,
                "std_frequency_weighted_iou": std_fwiou,
                "mean_foreground_dice": mean_fg_dice,
                "std_foreground_dice": std_fg_dice,
                "mean_foreground_iou": mean_fg_iou,
                "std_foreground_iou": std_fg_iou,
                "mean_foreground_precision": mean_fg_precision,
                "mean_foreground_recall": mean_fg_recall,
                "mean_foreground_specificity": mean_fg_specificity,
                "mean_false_positive_rate": mean_fpr,
                "mean_false_negative_rate": mean_fnr,
                "mean_matthews_corrcoef": mean_mcc,
                "mean_eval_runtime_seconds": mean_eval_rt,
                "std_eval_runtime_seconds": std_eval_rt,
                "mean_training_runtime_seconds": mean_train_rt,
                "std_training_runtime_seconds": std_train_rt,
                "mean_total_runtime_seconds": mean_total_rt,
                "std_total_runtime_seconds": std_total_rt,
                "mean_model_parameter_count": mean_params,
                "mean_model_trainable_parameter_count": mean_trainable_params,
                "mean_model_artifact_size_mb": mean_ckpt_mb,
                "mean_model_weight_mean": mean_weight_mean,
                "mean_model_weight_std": mean_weight_std,
                "mean_model_weight_min": mean_weight_min,
                "mean_model_weight_max": mean_weight_max,
                "mean_compute_estimated_total_tflops": mean_total_tflops,
                "mean_runtime_gpu_peak_memory_allocated_mb": mean_gpu_peak_mb,
                "mean_last_train_loss": mean_train_loss,
                "mean_last_val_loss": mean_val_loss,
                "mean_last_train_accuracy": mean_train_acc,
                "mean_last_val_accuracy": mean_val_acc,
                "mean_last_train_iou": mean_train_iou,
                "mean_last_val_iou": mean_val_iou,
                "mean_tracked_samples_iou": mean_tracked_iou,
                "mean_mask_area_fraction_abs_error": mean_area_err,
                "mean_hydride_count_abs_error": mean_count_err,
                "mean_hydride_size_wasserstein": mean_size_w,
                "mean_hydride_orientation_wasserstein": mean_orient_w,
                "mean_overfit_iou_gap": mean_overfit_iou_gap,
                "mean_tracked_iou_span": mean_tracked_iou_span,
                "mean_best_tracked_sample_delta_iou": mean_best_delta_iou,
                "mean_worst_tracked_sample_delta_iou": mean_worst_delta_iou,
                "quality_score": quality_score,
                "efficiency_score": efficiency_score,
                "robustness_score": robustness_score,
            }
        )

    if not out:
        return out

    quality_sorted = sorted(
        out,
        key=lambda item: (
            -float(item.get("quality_score", 0.0)),
            -float(item.get("mean_mean_iou", 0.0)),
            float(item.get("mean_total_runtime_seconds", 0.0)),
        ),
    )
    efficiency_sorted = sorted(
        out,
        key=lambda item: (
            -float(item.get("efficiency_score", 0.0)),
            -float(item.get("mean_mean_iou", 0.0)),
        ),
    )
    runtime_sorted = sorted(
        out,
        key=lambda item: (
            float(item.get("mean_total_runtime_seconds", 0.0)),
            -float(item.get("mean_mean_iou", 0.0)),
        ),
    )
    robust_sorted = sorted(
        out,
        key=lambda item: (
            -float(item.get("robustness_score", 0.0)),
            -float(item.get("mean_mean_iou", 0.0)),
        ),
    )

    rank_quality = {item["model"]: idx + 1 for idx, item in enumerate(quality_sorted)}
    rank_efficiency = {item["model"]: idx + 1 for idx, item in enumerate(efficiency_sorted)}
    rank_runtime = {item["model"]: idx + 1 for idx, item in enumerate(runtime_sorted)}
    rank_robustness = {item["model"]: idx + 1 for idx, item in enumerate(robust_sorted)}
    for item in out:
        model = str(item.get("model", ""))
        item["rank_quality"] = int(rank_quality.get(model, 0))
        item["rank_efficiency"] = int(rank_efficiency.get(model, 0))
        item["rank_runtime"] = int(rank_runtime.get(model, 0))
        item["rank_robustness"] = int(rank_robustness.get(model, 0))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _write_dashboard(path: Path, rows: list[dict[str, Any]], agg: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_runs = len(rows)
    failed_runs = len([r for r in rows if str(r.get("status", "")).strip().lower() != "ok"])
    ok_runs = total_runs - failed_runs

    def _pick_best(items: list[dict[str, Any]], key: str, *, higher_is_better: bool = True) -> dict[str, Any] | None:
        if not items:
            return None
        filtered = [item for item in items if _safe_float(item.get(key)) is not None]
        if not filtered:
            return None
        return sorted(filtered, key=lambda item: float(item.get(key, 0.0)), reverse=higher_is_better)[0]

    best_quality = _pick_best(agg, "quality_score", higher_is_better=True)
    best_efficiency = _pick_best(agg, "efficiency_score", higher_is_better=True)
    fastest = _pick_best(agg, "mean_total_runtime_seconds", higher_is_better=False)

    def _summary_line(item: dict[str, Any] | None, metric_key: str) -> str:
        if item is None:
            return "n/a"
        model = html.escape(str(item.get("model", "")))
        value = float(item.get(metric_key, 0.0))
        return f"{model} ({value:.6f})"

    tracked_sample_metric_order = [
        "pixel_accuracy",
        "macro_f1",
        "mean_iou",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
        "frequency_weighted_iou",
        "foreground_precision",
        "foreground_recall",
        "foreground_specificity",
        "foreground_iou",
        "foreground_dice",
        "false_positive_rate",
        "false_negative_rate",
        "matthews_corrcoef",
        "mask_area_fraction_abs_error",
        "hydride_count_abs_error",
        "hydride_size_wasserstein",
        "hydride_orientation_wasserstein",
    ]

    def _tracked_sample_metrics_html(sample: dict[str, Any]) -> str:
        items: list[str] = []
        for key in tracked_sample_metric_order:
            value = _safe_float(sample.get(key))
            if value is None:
                continue
            items.append(
                "<li><b>"
                + html.escape(key.replace("_", " "))
                + "</b>: "
                + f"{float(value):.6f}"
                + "</li>"
            )
        if not items:
            return ""
        return "<ul style='margin:8px 0 0 18px;'>" + "".join(items) + "</ul>"

    def _metrics_block_html(items: list[tuple[str, object]]) -> str:
        rows_html: list[str] = []
        for label, raw_value in items:
            value = _safe_float(raw_value)
            value_text = f"{float(value):.6f}" if value is not None else "n/a"
            rows_html.append(
                "<li><b>"
                + html.escape(str(label))
                + "</b>: "
                + value_text
                + "</li>"
            )
        if not rows_html:
            return ""
        return "<ul style='margin:8px 0 0 18px;'>" + "".join(rows_html) + "</ul>"

    style = (
        "<style>"
        "body{font-family:Arial,sans-serif;margin:16px;line-height:1.35;}"
        ".cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;}"
        ".card{border:1px solid #bbb;border-radius:8px;padding:10px 12px;min-width:240px;background:#fafafa;}"
        "table{border-collapse:collapse;margin:10px 0;width:100%;}"
        "th,td{border:1px solid #bbb;padding:6px 8px;vertical-align:top;}"
        "th{background:#f0f0f0;position:sticky;top:0;}"
        ".mono{font-family:Menlo,Consolas,monospace;font-size:12px;}"
        "</style>"
    )
    lines: list[str] = [
        "<html><head><meta charset='utf-8'><title>Hydride Benchmark Dashboard</title>"
        + style
        + "</head><body>",
        "<h1>Hydride Benchmark Dashboard</h1>",
        "<div class='cards'>",
        f"<div class='card'><b>Total Runs</b><div>{total_runs}</div></div>",
        f"<div class='card'><b>Successful Runs</b><div>{ok_runs}</div></div>",
        f"<div class='card'><b>Failed Runs</b><div>{failed_runs}</div></div>",
        f"<div class='card'><b>Top Quality</b><div>{_summary_line(best_quality, 'quality_score')}</div></div>",
        f"<div class='card'><b>Top Efficiency</b><div>{_summary_line(best_efficiency, 'efficiency_score')}</div></div>",
        f"<div class='card'><b>Fastest Model (mean total runtime)</b><div>{_summary_line(fastest, 'mean_total_runtime_seconds')}</div></div>",
        "</div>",
        "<h2>Model Summary</h2>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Model</th><th>Rank Quality</th><th>Rank Efficiency</th><th>Rank Runtime</th><th>Rank Robustness</th><th>Runs</th><th>OK</th><th>Failed</th><th>Quality Score</th><th>Efficiency Score</th><th>Pixel Acc (mean±std)</th><th>Macro F1 (mean±std)</th><th>Mean IoU (mean±std)</th><th>Weighted F1 (mean±std)</th><th>Balanced Acc (mean±std)</th><th>Foreground Dice (mean±std)</th><th>Foreground Specificity (mean)</th><th>FPR (mean)</th><th>FNR (mean)</th><th>MCC (mean)</th><th>Tracked Val Sample IoU (mean)</th><th>Tracked IoU Span (mean)</th><th>Overfit Gap IoU (train-val, mean)</th><th>Eval Runtime (s)</th><th>Train Runtime (s)</th><th>Total Runtime (s)</th><th>Params (mean)</th><th>Trainable Params (mean)</th><th>Model Size MB (mean)</th><th>Weight Mean</th><th>Weight Std</th><th>Weight Min</th><th>Weight Max</th><th>Total TFLOPs (mean est.)</th><th>GPU Peak MB (mean)</th></tr>",
    ]
    for item in agg:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(item['model']))}</td>"
            f"<td>{int(item.get('rank_quality', 0))}</td>"
            f"<td>{int(item.get('rank_efficiency', 0))}</td>"
            f"<td>{int(item.get('rank_runtime', 0))}</td>"
            f"<td>{int(item.get('rank_robustness', 0))}</td>"
            f"<td>{int(item.get('runs', 0))}</td>"
            f"<td>{int(item.get('ok_runs', 0))}</td>"
            f"<td>{int(item.get('failed_runs', 0))}</td>"
            f"<td>{float(item.get('quality_score', 0.0)):.6f}</td>"
            f"<td>{float(item.get('efficiency_score', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_pixel_accuracy', 0.0)):.6f} ± {float(item.get('std_pixel_accuracy', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_macro_f1', 0.0)):.6f} ± {float(item.get('std_macro_f1', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_mean_iou', 0.0)):.6f} ± {float(item.get('std_mean_iou', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_weighted_f1', 0.0)):.6f} ± {float(item.get('std_weighted_f1', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_balanced_accuracy', 0.0)):.6f} ± {float(item.get('std_balanced_accuracy', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_foreground_dice', 0.0)):.6f} ± {float(item.get('std_foreground_dice', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_foreground_specificity', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_false_positive_rate', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_false_negative_rate', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_matthews_corrcoef', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_tracked_samples_iou', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_tracked_iou_span', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_overfit_iou_gap', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_eval_runtime_seconds', 0.0)):.2f}</td>"
            f"<td>{float(item.get('mean_training_runtime_seconds', 0.0)):.2f}</td>"
            f"<td>{float(item.get('mean_total_runtime_seconds', 0.0)):.2f}</td>"
            f"<td>{float(item.get('mean_model_parameter_count', 0.0)):.0f}</td>"
            f"<td>{float(item.get('mean_model_trainable_parameter_count', 0.0)):.0f}</td>"
            f"<td>{float(item.get('mean_model_artifact_size_mb', 0.0)):.2f}</td>"
            f"<td>{float(item.get('mean_model_weight_mean', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_model_weight_std', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_model_weight_min', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_model_weight_max', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_compute_estimated_total_tflops', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_runtime_gpu_peak_memory_allocated_mb', 0.0)):.2f}</td>"
            "</tr>"
        )
    lines.extend(
        [
            "</table>",
            "<h2>Scientific Error Summary (Lower Is Better)</h2>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr><th>Model</th><th>Area Fraction Abs Error</th><th>Hydride Count Abs Error</th><th>Size Wasserstein</th><th>Orientation Wasserstein</th><th>Best Tracked Delta IoU (mean)</th><th>Worst Tracked Delta IoU (mean)</th></tr>",
        ]
    )
    for item in agg:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('model', '')))}</td>"
            f"<td>{float(item.get('mean_mask_area_fraction_abs_error', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_hydride_count_abs_error', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_hydride_size_wasserstein', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_hydride_orientation_wasserstein', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_best_tracked_sample_delta_iou', 0.0)):.6f}</td>"
            f"<td>{float(item.get('mean_worst_tracked_sample_delta_iou', 0.0)):.6f}</td>"
            "</tr>"
        )
    lines.extend(
        [
            "</table>",
            "<h2>Run-Level Results</h2>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr><th>Model</th><th>Seed</th><th>Status</th><th>Status Detail</th><th>Backend</th><th>Architecture</th><th>Init</th><th>Runtime Device</th><th>GPU</th><th>Pixel Acc</th><th>Macro F1</th><th>Mean IoU</th><th>Weighted F1</th><th>Balanced Acc</th><th>Foreground Dice</th><th>Foreground Precision</th><th>Foreground Recall</th><th>Foreground Specificity</th><th>FPR</th><th>FNR</th><th>MCC</th><th>Area Abs Err</th><th>Count Abs Err</th><th>Size W</th><th>Orientation W</th><th>Tracked Val Sample IoU</th><th>Tracked Sample Count</th><th>Tracked Evol. Curves</th><th>Eval Runtime (s)</th><th>Train Runtime (s)</th><th>Total Runtime (s)</th><th>Params</th><th>Trainable Params</th><th>Model Size MB</th><th>Weight Mean</th><th>Weight Std</th><th>Weight Min</th><th>Weight Max</th><th>Total TFLOPs (est.)</th><th>GPU Peak MB</th><th>Hyperparameters</th><th>Train Config</th><th>Train Dir</th><th>Eval Report</th><th>Loss Curve</th><th>Acc Curve</th><th>IoU Curve</th></tr>",
        ]
    )
    for row in rows:
        hparams = (
            f"ep={row.get('resolved_epochs','')}, "
            f"bs={row.get('resolved_batch_size','')}, "
            f"lr={row.get('resolved_learning_rate','')}, "
            f"wd={row.get('resolved_weight_decay','')}"
        )
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('model','')))}</td>"
            f"<td>{html.escape(str(row.get('seed','')))}</td>"
            f"<td>{html.escape(str(row.get('status','')))}</td>"
            f"<td>{html.escape(str(row.get('status_message','')))}</td>"
            f"<td>{html.escape(str(row.get('resolved_backend') or row.get('backend','')))}</td>"
            f"<td>{html.escape(str(row.get('resolved_model_architecture','')))}</td>"
            f"<td>{html.escape(str(row.get('model_initialization','')))}</td>"
            f"<td>{html.escape(str(row.get('runtime_device','')))}</td>"
            f"<td>{html.escape(str(row.get('runtime_gpu_name','')))}</td>"
            f"<td>{_safe_float(row.get('pixel_accuracy')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('macro_f1')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('mean_iou')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('weighted_f1')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('balanced_accuracy')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('foreground_dice')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('foreground_precision')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('foreground_recall')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('foreground_specificity')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('false_positive_rate')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('false_negative_rate')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('matthews_corrcoef')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('mask_area_fraction_abs_error')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('hydride_count_abs_error')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('hydride_size_wasserstein')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('hydride_orientation_wasserstein')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('tracked_samples_mean_iou')) or 0.0:.6f}</td>"
            f"<td>{_safe_int(row.get('tracked_samples_count')) or 0}</td>"
            f"<td>{_safe_int(row.get('tracked_sample_evolution_count')) or 0}</td>"
            f"<td>{_safe_float(row.get('runtime_seconds')) or 0.0:.2f}</td>"
            f"<td>{_safe_float(row.get('training_runtime_seconds')) or 0.0:.2f}</td>"
            f"<td>{_safe_float(row.get('total_runtime_seconds')) or 0.0:.2f}</td>"
            f"<td>{_safe_int(row.get('model_parameter_count')) or 0}</td>"
            f"<td>{_safe_int(row.get('model_trainable_parameter_count')) or 0}</td>"
            f"<td>{_safe_float(row.get('model_artifact_size_mb')) or 0.0:.2f}</td>"
            f"<td>{_safe_float(row.get('model_weight_mean')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('model_weight_std')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('model_weight_min')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('model_weight_max')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('compute_estimated_total_tflops')) or 0.0:.6f}</td>"
            f"<td>{_safe_float(row.get('runtime_gpu_peak_memory_allocated_mb')) or 0.0:.2f}</td>"
            f"<td>{html.escape(hparams)}</td>"
            f"<td>{html.escape(str(row.get('train_config','')))}</td>"
            f"<td>{html.escape(str(row.get('train_dir','')))}</td>"
            f"<td>{html.escape(str(row.get('eval_report','')))}</td>"
            f"<td>{html.escape(str(row.get('loss_curve_png','')))}</td>"
            f"<td>{html.escape(str(row.get('accuracy_curve_png','')))}</td>"
            f"<td>{html.escape(str(row.get('iou_curve_png','')))}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "<h2>Training Curve Gallery</h2>"])
    for row in rows:
        title = f"{row.get('model', '')} seed {row.get('seed', '')}"
        loss_curve = str(row.get("loss_curve_png", ""))
        acc_curve = str(row.get("accuracy_curve_png", ""))
        iou_curve = str(row.get("iou_curve_png", ""))
        if not loss_curve and not acc_curve and not iou_curve:
            continue
        lines.append(f"<h3>{html.escape(title)}</h3>")
        lines.append("<div style='display:flex;gap:12px;flex-wrap:wrap;'>")
        if loss_curve:
            lines.append(
                "<div><div><b>Loss vs Epoch</b></div>"
                f"<img src='{html.escape(loss_curve)}' style='max-width:520px;border:1px solid #333;'>"
                + _metrics_block_html(
                    [
                        ("Last Train Loss", row.get("last_train_loss")),
                        ("Last Val Loss", row.get("last_val_loss")),
                        ("Best Val Loss", row.get("best_val_loss_train")),
                    ]
                )
                + "</div>"
            )
        if acc_curve:
            lines.append(
                "<div><div><b>Accuracy vs Epoch</b></div>"
                f"<img src='{html.escape(acc_curve)}' style='max-width:520px;border:1px solid #333;'>"
                + _metrics_block_html(
                    [
                        ("Last Train Accuracy", row.get("last_train_accuracy")),
                        ("Last Val Accuracy", row.get("last_val_accuracy")),
                        ("Eval Pixel Accuracy", row.get("pixel_accuracy")),
                    ]
                )
                + "</div>"
            )
        if iou_curve:
            lines.append(
                "<div><div><b>IoU vs Epoch</b></div>"
                f"<img src='{html.escape(iou_curve)}' style='max-width:520px;border:1px solid #333;'>"
                + _metrics_block_html(
                    [
                        ("Last Train IoU", row.get("last_train_iou")),
                        ("Last Val IoU", row.get("last_val_iou")),
                        ("Eval Mean IoU", row.get("mean_iou")),
                        ("Tracked Samples Mean IoU", row.get("tracked_samples_mean_iou")),
                    ]
                )
                + "</div>"
            )
        lines.append("</div>")
    lines.extend(["<h2>Tracked Sample Evolution (IoU vs Epoch)</h2>"])
    for row in rows:
        tracked_evolution = row.get("tracked_sample_evolution")
        if not isinstance(tracked_evolution, list) or not tracked_evolution:
            continue
        title = f"{row.get('model', '')} seed {row.get('seed', '')}"
        lines.append(f"<h3>{html.escape(title)}</h3>")
        lines.append(
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Sample</th><th>Points</th><th>First IoU</th><th>Last IoU</th><th>Delta IoU</th><th>Best IoU</th><th>Worst IoU</th><th>Curve</th></tr>"
        )
        for item in tracked_evolution:
            if not isinstance(item, dict):
                continue
            curve_png = str(item.get("iou_curve_png", "")).strip()
            curve_cell = html.escape(curve_png) if curve_png else "-"
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('sample_name', '')))}</td>"
                f"<td>{int(_safe_int(item.get('points')) or 0)}</td>"
                f"<td>{_safe_float(item.get('first_iou')) or 0.0:.6f}</td>"
                f"<td>{_safe_float(item.get('last_iou')) or 0.0:.6f}</td>"
                f"<td>{_safe_float(item.get('delta_iou')) or 0.0:.6f}</td>"
                f"<td>{_safe_float(item.get('best_iou')) or 0.0:.6f}</td>"
                f"<td>{_safe_float(item.get('worst_iou')) or 0.0:.6f}</td>"
                f"<td class='mono'>{curve_cell}</td>"
                "</tr>"
            )
        lines.append("</table>")
        lines.append("<div style='display:flex;gap:12px;flex-wrap:wrap;'>")
        for item in tracked_evolution:
            if not isinstance(item, dict):
                continue
            curve_png = str(item.get("iou_curve_png", "")).strip()
            if not curve_png:
                continue
            lines.append(
                "<div><div><b>"
                + html.escape(str(item.get("sample_name", "")))
                + "</b></div>"
                + f"<img src='{html.escape(curve_png)}' style='max-width:520px;border:1px solid #333;'></div>"
            )
        lines.append("</div>")

    lines.extend(["<h2>Validation Sample Panels</h2>"])
    for row in rows:
        tracked_samples = row.get("tracked_samples")
        if not isinstance(tracked_samples, list) or not tracked_samples:
            continue
        title = f"{row.get('model', '')} seed {row.get('seed', '')}"
        lines.append(f"<h3>{html.escape(title)}</h3>")
        lines.append("<div style='display:flex;gap:12px;flex-wrap:wrap;'>")
        for sample in tracked_samples:
            if not isinstance(sample, dict):
                continue
            panel_png = str(sample.get("panel_png", "")).strip()
            sample_name = str(sample.get("sample_name", "")).strip()
            sample_iou = _safe_float(sample.get("iou"))
            if not panel_png:
                continue
            iou_text = f" (IoU: {float(sample_iou):.6f})" if sample_iou is not None else ""
            lines.append(
                "<div><div><b>"
                + html.escape(sample_name or "tracked_sample")
                + "</b>"
                + iou_text
                + "</div>"
                + f"<img src='{html.escape(panel_png)}' style='max-width:520px;border:1px solid #333;'>"
                + _tracked_sample_metrics_html(sample)
                + "</div>"
            )
        lines.append("</div>")
    lines.extend(["</body></html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(cfg_path: Path, *, dry_run: bool, strict: bool, skip_train: bool, skip_eval: bool) -> int:
    cfg = _load_yaml(cfg_path)
    dataset_dir = str(cfg.get("dataset_dir", "")).strip()
    output_root = Path(str(cfg.get("output_root", "outputs/benchmarks/suite_runs")).strip())
    eval_config = str(cfg.get("eval_config", "configs/hydride/evaluate.hydride.yml")).strip()
    eval_split = str(cfg.get("eval_split", "test")).strip()
    python_exe = str(cfg.get("python_executable", sys.executable)).strip() or sys.executable
    seeds = [int(v) for v in _ensure_list(cfg.get("seeds", [42]))]
    experiments = _ensure_list(cfg.get("experiments", []))
    benchmark_mode = bool(cfg.get("benchmark_mode", True))
    expected_manifest_sha = str(cfg.get("expected_dataset_manifest_sha256", "")).strip().lower()
    expected_split_id_file = str(cfg.get("expected_split_id_file", "")).strip()
    idle_timeout_seconds = _safe_positive_seconds(cfg.get("command_idle_timeout_seconds"))
    wall_timeout_seconds = _safe_positive_seconds(cfg.get("command_wall_timeout_seconds"))
    terminate_grace_seconds = _safe_positive_seconds(cfg.get("command_terminate_grace_seconds")) or 30.0
    poll_interval_seconds = _safe_positive_seconds(cfg.get("command_poll_interval_seconds")) or 1.0

    if not dataset_dir:
        raise ValueError("dataset_dir is required in suite config")

    dataset_manifest = Path(dataset_dir) / "dataset_manifest.json"

    if benchmark_mode:
        if not dataset_manifest.exists():
            dataset_manifest = generate_dataset_split_manifest_from_splits(dataset_dir)
            print(f"[benchmark-mode] generated missing dataset manifest: {dataset_manifest}")
        if expected_manifest_sha:
            observed_sha = _sha256_file(dataset_manifest).lower()
            if observed_sha != expected_manifest_sha:
                raise RuntimeError(
                    "dataset manifest hash mismatch in benchmark mode: "
                    f"expected={expected_manifest_sha} observed={observed_sha}"
                )
        if expected_split_id_file:
            split_file = Path(expected_split_id_file)
            if not split_file.exists():
                raise FileNotFoundError(f"expected_split_id_file missing: {split_file}")
            expected_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest = _read_json(dataset_manifest)
            observed_ids = sorted((manifest.get("sample_to_split") or {}).keys())
            if sorted(expected_ids) != observed_ids:
                raise RuntimeError("dataset split IDs do not match expected_split_id_file in benchmark_mode")

    if not experiments:
        raise ValueError("suite config must include experiments")

    repo_root = Path(__file__).resolve().parents[1]
    cli = repo_root / "scripts" / "microseg_cli.py"
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "[suite] execution policy: "
        f"idle_timeout={idle_timeout_seconds if idle_timeout_seconds is not None else 'off'}s "
        f"wall_timeout={wall_timeout_seconds if wall_timeout_seconds is not None else 'off'}s "
        f"terminate_grace={terminate_grace_seconds:.1f}s poll_interval={poll_interval_seconds:.1f}s"
    )

    rows: list[dict[str, Any]] = []
    failures = 0
    preflight_cache: dict[tuple[str, bool], Any] = {}

    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        model_name = str(exp.get("name", "")).strip() or "unnamed_model"
        train_config = str(exp.get("train_config", "")).strip()
        if not train_config:
            raise ValueError(f"experiment '{model_name}' missing train_config")
        train_overrides = _parse_overrides(exp.get("train_overrides"))
        eval_overrides = _parse_overrides(exp.get("eval_overrides"))

        for seed in seeds:
            run_tag = f"{model_name}_seed{seed}"
            train_dir = output_root / "runs" / run_tag
            eval_report = output_root / "eval" / f"{run_tag}_{eval_split}.json"
            logs_dir = output_root / "logs" / run_tag
            train_log = logs_dir / "train.log"
            eval_log = logs_dir / "eval.log"

            status = "ok"
            status_message = ""
            model_path = None
            if not skip_train:
                preflight_ok, preflight = _pretrained_preflight(
                    train_config=train_config,
                    train_overrides=[f"seed={seed}", *train_overrides],
                    repo_root=repo_root,
                    validation_cache=preflight_cache,
                )
                if not preflight_ok:
                    status = "pretrained_missing"
                    status_message = str(preflight.get("reason", "pretrained artifacts are unavailable"))
                    failures += 1
                    _write_skip_log(
                        log_path=train_log,
                        run_tag=run_tag,
                        reason=status_message,
                        actions=[str(x) for x in preflight.get("actions", []) if str(x).strip()],
                        details=preflight,
                    )
                else:
                    preflight = {}
                train_cmd = [
                    python_exe,
                    str(cli),
                    "train",
                    "--config",
                    train_config,
                    "--dataset-dir",
                    dataset_dir,
                    "--output-dir",
                    str(train_dir),
                    "--set",
                    f"seed={seed}",
                    "--no-auto-prepare-dataset",
                ]
                for item in train_overrides:
                    train_cmd.extend(["--set", item])
                if status == "ok":
                    rc = _run_cmd(
                        train_cmd,
                        train_log,
                        dry_run=dry_run,
                        run_label=f"{run_tag}:train",
                        idle_timeout_seconds=idle_timeout_seconds,
                        wall_timeout_seconds=wall_timeout_seconds,
                        terminate_grace_seconds=terminate_grace_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                    )
                    if rc != 0:
                        status = f"train_failed({rc})"
                        if rc == 124:
                            status_message = (
                                "training command timed out by suite watchdog; "
                                "see train.log for timeout reason and partial output"
                            )
                        else:
                            status_message = f"training command failed with exit code {rc}"
                        failures += 1
            if status == "ok" and not skip_eval:
                try:
                    model_path = _resolve_model_path(train_dir)
                except Exception:
                    if dry_run:
                        model_path = train_dir / "best_checkpoint.pt"
                    else:
                        status = "model_missing"
                        status_message = "training did not produce a model checkpoint"
                        failures += 1
                if status == "ok":
                    eval_cmd = [
                        python_exe,
                        str(cli),
                        "evaluate",
                        "--config",
                        eval_config,
                        "--dataset-dir",
                        dataset_dir,
                        "--model-path",
                        str(model_path),
                        "--split",
                        eval_split,
                        "--output-path",
                        str(eval_report),
                        "--no-auto-prepare-dataset",
                    ]
                    for item in eval_overrides:
                        eval_cmd.extend(["--set", item])
                    rc = _run_cmd(
                        eval_cmd,
                        eval_log,
                        dry_run=dry_run,
                        run_label=f"{run_tag}:eval",
                        idle_timeout_seconds=idle_timeout_seconds,
                        wall_timeout_seconds=wall_timeout_seconds,
                        terminate_grace_seconds=terminate_grace_seconds,
                        poll_interval_seconds=poll_interval_seconds,
                    )
                    if rc != 0:
                        status = f"eval_failed({rc})"
                        if rc == 124:
                            status_message = (
                                "evaluation command timed out by suite watchdog; "
                                "see eval.log for timeout reason and partial output"
                            )
                        else:
                            status_message = f"evaluation command failed with exit code {rc}"
                        failures += 1

            eval_payload = _read_json(eval_report)
            metrics = eval_payload.get("metrics", {}) if isinstance(eval_payload, dict) else {}
            scientific_metrics = eval_payload.get("scientific_metrics", {}) if isinstance(eval_payload, dict) else {}
            train_resolved = _read_json(train_dir / "resolved_config.json")
            train_meta = _read_training_metadata(train_dir, run_tag, output_root)
            checkpoint_stats = _checkpoint_state_stats(model_path) if isinstance(model_path, Path) else {}
            model_param_count = _safe_int(train_meta.get("model_parameter_count_report"))
            if model_param_count is None:
                model_param_count = _safe_int(checkpoint_stats.get("parameter_count"))
            model_trainable_param_count = _safe_int(train_meta.get("model_trainable_parameter_count"))
            model_artifact_bytes = model_path.stat().st_size if isinstance(model_path, Path) and model_path.exists() else None
            model_weight_mean = _safe_float(train_meta.get("model_weight_mean"))
            model_weight_std = _safe_float(train_meta.get("model_weight_std"))
            model_weight_min = _safe_float(train_meta.get("model_weight_min"))
            model_weight_max = _safe_float(train_meta.get("model_weight_max"))
            if model_weight_mean is None:
                model_weight_mean = _safe_float(checkpoint_stats.get("weight_mean"))
            if model_weight_std is None:
                model_weight_std = _safe_float(checkpoint_stats.get("weight_std"))
            if model_weight_min is None:
                model_weight_min = _safe_float(checkpoint_stats.get("weight_min"))
            if model_weight_max is None:
                model_weight_max = _safe_float(checkpoint_stats.get("weight_max"))
            eval_runtime_seconds = eval_payload.get("runtime_seconds") if isinstance(eval_payload, dict) else None
            train_runtime_seconds = train_meta.get("training_runtime_seconds")
            total_runtime_seconds = None
            if _safe_float(train_runtime_seconds) is not None or _safe_float(eval_runtime_seconds) is not None:
                total_runtime_seconds = float(_safe_float(train_runtime_seconds) or 0.0) + float(
                    _safe_float(eval_runtime_seconds) or 0.0
                )

            row = {
                "model": model_name,
                "seed": seed,
                "status": status,
                "status_message": status_message,
                "dataset_dir": dataset_dir,
                "train_config": train_config,
                "eval_config": eval_config,
                "train_dir": str(train_dir),
                "eval_report": str(eval_report),
                "pixel_accuracy": metrics.get("pixel_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "mean_iou": metrics.get("mean_iou"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "weighted_f1": metrics.get("weighted_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "frequency_weighted_iou": metrics.get("frequency_weighted_iou"),
                "foreground_precision": metrics.get("foreground_precision"),
                "foreground_recall": metrics.get("foreground_recall"),
                "foreground_specificity": metrics.get("foreground_specificity"),
                "foreground_iou": metrics.get("foreground_iou"),
                "foreground_dice": metrics.get("foreground_dice"),
                "false_positive_rate": metrics.get("false_positive_rate"),
                "false_negative_rate": metrics.get("false_negative_rate"),
                "matthews_corrcoef": metrics.get("matthews_corrcoef"),
                "runtime_seconds": eval_runtime_seconds,
                "training_runtime_seconds": train_runtime_seconds,
                "total_runtime_seconds": total_runtime_seconds,
                "mask_area_fraction_abs_error": scientific_metrics.get("mask_area_fraction_abs_error"),
                "hydride_count_abs_error": scientific_metrics.get("hydride_count_abs_error"),
                "hydride_size_wasserstein": scientific_metrics.get("hydride_size_wasserstein"),
                "hydride_orientation_wasserstein": scientific_metrics.get("hydride_orientation_wasserstein"),
                "backend": eval_payload.get("backend") if isinstance(eval_payload, dict) else "",
                "model_initialization": eval_payload.get("model_initialization") if isinstance(eval_payload, dict) else "",
                "model_artifact_path": str(model_path) if isinstance(model_path, Path) else "",
                "model_artifact_size_bytes": model_artifact_bytes,
                "model_artifact_size_mb": _bytes_to_mb(model_artifact_bytes),
                "model_parameter_count": model_param_count,
                "model_trainable_parameter_count": model_trainable_param_count,
                "model_weight_mean": model_weight_mean,
                "model_weight_std": model_weight_std,
                "model_weight_min": model_weight_min,
                "model_weight_max": model_weight_max,
                "resolved_backend": train_resolved.get("backend") if isinstance(train_resolved, dict) else "",
                "resolved_model_architecture": train_resolved.get("model_architecture")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_epochs": train_resolved.get("epochs") if isinstance(train_resolved, dict) else "",
                "resolved_batch_size": train_resolved.get("batch_size") if isinstance(train_resolved, dict) else "",
                "resolved_learning_rate": train_resolved.get("learning_rate") if isinstance(train_resolved, dict) else "",
                "resolved_weight_decay": train_resolved.get("weight_decay") if isinstance(train_resolved, dict) else "",
                "resolved_model_base_channels": train_resolved.get("model_base_channels")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_depth": train_resolved.get("transformer_depth")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_num_heads": train_resolved.get("transformer_num_heads")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_mlp_ratio": train_resolved.get("transformer_mlp_ratio")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_transformer_dropout": train_resolved.get("transformer_dropout")
                if isinstance(train_resolved, dict)
                else "",
                "resolved_segformer_patch_size": train_resolved.get("segformer_patch_size")
                if isinstance(train_resolved, dict)
                else "",
                "training_status": train_meta.get("training_status"),
                "training_runtime_human": train_meta.get("training_runtime_human"),
                "runtime_device": train_meta.get("runtime_device"),
                "runtime_device_reason": train_meta.get("runtime_device_reason"),
                "runtime_gpu_name": train_meta.get("runtime_gpu_name"),
                "runtime_gpu_peak_memory_allocated_mb": train_meta.get("runtime_gpu_peak_memory_allocated_mb"),
                "runtime_gpu_total_memory_mb": train_meta.get("runtime_gpu_total_memory_mb"),
                "runtime_gpu_utilization_pct": train_meta.get("runtime_gpu_utilization_pct"),
                "training_epochs_total": train_meta.get("training_epochs_total"),
                "training_epochs_completed": train_meta.get("training_epochs_completed"),
                "training_history_points": train_meta.get("training_history_points"),
                "best_val_loss_train": train_meta.get("best_val_loss_train"),
                "last_train_loss": train_meta.get("last_train_loss"),
                "last_val_loss": train_meta.get("last_val_loss"),
                "last_train_accuracy": train_meta.get("last_train_accuracy"),
                "last_val_accuracy": train_meta.get("last_val_accuracy"),
                "last_train_iou": train_meta.get("last_train_iou"),
                "last_val_iou": train_meta.get("last_val_iou"),
                "avg_epoch_runtime_seconds": train_meta.get("avg_epoch_runtime_seconds"),
                "loss_curve_png": train_meta.get("loss_curve_png"),
                "accuracy_curve_png": train_meta.get("accuracy_curve_png"),
                "iou_curve_png": train_meta.get("iou_curve_png"),
                "tracked_samples_count": train_meta.get("tracked_samples_count"),
                "tracked_samples_mean_iou": train_meta.get("tracked_samples_mean_iou"),
                "tracked_samples_min_iou": train_meta.get("tracked_samples_min_iou"),
                "tracked_samples_max_iou": train_meta.get("tracked_samples_max_iou"),
                "tracked_samples": train_meta.get("tracked_samples"),
                "tracked_sample_evolution_count": train_meta.get("tracked_sample_evolution_count"),
                "tracked_sample_evolution": train_meta.get("tracked_sample_evolution"),
                "best_tracked_sample_delta_iou": train_meta.get("best_tracked_sample_delta_iou"),
                "worst_tracked_sample_delta_iou": train_meta.get("worst_tracked_sample_delta_iou"),
                "compute_train_samples_processed": train_meta.get("compute_train_samples_processed"),
                "compute_val_samples_processed": train_meta.get("compute_val_samples_processed"),
                "compute_tracking_samples_processed": train_meta.get("compute_tracking_samples_processed"),
                "compute_estimated_forward_flops_per_sample": train_meta.get("compute_estimated_forward_flops_per_sample"),
                "compute_estimated_total_flops": train_meta.get("compute_estimated_total_flops"),
                "compute_estimated_total_tflops": train_meta.get("compute_estimated_total_tflops"),
                "compute_flops_estimate_method": train_meta.get("compute_flops_estimate_method"),
                "train_overrides": "|".join(train_overrides),
                "eval_overrides": "|".join(eval_overrides),
            }
            rows.append(row)

    agg = _aggregate(rows)
    summary_json = output_root / "benchmark_summary.json"
    summary_csv = output_root / "benchmark_summary.csv"
    aggregate_csv = output_root / "benchmark_aggregate.csv"
    dashboard_html = output_root / "benchmark_dashboard.html"
    canonical_summary_json = output_root / "summary.json"
    canonical_summary_html = output_root / "summary.html"

    summary_payload = {
        "schema_version": "microseg.hydride_benchmark_suite.v3",
        "config_path": str(cfg_path),
        "dataset_dir": dataset_dir,
        "output_root": str(output_root),
        "eval_split": eval_split,
        "benchmark_mode": benchmark_mode,
        "expected_dataset_manifest_sha256": expected_manifest_sha,
        "run_count": len(rows),
        "failure_count": failures,
        "rows": rows,
        "aggregate": agg,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_csv(
        summary_csv,
        rows,
        [
            "model",
            "seed",
            "status",
            "status_message",
            "backend",
            "model_initialization",
            "runtime_device",
            "runtime_gpu_name",
            "pixel_accuracy",
            "macro_f1",
            "mean_iou",
            "macro_precision",
            "macro_recall",
            "weighted_f1",
            "balanced_accuracy",
            "frequency_weighted_iou",
            "foreground_precision",
            "foreground_recall",
            "foreground_specificity",
            "foreground_iou",
            "foreground_dice",
            "false_positive_rate",
            "false_negative_rate",
            "matthews_corrcoef",
            "mask_area_fraction_abs_error",
            "hydride_count_abs_error",
            "hydride_size_wasserstein",
            "hydride_orientation_wasserstein",
            "runtime_seconds",
            "training_runtime_seconds",
            "total_runtime_seconds",
            "training_status",
            "training_runtime_human",
            "training_epochs_total",
            "training_epochs_completed",
            "training_history_points",
            "best_val_loss_train",
            "last_train_loss",
            "last_val_loss",
            "last_train_accuracy",
            "last_val_accuracy",
            "last_train_iou",
            "last_val_iou",
            "avg_epoch_runtime_seconds",
            "tracked_samples_count",
            "tracked_samples_mean_iou",
            "tracked_samples_min_iou",
            "tracked_samples_max_iou",
            "tracked_sample_evolution_count",
            "best_tracked_sample_delta_iou",
            "worst_tracked_sample_delta_iou",
            "model_artifact_path",
            "model_artifact_size_bytes",
            "model_artifact_size_mb",
            "model_parameter_count",
            "model_trainable_parameter_count",
            "model_weight_mean",
            "model_weight_std",
            "model_weight_min",
            "model_weight_max",
            "compute_estimated_total_flops",
            "compute_estimated_total_tflops",
            "runtime_gpu_peak_memory_allocated_mb",
            "loss_curve_png",
            "accuracy_curve_png",
            "iou_curve_png",
            "train_config",
            "eval_config",
            "train_dir",
            "eval_report",
            "resolved_backend",
            "resolved_model_architecture",
            "resolved_epochs",
            "resolved_batch_size",
            "resolved_learning_rate",
            "resolved_weight_decay",
            "resolved_model_base_channels",
            "resolved_transformer_depth",
            "resolved_transformer_num_heads",
            "resolved_transformer_mlp_ratio",
            "resolved_transformer_dropout",
            "resolved_segformer_patch_size",
            "train_overrides",
            "eval_overrides",
        ],
    )
    _write_csv(
        aggregate_csv,
        agg,
        [
            "model",
            "runs",
            "ok_runs",
            "failed_runs",
            "mean_pixel_accuracy",
            "std_pixel_accuracy",
            "mean_macro_f1",
            "std_macro_f1",
            "mean_mean_iou",
            "std_mean_iou",
            "mean_macro_precision",
            "std_macro_precision",
            "mean_macro_recall",
            "std_macro_recall",
            "mean_weighted_f1",
            "std_weighted_f1",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "mean_frequency_weighted_iou",
            "std_frequency_weighted_iou",
            "mean_foreground_dice",
            "std_foreground_dice",
            "mean_foreground_iou",
            "std_foreground_iou",
            "mean_foreground_precision",
            "mean_foreground_recall",
            "mean_foreground_specificity",
            "mean_false_positive_rate",
            "mean_false_negative_rate",
            "mean_matthews_corrcoef",
            "mean_eval_runtime_seconds",
            "std_eval_runtime_seconds",
            "mean_training_runtime_seconds",
            "std_training_runtime_seconds",
            "mean_total_runtime_seconds",
            "std_total_runtime_seconds",
            "mean_model_parameter_count",
            "mean_model_trainable_parameter_count",
            "mean_model_artifact_size_mb",
            "mean_model_weight_mean",
            "mean_model_weight_std",
            "mean_model_weight_min",
            "mean_model_weight_max",
            "mean_compute_estimated_total_tflops",
            "mean_runtime_gpu_peak_memory_allocated_mb",
            "mean_mask_area_fraction_abs_error",
            "mean_hydride_count_abs_error",
            "mean_hydride_size_wasserstein",
            "mean_hydride_orientation_wasserstein",
            "mean_last_train_loss",
            "mean_last_val_loss",
            "mean_last_train_accuracy",
            "mean_last_val_accuracy",
            "mean_last_train_iou",
            "mean_last_val_iou",
            "mean_tracked_samples_iou",
            "mean_tracked_iou_span",
            "mean_best_tracked_sample_delta_iou",
            "mean_worst_tracked_sample_delta_iou",
            "mean_overfit_iou_gap",
            "quality_score",
            "efficiency_score",
            "robustness_score",
            "rank_quality",
            "rank_efficiency",
            "rank_runtime",
            "rank_robustness",
        ],
    )
    _write_dashboard(dashboard_html, rows, agg)
    canonical_summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_dashboard(canonical_summary_html, rows, agg)

    print(f"suite summary json: {summary_json}")
    print(f"suite canonical summary json: {canonical_summary_json}")
    print(f"suite summary csv: {summary_csv}")
    print(f"suite aggregate csv: {aggregate_csv}")
    print(f"suite dashboard html: {dashboard_html}")
    print(f"suite canonical summary html: {canonical_summary_html}")
    print(f"runs: {len(rows)} failures: {failures}")

    if strict and failures > 0:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydride benchmark suite runner and summarizer")
    parser.add_argument("--config", required=True, help="Path to benchmark suite YAML")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands but do not execute train/eval")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any run fails")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage")
    args = parser.parse_args()
    rc = run_suite(
        Path(args.config),
        dry_run=bool(args.dry_run),
        strict=bool(args.strict),
        skip_train=bool(args.skip_train),
        skip_eval=bool(args.skip_eval),
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
