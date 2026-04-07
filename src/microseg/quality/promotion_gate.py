"""Benchmark-driven model promotion gate and registry update helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _to_repo_relative(path_value: str, *, registry_path: Path) -> str:
    p = Path(path_value).resolve()
    repo_root = registry_path.resolve().parents[1]
    try:
        return p.relative_to(repo_root).as_posix()
    except Exception:
        return str(p)


@dataclass(frozen=True)
class PromotionPolicy:
    """Thresholds used for objective model promotion decisions."""

    min_runs: int = 1
    min_ok_runs: int = 1
    min_mean_iou: float = 0.0
    min_macro_f1: float = 0.0
    min_foreground_dice: float = 0.0
    min_cohen_kappa: float = -1.0
    max_mean_false_positive_rate: float = 1.0
    max_mean_false_negative_rate: float = 1.0
    max_mean_total_runtime_seconds: float = 0.0
    require_no_failed_runs: bool = False


@dataclass
class PromotionDecision:
    """Outcome from applying a promotion gate to one benchmarked model."""

    schema_version: str
    created_utc: str
    summary_path: str
    model_name: str
    registry_model_id: str
    target_stage: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    registry_updated: bool = False
    registry_path: str = ""


def load_promotion_policy(path: str | Path | None = None) -> PromotionPolicy:
    """Load promotion policy from YAML file or use defaults."""

    if path is None or not str(path).strip():
        return PromotionPolicy()
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"promotion policy must be mapping: {p}")
    return PromotionPolicy(
        min_runs=int(payload.get("min_runs", 1)),
        min_ok_runs=int(payload.get("min_ok_runs", 1)),
        min_mean_iou=float(payload.get("min_mean_iou", 0.0)),
        min_macro_f1=float(payload.get("min_macro_f1", 0.0)),
        min_foreground_dice=float(payload.get("min_foreground_dice", 0.0)),
        min_cohen_kappa=float(payload.get("min_cohen_kappa", -1.0)),
        max_mean_false_positive_rate=float(payload.get("max_mean_false_positive_rate", 1.0)),
        max_mean_false_negative_rate=float(payload.get("max_mean_false_negative_rate", 1.0)),
        max_mean_total_runtime_seconds=float(payload.get("max_mean_total_runtime_seconds", 0.0)),
        require_no_failed_runs=bool(payload.get("require_no_failed_runs", False)),
    )


def _apply_thresholds(observed: dict[str, Any], policy: PromotionPolicy) -> list[str]:
    reasons: list[str] = []

    def _lt(name: str, observed_value: float, threshold: float) -> None:
        if observed_value < threshold:
            reasons.append(f"{name} below threshold: observed={observed_value:.6f} required>={threshold:.6f}")

    def _gt(name: str, observed_value: float, threshold: float) -> None:
        if observed_value > threshold:
            reasons.append(f"{name} above threshold: observed={observed_value:.6f} required<={threshold:.6f}")

    runs = int(observed.get("runs", 0))
    ok_runs = int(observed.get("ok_runs", 0))
    failed_runs = int(observed.get("failed_runs", 0))
    if runs < int(policy.min_runs):
        reasons.append(f"insufficient runs: observed={runs} required>={int(policy.min_runs)}")
    if ok_runs < int(policy.min_ok_runs):
        reasons.append(f"insufficient ok runs: observed={ok_runs} required>={int(policy.min_ok_runs)}")
    if bool(policy.require_no_failed_runs) and failed_runs > 0:
        reasons.append(f"failed runs present: failed_runs={failed_runs}")

    _lt("mean_mean_iou", float(observed.get("mean_mean_iou", 0.0)), float(policy.min_mean_iou))
    _lt("mean_macro_f1", float(observed.get("mean_macro_f1", 0.0)), float(policy.min_macro_f1))
    _lt("mean_foreground_dice", float(observed.get("mean_foreground_dice", 0.0)), float(policy.min_foreground_dice))
    _lt("mean_cohen_kappa", float(observed.get("mean_cohen_kappa", 0.0)), float(policy.min_cohen_kappa))
    _gt(
        "mean_false_positive_rate",
        float(observed.get("mean_false_positive_rate", 0.0)),
        float(policy.max_mean_false_positive_rate),
    )
    _gt(
        "mean_false_negative_rate",
        float(observed.get("mean_false_negative_rate", 0.0)),
        float(policy.max_mean_false_negative_rate),
    )
    max_runtime = float(policy.max_mean_total_runtime_seconds)
    if max_runtime > 0.0:
        _gt(
            "mean_total_runtime_seconds",
            float(observed.get("mean_total_runtime_seconds", 0.0)),
            max_runtime,
        )
    return reasons


def _update_frozen_registry(
    *,
    registry_path: Path,
    registry_model_id: str,
    target_stage: str,
    source_run_manifest: str,
    quality_report_path: str,
    model_nickname: str,
    create_if_missing: bool,
) -> bool:
    if registry_path.exists():
        payload = _read_json(registry_path)
    else:
        if not create_if_missing:
            raise FileNotFoundError(f"registry path does not exist: {registry_path}")
        payload = {
            "schema_version": "microseg.frozen_checkpoint_registry.v1",
            "models": [],
        }
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"registry models must be list: {registry_path}")
    found = None
    for item in models:
        if not isinstance(item, dict):
            continue
        if str(item.get("model_id", "")).strip() == registry_model_id:
            found = item
            break
    if found is None:
        if not create_if_missing:
            raise KeyError(f"model_id not found in registry: {registry_model_id}")
        found = {
            "model_id": registry_model_id,
            "model_nickname": model_nickname,
            "model_type": "binary_segmentation",
            "framework": "pytorch",
            "input_size": "variable",
            "input_dimensions": "H x W x 3",
            "checkpoint_path_hint": "n/a",
            "application_remarks": "Auto-created by promotion gate.",
            "short_description": "Auto-created placeholder; update metadata before production.",
            "detailed_description": "Created from benchmark promotion workflow.",
            "classes": [
                {"index": 0, "name": "background", "color_hex": "#000000"},
                {"index": 1, "name": "foreground", "color_hex": "#FFFFFF"},
            ],
        }
        models.append(found)

    found["artifact_stage"] = str(target_stage)
    found["source_run_manifest"] = _to_repo_relative(str(source_run_manifest), registry_path=registry_path)
    found["quality_report_path"] = _to_repo_relative(str(quality_report_path), registry_path=registry_path)
    payload["models"] = models
    payload["updated_utc"] = _utc_now()
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def evaluate_and_promote_model(
    *,
    summary_json_path: str | Path,
    model_name: str,
    registry_model_id: str,
    target_stage: str,
    policy: PromotionPolicy,
    registry_path: str | Path,
    update_registry: bool,
    create_if_missing: bool = False,
) -> PromotionDecision:
    """Evaluate a benchmarked model against policy and optionally update registry stage."""

    summary_path = Path(summary_json_path).resolve()
    payload = _read_json(summary_path)
    aggregate = payload.get("aggregate", [])
    if not isinstance(aggregate, list):
        raise ValueError(f"summary aggregate missing or invalid: {summary_path}")
    observed = None
    for row in aggregate:
        if not isinstance(row, dict):
            continue
        if str(row.get("model", "")).strip() == str(model_name).strip():
            observed = row
            break
    if observed is None:
        raise KeyError(f"model not found in summary aggregate: {model_name}")

    reasons = _apply_thresholds(dict(observed), policy)
    passed = len(reasons) == 0

    decision = PromotionDecision(
        schema_version="microseg.promotion_decision.v1",
        created_utc=_utc_now(),
        summary_path=str(summary_path),
        model_name=str(model_name),
        registry_model_id=str(registry_model_id),
        target_stage=str(target_stage),
        passed=passed,
        reasons=reasons,
        observed=dict(observed),
        policy=asdict(policy),
        registry_updated=False,
        registry_path=str(Path(registry_path).resolve()),
    )

    if passed and bool(update_registry):
        reg_path = Path(registry_path).resolve()
        summary_html = summary_path.with_suffix(".html")
        _update_frozen_registry(
            registry_path=reg_path,
            registry_model_id=str(registry_model_id),
            target_stage=str(target_stage),
            source_run_manifest=str(summary_path),
            quality_report_path=str(summary_html if summary_html.exists() else summary_path),
            model_nickname=str(model_name),
            create_if_missing=bool(create_if_missing),
        )
        decision.registry_updated = True
    return decision


def write_promotion_decision(decision: PromotionDecision, *, output_path: str | Path) -> Path:
    """Write promotion decision JSON and companion markdown summary."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        "# Model Promotion Decision",
        "",
        f"- created_utc: `{decision.created_utc}`",
        f"- model_name: `{decision.model_name}`",
        f"- registry_model_id: `{decision.registry_model_id}`",
        f"- target_stage: `{decision.target_stage}`",
        f"- passed: `{decision.passed}`",
        f"- registry_updated: `{decision.registry_updated}`",
        f"- summary_path: `{decision.summary_path}`",
        "",
        "## Reasons",
    ]
    if decision.reasons:
        lines.extend([f"- {reason}" for reason in decision.reasons])
    else:
        lines.append("- all thresholds satisfied")
    lines.extend(
        [
            "",
            "## Observed",
            "",
            "```json",
            json.dumps(decision.observed, indent=2),
            "```",
            "",
            "## Policy",
            "",
            "```json",
            json.dumps(decision.policy, indent=2),
            "```",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    return out
