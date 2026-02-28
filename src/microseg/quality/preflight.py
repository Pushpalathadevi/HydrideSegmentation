"""Unified preflight checks for train/eval/benchmark/deploy workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from src.microseg.dataops import DatasetPrepareConfig, preview_training_dataset_layout, run_dataset_quality_checks, DatasetQualityConfig
from src.microseg.io import resolve_config
from src.microseg.plugins import (
    resolve_bundle_paths,
    resolve_pretrained_record,
    validate_pretrained_registry,
)
from src.microseg.quality.failure_codes import (
    DEPLOY_PACKAGE_INVALID,
    PREFLIGHT_BENCHMARK_CONFIG_INVALID,
    PREFLIGHT_DATASET_INVALID,
    PREFLIGHT_MODEL_MISSING,
    PREFLIGHT_PRETRAINED_MISSING,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PreflightIssue:
    """One structured preflight finding."""

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    error_code: str = ""
    hint: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Workflow preflight report."""

    schema_version: str
    created_utc: str
    mode: str
    ok: bool
    issues: list[PreflightIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass(frozen=True)
class PreflightConfig:
    """Configuration for generic workflow preflight runs."""

    mode: Literal["train", "eval", "benchmark", "deploy"]
    dataset_dir: str = ""
    model_path: str = ""
    train_config: str = ""
    train_overrides: tuple[str, ...] = ()
    eval_config: str = ""
    benchmark_config: str = ""
    deployment_package_dir: str = ""
    require_dataset_qa: bool = False
    dataset_qa_report_path: str = ""
    verify_pretrained_sha256: bool = True
    output_path: str = ""


def _resolve_path(path_value: str | Path, *, base: Path) -> Path:
    p = Path(str(path_value).strip())
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _append_issue(report: PreflightReport, issue: PreflightIssue) -> None:
    if not issue.error_code:
        issue.error_code = _map_preflight_failure_code(issue.code)
    report.issues.append(issue)
    if issue.severity == "error":
        report.error_count += 1
    elif issue.severity == "warning":
        report.warning_count += 1
    else:
        report.info_count += 1


def _write_report(report: PreflightReport, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _map_preflight_failure_code(code: str) -> str:
    text = str(code).strip()
    mapping = {
        "dataset.missing": PREFLIGHT_DATASET_INVALID,
        "dataset.not_found": PREFLIGHT_DATASET_INVALID,
        "dataset.layout_invalid": PREFLIGHT_DATASET_INVALID,
        "dataset.qa_error": PREFLIGHT_DATASET_INVALID,
        "dataset.qa_result": PREFLIGHT_DATASET_INVALID,
        "model.missing": PREFLIGHT_MODEL_MISSING,
        "model.not_found": PREFLIGHT_MODEL_MISSING,
        "pretrained.not_ready": PREFLIGHT_PRETRAINED_MISSING,
        "benchmark.config_missing": PREFLIGHT_BENCHMARK_CONFIG_INVALID,
        "benchmark.config_invalid": PREFLIGHT_BENCHMARK_CONFIG_INVALID,
        "benchmark.experiments_missing": PREFLIGHT_BENCHMARK_CONFIG_INVALID,
        "benchmark.pretrained_not_ready": PREFLIGHT_PRETRAINED_MISSING,
        "deploy.package_invalid": DEPLOY_PACKAGE_INVALID,
    }
    return mapping.get(text, "")


def preflight_pretrained_train_config(
    *,
    train_config: str,
    train_overrides: list[str],
    repo_root: Path,
    validation_cache: dict[tuple[str, bool], Any],
) -> tuple[bool, dict[str, Any]]:
    """Check local-pretrained readiness for one resolved train config.

    This helper is intentionally shared by CLI preflight and benchmark suite orchestration.
    """

    try:
        cfg_path = _resolve_path(train_config, base=repo_root)
        if not cfg_path.exists():
            raise FileNotFoundError(f"train config does not exist: {cfg_path}")
        cfg = resolve_config(str(cfg_path), list(train_overrides))
        if not isinstance(cfg, dict):
            raise ValueError(f"resolved train config must be mapping: {cfg_path}")
    except Exception as exc:
        return (
            False,
            {
                "required": False,
                "reason": f"failed to resolve train config: {exc}",
                "actions": ["fix train config path/syntax before running"],
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
        "on connected machine: python scripts/download_pretrained_weights.py --targets all --force",
        f"on target machine: microseg-cli validate-pretrained --registry-path {registry_cfg} --strict",
        "confirm pre_trained_weights/ copied under repo root",
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
            _bundle, weights_path, _metadata = resolve_bundle_paths(rec, registry_path=str(registry_path))
        except Exception as exc:
            return (
                False,
                {
                    "required": True,
                    "mode": mode,
                    "model_id": model_id,
                    "registry_path": str(registry_path),
                    "reason": f"failed to resolve pretrained artifacts: {exc}",
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
                    "bundle_dir": str(bundle_path),
                    "reason": "",
                },
            )
        return (
            False,
            {
                "required": True,
                "mode": mode,
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
                "set pretrained_model_id to an id present in pre_trained_weights/registry.json",
                "or set pretrained_bundle_dir to an existing bundle folder",
            ],
        },
    )


def _check_dataset_preview(dataset_dir: str, report: PreflightReport) -> None:
    if not str(dataset_dir).strip():
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="dataset.missing",
                message="dataset_dir is required for this preflight mode",
            ),
        )
        return
    root = Path(dataset_dir).resolve()
    if not root.exists():
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="dataset.not_found",
                message=f"dataset_dir does not exist: {root}",
            ),
        )
        return
    preview_cfg = DatasetPrepareConfig(dataset_dir=str(root), output_dir=str(root / "_preflight_preview"))
    try:
        preview = preview_training_dataset_layout(preview_cfg)
    except Exception as exc:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="dataset.layout_invalid",
                message=f"dataset preview failed: {exc}",
                hint="ensure split layout or unsplit source/masks layout is valid",
            ),
        )
        return
    _append_issue(
        report,
        PreflightIssue(
            severity="info",
            code="dataset.preview_ok",
            message=(
                f"dataset preview ok | source_layout={preview.source_layout} "
                f"total_pairs={preview.total_pairs} split_counts={preview.split_counts}"
            ),
            context={
                "dataset_dir": str(root),
                "source_layout": preview.source_layout,
                "total_pairs": preview.total_pairs,
                "split_counts": dict(preview.split_counts),
            },
        ),
    )


def _check_dataset_qa(dataset_dir: str, report: PreflightReport, *, qa_report_path: str = "") -> None:
    root = Path(dataset_dir).resolve()
    out_path = (
        Path(qa_report_path).resolve()
        if str(qa_report_path).strip()
        else (root / "preflight_dataset_qa_report.json")
    )
    try:
        qa = run_dataset_quality_checks(
            DatasetQualityConfig(
                dataset_dir=str(root),
                output_path=str(out_path),
                imbalance_ratio_warn=0.98,
                strict=False,
            )
        )
    except Exception as exc:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="dataset.qa_error",
                message=f"dataset QA preflight failed: {exc}",
                hint="ensure dataset has train/val/test split folders with image/mask pairs",
            ),
        )
        return
    severity = "info" if qa.ok else "error"
    _append_issue(
        report,
        PreflightIssue(
            severity=severity,
            code="dataset.qa_result",
            message=f"dataset QA {'passed' if qa.ok else 'failed'} | report={out_path}",
            context={"qa_ok": bool(qa.ok), "errors": len(qa.errors), "warnings": len(qa.warnings)},
        ),
    )


def _check_model_path(model_path: str, report: PreflightReport) -> None:
    if not str(model_path).strip():
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="model.missing",
                message="model path is required for this preflight mode",
            ),
        )
        return
    p = Path(model_path).resolve()
    if not p.exists():
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="model.not_found",
                message=f"model path does not exist: {p}",
            ),
        )
        return
    _append_issue(
        report,
        PreflightIssue(
            severity="info",
            code="model.found",
            message=f"model artifact present: {p}",
            context={"model_path": str(p), "is_file": p.is_file(), "is_dir": p.is_dir()},
        ),
    )


def _check_train_pretrained(
    *,
    train_config: str,
    train_overrides: tuple[str, ...],
    repo_root: Path,
    report: PreflightReport,
) -> None:
    if not str(train_config).strip():
        _append_issue(
            report,
            PreflightIssue(
                severity="warning",
                code="train_config.missing",
                message="train_config not provided; skipping pretrained preflight",
            ),
        )
        return
    cache: dict[tuple[str, bool], Any] = {}
    ok, details = preflight_pretrained_train_config(
        train_config=train_config,
        train_overrides=list(train_overrides),
        repo_root=repo_root,
        validation_cache=cache,
    )
    if ok:
        _append_issue(
            report,
            PreflightIssue(
                severity="info",
                code="pretrained.ready",
                message="pretrained preflight passed",
                context=details,
            ),
        )
    else:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="pretrained.not_ready",
                message=str(details.get("reason", "pretrained artifacts are unavailable")),
                hint="; ".join(str(x) for x in details.get("actions", []) if str(x).strip()),
                context=details,
            ),
        )


def _check_benchmark_config(cfg_path: Path, report: PreflightReport) -> None:
    if not cfg_path.exists():
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="benchmark.config_missing",
                message=f"benchmark config missing: {cfg_path}",
            ),
        )
        return
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("benchmark config must be a mapping")
    except Exception as exc:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="benchmark.config_invalid",
                message=f"failed to parse benchmark config: {exc}",
            ),
        )
        return

    dataset_dir = str(cfg.get("dataset_dir", "")).strip()
    _check_dataset_preview(dataset_dir, report)
    eval_config = str(cfg.get("eval_config", "")).strip()
    if eval_config:
        eval_path = _resolve_path(eval_config, base=cfg_path.parent)
        if not eval_path.exists():
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.eval_config_missing",
                    message=f"eval_config missing: {eval_path}",
                ),
            )
    experiments = cfg.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="benchmark.experiments_missing",
                message="benchmark config has no experiments",
            ),
        )
        return
    seeds_raw = cfg.get("seeds", [42])
    if isinstance(seeds_raw, list):
        seeds = [int(v) for v in seeds_raw]
    else:
        seeds = [int(seeds_raw)]
    repo_root = cfg_path.resolve().parents[1]
    cache: dict[tuple[str, bool], Any] = {}
    for exp in experiments:
        if not isinstance(exp, dict):
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.experiment_invalid",
                    message=f"invalid experiment entry (not mapping): {exp!r}",
                ),
            )
            continue
        name = str(exp.get("name", "")).strip() or "unnamed"
        train_cfg = str(exp.get("train_config", "")).strip()
        if not train_cfg:
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.train_config_missing",
                    message=f"experiment '{name}' missing train_config",
                ),
            )
            continue
        train_cfg_path = _resolve_path(train_cfg, base=cfg_path.parent)
        if not train_cfg_path.exists():
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.train_config_not_found",
                    message=f"experiment '{name}' train_config missing: {train_cfg_path}",
                ),
            )
            continue
        over_raw = exp.get("train_overrides", [])
        if isinstance(over_raw, list):
            over_list = [str(v).strip() for v in over_raw if str(v).strip()]
        else:
            text = str(over_raw).strip()
            over_list = [p.strip() for p in text.replace("|", ",").split(",") if p.strip()] if text else []
        for seed in seeds:
            ok, details = preflight_pretrained_train_config(
                train_config=str(train_cfg_path),
                train_overrides=[f"seed={seed}", *over_list],
                repo_root=repo_root,
                validation_cache=cache,
            )
            if ok:
                continue
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.pretrained_not_ready",
                    message=f"experiment={name} seed={seed}: {details.get('reason', 'pretrained not ready')}",
                    hint="; ".join(str(x) for x in details.get("actions", []) if str(x).strip()),
                    context={"experiment": name, "seed": seed, **details},
                ),
            )


def run_preflight(config: PreflightConfig) -> PreflightReport:
    """Run workflow preflight checks and optionally persist JSON report."""

    report = PreflightReport(
        schema_version="microseg.preflight_report.v1",
        created_utc=_utc_now(),
        mode=str(config.mode),
        ok=False,
    )
    repo_root = Path(__file__).resolve().parents[3]

    if config.mode == "train":
        _check_dataset_preview(config.dataset_dir, report)
        _check_train_pretrained(
            train_config=str(config.train_config),
            train_overrides=tuple(config.train_overrides),
            repo_root=repo_root,
            report=report,
        )
        if bool(config.require_dataset_qa):
            _check_dataset_qa(config.dataset_dir, report, qa_report_path=str(config.dataset_qa_report_path))
    elif config.mode == "eval":
        _check_dataset_preview(config.dataset_dir, report)
        _check_model_path(config.model_path, report)
    elif config.mode == "benchmark":
        if not str(config.benchmark_config).strip():
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="benchmark.config_required",
                    message="benchmark_config is required for benchmark preflight",
                ),
            )
        else:
            _check_benchmark_config(Path(config.benchmark_config).resolve(), report)
    elif config.mode == "deploy":
        from src.microseg.deployment import validate_deployment_package

        pkg = str(config.deployment_package_dir).strip()
        if not pkg:
            _append_issue(
                report,
                PreflightIssue(
                    severity="error",
                    code="deploy.package_required",
                    message="deployment_package_dir is required for deploy preflight",
                ),
            )
        else:
            validation = validate_deployment_package(pkg, verify_sha256=bool(config.verify_pretrained_sha256))
            if validation.ok:
                _append_issue(
                    report,
                    PreflightIssue(
                        severity="info",
                        code="deploy.package_valid",
                        message=f"deployment package valid: {pkg}",
                        context={"file_count": validation.file_count},
                    ),
                )
            else:
                _append_issue(
                    report,
                    PreflightIssue(
                        severity="error",
                        code="deploy.package_invalid",
                        message="deployment package validation failed",
                        hint="; ".join(validation.errors[:5]),
                        context={"errors": validation.errors, "warnings": validation.warnings},
                    ),
                )
    else:
        _append_issue(
            report,
            PreflightIssue(
                severity="error",
                code="mode.unsupported",
                message=f"unsupported preflight mode: {config.mode!r}",
            ),
        )

    report.ok = report.error_count == 0
    if str(config.output_path).strip():
        _write_report(report, str(config.output_path))
    return report
