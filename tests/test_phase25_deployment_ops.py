"""Phase 25 tests for deployment/preflight/promotion/support operations."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.deployment import (
    DeploymentPackageConfig,
    create_deployment_package,
    validate_deployment_package,
)
from src.microseg.quality import (
    PreflightConfig,
    PromotionPolicy,
    SupportBundleConfig,
    create_support_bundle,
    evaluate_and_promote_model,
    run_preflight,
)


def _registry_model_row(model_id: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_nickname": model_id,
        "model_type": "binary_segmentation",
        "framework": "pytorch",
        "input_size": "512x512",
        "input_dimensions": "H x W x 3",
        "checkpoint_path_hint": "frozen_checkpoints/candidates/model.pth",
        "application_remarks": "unit-test",
        "classes": [
            {"index": 0, "name": "background", "color_hex": "#000000"},
            {"index": 1, "name": "foreground", "color_hex": "#FFFFFF"},
        ],
    }


def test_phase25_create_and_validate_deployment_package(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"unit-test-model")
    resolved_cfg = tmp_path / "resolved_config.json"
    resolved_cfg.write_text(
        json.dumps({"model_architecture": "unet_binary", "input_hw": [512, 512]}, indent=2),
        encoding="utf-8",
    )

    result = create_deployment_package(
        DeploymentPackageConfig(
            model_path=str(model_path),
            output_dir=str(tmp_path / "deployments"),
            package_name="unit_pkg",
            resolved_config_path=str(resolved_cfg),
        )
    )

    manifest_path = Path(result.manifest_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == "microseg.deployment_package.v1"
    assert payload.get("file_count", 0) >= 2

    report = validate_deployment_package(result.package_dir, verify_sha256=True)
    assert report.ok is True
    assert report.file_count >= 2


def test_phase25_preflight_deploy_mode_reports_invalid_package(tmp_path: Path) -> None:
    bad_pkg = tmp_path / "broken_package"
    bad_pkg.mkdir(parents=True, exist_ok=True)

    report = run_preflight(
        PreflightConfig(
            mode="deploy",
            deployment_package_dir=str(bad_pkg),
        )
    )

    assert report.ok is False
    assert any(issue.code == "deploy.package_invalid" for issue in report.issues)


def test_phase25_promotion_gate_updates_registry_with_relative_paths(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": [
                    {
                        "model": "candidate_model",
                        "runs": 3,
                        "ok_runs": 3,
                        "failed_runs": 0,
                        "mean_mean_iou": 0.72,
                        "mean_macro_f1": 0.82,
                        "mean_foreground_dice": 0.79,
                        "mean_cohen_kappa": 0.61,
                        "mean_false_positive_rate": 0.10,
                        "mean_false_negative_rate": 0.12,
                        "mean_total_runtime_seconds": 15.0,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "frozen_checkpoints" / "model_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "microseg.frozen_checkpoint_registry.v1",
                "models": [_registry_model_row("candidate_model")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    policy = PromotionPolicy(
        min_runs=1,
        min_ok_runs=1,
        min_mean_iou=0.60,
        min_macro_f1=0.70,
        min_foreground_dice=0.70,
        min_cohen_kappa=0.50,
        max_mean_false_positive_rate=0.20,
        max_mean_false_negative_rate=0.20,
        require_no_failed_runs=True,
    )

    decision = evaluate_and_promote_model(
        summary_json_path=summary_path,
        model_name="candidate_model",
        registry_model_id="candidate_model",
        target_stage="candidate",
        policy=policy,
        registry_path=registry_path,
        update_registry=True,
    )

    assert decision.passed is True
    assert decision.registry_updated is True

    updated = json.loads(registry_path.read_text(encoding="utf-8"))
    model = updated["models"][0]
    assert model["artifact_stage"] == "candidate"
    assert not Path(str(model["source_run_manifest"])).is_absolute()
    assert not Path(str(model["quality_report_path"])).is_absolute()


def test_phase25_support_bundle_collects_run_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "suite_run"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "train.log").write_text("train log", encoding="utf-8")
    (run_root / "summary.html").write_text("<html></html>", encoding="utf-8")
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "train_log": "logs/train.log",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = create_support_bundle(
        SupportBundleConfig(
            run_root=str(run_root),
            output_dir=str(tmp_path / "support"),
            bundle_name="phase25",
        )
    )

    assert Path(result.bundle_dir).exists()
    assert Path(result.zip_path).exists()
    assert Path(result.manifest_path).exists()
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    included = manifest.get("included", [])
    assert included
    assert any(str(item.get("source", "")).endswith("logs/train.log") for item in included)
    assert (Path(result.bundle_dir) / "environment_fingerprint.json").exists()
