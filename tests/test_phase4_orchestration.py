"""Phase 4 orchestration tests for training/evaluation command stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.app import OrchestrationCommandBuilder
from src.microseg.evaluation.pixel_model_eval import PixelEvaluationConfig, PixelModelEvaluator
from src.microseg.training import PixelClassifierTrainer, PixelTrainingConfig


def _write_pair(images_dir: Path, masks_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(images_dir / name)
    Image.fromarray(mask).save(masks_dir / name)


def _build_dataset(root: Path) -> Path:
    ds = root / "dataset"

    # Train split: bright -> class 1, dark -> class 0
    for i in range(3):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        msk = np.zeros((32, 32), dtype=np.uint8)
        img[:, :16] = 20
        img[:, 16:] = 230
        msk[:, 16:] = 1
        _write_pair(ds / "train" / "images", ds / "train" / "masks", f"train_{i}.png", img, msk)

    # Val split with same rule.
    for i in range(2):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        msk = np.zeros((32, 32), dtype=np.uint8)
        img[:16, :] = 220
        msk[:16, :] = 1
        _write_pair(ds / "val" / "images", ds / "val" / "masks", f"val_{i}.png", img, msk)

    return ds


def test_phase4_command_builder_constructs_expected_commands() -> None:
    builder = OrchestrationCommandBuilder.discover(start=Path(__file__))

    infer_cmd = builder.infer(config="configs/inference.default.yml", overrides=["a=1"], image="x.png")
    train_cmd = builder.train(config="configs/train.default.yml", dataset_dir="d", output_dir="o")
    eval_cmd = builder.evaluate(config="configs/evaluate.default.yml", model_path="m.joblib", dataset_dir="d")
    prep_cmd = builder.dataset_prepare(config="configs/dataset_prepare.default.yml", dataset_dir="d", output_dir="o")
    qa_cmd = builder.dataset_qa(config="configs/dataset_qa.default.yml", dataset_dir="d", strict=True)
    hpc_cmd = builder.hpc_ga_generate(config="configs/hpc_ga.default.yml", dataset_dir="d", output_dir="o")
    hpc_feedback_cmd = builder.hpc_ga_feedback_report(
        config="configs/hpc_ga.default.yml",
        feedback_sources="outputs/a,outputs/b",
        output_path="outputs/hpc_ga_feedback/report.json",
    )
    feedback_bundle_cmd = builder.feedback_bundle(
        config="configs/feedback_bundle.default.yml",
        feedback_root="outputs/feedback_records",
        output_dir="outputs/feedback_bundles",
        deployment_id="site_a",
    )
    feedback_ingest_cmd = builder.feedback_ingest(
        config="configs/feedback_ingest.default.yml",
        bundle_paths=["outputs/feedback_bundles/a.zip"],
        ingest_root="outputs/feedback_lake",
        output_path="outputs/feedback_ingest/report.json",
    )
    feedback_dataset_cmd = builder.feedback_build_dataset(
        config="configs/feedback_build_dataset.default.yml",
        feedback_root="outputs/feedback_lake",
        output_dir="outputs/feedback_training_dataset",
    )
    feedback_trigger_cmd = builder.feedback_train_trigger(
        config="configs/feedback_train_trigger.default.yml",
        feedback_root="outputs/feedback_lake",
        output_path="outputs/feedback_trigger/report.json",
        execute=True,
    )
    preflight_cmd = builder.preflight(config="configs/preflight.default.yml", mode="train", dataset_dir="d")
    deploy_package_cmd = builder.deploy_package(config="configs/deployment_package.default.yml", model_path="m.pth")
    deploy_validate_cmd = builder.deploy_validate(package_dir="outputs/deployments/p1", strict=True)
    deploy_smoke_cmd = builder.deploy_smoke(package_dir="outputs/deployments/p1", image_path="x.png")
    deploy_health_cmd = builder.deploy_health(package_dir="outputs/deployments/p1", image_dir="test_data")
    deploy_worker_cmd = builder.deploy_worker_run(package_dir="outputs/deployments/p1", image_dir="test_data")
    deploy_canary_cmd = builder.deploy_canary_shadow(
        baseline_package_dir="outputs/deployments/base",
        candidate_package_dir="outputs/deployments/cand",
        image_dir="test_data",
    )
    deploy_perf_cmd = builder.deploy_perf(package_dir="outputs/deployments/p1", image_dir="test_data")
    promote_cmd = builder.promote_model(
        summary_json="outputs/hydride_benchmark/summary.json",
        model_name="unet_binary",
    )
    support_cmd = builder.support_bundle(run_root="outputs/hydride_benchmark")
    compat_cmd = builder.compatibility_matrix(output_path="outputs/support_bundles/compat.json")

    assert infer_cmd[0].endswith("python") or "python" in infer_cmd[0]
    assert infer_cmd[2] == "infer"
    assert "--set" in infer_cmd
    assert train_cmd[2] == "train"
    assert eval_cmd[2] == "evaluate"
    assert prep_cmd[2] == "dataset-prepare"
    assert qa_cmd[2] == "dataset-qa"
    assert hpc_cmd[2] == "hpc-ga-generate"
    assert hpc_feedback_cmd[2] == "hpc-ga-feedback-report"
    assert feedback_bundle_cmd[2] == "feedback-bundle"
    assert feedback_ingest_cmd[2] == "feedback-ingest"
    assert feedback_dataset_cmd[2] == "feedback-build-dataset"
    assert feedback_trigger_cmd[2] == "feedback-train-trigger"
    assert preflight_cmd[2] == "preflight"
    assert deploy_package_cmd[2] == "deploy-package"
    assert deploy_validate_cmd[2] == "deploy-validate"
    assert deploy_smoke_cmd[2] == "deploy-smoke"
    assert deploy_health_cmd[2] == "deploy-health"
    assert deploy_worker_cmd[2] == "deploy-worker-run"
    assert deploy_canary_cmd[2] == "deploy-canary-shadow"
    assert deploy_perf_cmd[2] == "deploy-perf"
    assert promote_cmd[2] == "promote-model"
    assert support_cmd[2] == "support-bundle"
    assert compat_cmd[2] == "compatibility-matrix"
    assert qa_cmd[-1] == "--strict"
    assert deploy_validate_cmd[-1] == "--strict"
    assert "--feedback-sources" in hpc_feedback_cmd
    assert "--bundle-path" in feedback_ingest_cmd
    assert "--execute" in feedback_trigger_cmd


def test_phase4_train_and_evaluate_pixel_model(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    training_out = tmp_path / "training"

    trainer = PixelClassifierTrainer()
    trained = trainer.train(
        PixelTrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(training_out),
            train_split="train",
            max_samples=2000,
            max_iter=300,
            seed=7,
        )
    )

    model_path = Path(trained["model_path"])
    assert model_path.exists()
    assert (training_out / "training_manifest.json").exists()

    report_path = tmp_path / "eval" / "report.json"
    evaluator = PixelModelEvaluator()
    payload = evaluator.evaluate(
        PixelEvaluationConfig(
            dataset_dir=str(dataset_dir),
            model_path=str(model_path),
            split="val",
            output_path=str(report_path),
        )
    )

    assert report_path.exists()
    assert payload["metrics"]["pixel_accuracy"] > 0.7
    assert "macro_f1" in payload["metrics"]
    assert "macro_precision" in payload["metrics"]
    assert "macro_recall" in payload["metrics"]
    assert "weighted_f1" in payload["metrics"]
    assert "balanced_accuracy" in payload["metrics"]
    assert "cohen_kappa" in payload["metrics"]
    assert "frequency_weighted_iou" in payload["metrics"]
    assert "foreground_dice" in payload["metrics"]
    assert payload["metrics"]["foreground_dice"] >= 0.0
    assert "per_class_precision" in payload
    assert "per_class_recall" in payload
    assert "per_class_f1" in payload
    assert "confusion_matrix" in payload

    tracked = payload.get("tracked_samples", [])
    assert tracked
    sample = tracked[0]
    assert "macro_precision" in sample
    assert "macro_recall" in sample
    assert "weighted_f1" in sample
    assert "balanced_accuracy" in sample
    assert "cohen_kappa" in sample
    assert "frequency_weighted_iou" in sample
    assert "foreground_dice" in sample
    assert "mask_area_fraction_abs_error" in sample

    html_path = Path(str(payload.get("html_report_path", "")))
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Tracked Samples (Input | GT | Pred | Diff)" in html_text
    assert "Each sample panel includes per-image values for all available run metrics." in html_text
    assert "matthews_corrcoef" in html_text
    assert "cohen_kappa" in html_text

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert str(raw["schema_version"]).startswith("microseg.pixel_eval.v")
