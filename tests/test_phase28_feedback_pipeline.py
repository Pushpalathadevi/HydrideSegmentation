"""Phase 28 tests for feedback capture and active-learning pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.microseg.deployment import DeploymentPackageConfig, run_service_worker_batch, create_deployment_package
from src.microseg.feedback import (
    FeedbackArtifactWriter,
    FeedbackBundleConfig,
    FeedbackCaptureConfig,
    FeedbackDatasetBuildConfig,
    FeedbackIngestConfig,
    FeedbackTrainTriggerConfig,
    build_feedback_training_dataset,
    discover_feedback_record_dirs,
    evaluate_feedback_train_trigger,
    export_feedback_bundle,
    ingest_feedback_bundles,
    load_feedback_record,
)


def _rgb() -> np.ndarray:
    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[:, :12, :] = [20, 20, 20]
    arr[:, 12:, :] = [220, 220, 220]
    return arr


def _mask(value: int = 1) -> np.ndarray:
    m = np.zeros((24, 24), dtype=np.uint8)
    m[:, 12:] = value
    return m


def test_phase28_feedback_writer_lifecycle(tmp_path: Path) -> None:
    writer = FeedbackArtifactWriter(
        FeedbackCaptureConfig(
            feedback_root=str(tmp_path / "feedback"),
            deployment_id="site_a",
            operator_id="operator_01",
            source="desktop_gui",
        )
    )
    capture = writer.create_from_inference_arrays(
        run_id="r001",
        image_path=str(tmp_path / "img.png"),
        input_image_rgb=_rgb(),
        predicted_mask_indexed=_mask(1),
        predicted_overlay_rgb=_rgb(),
        model_id="hydride_conventional",
        model_name="Hydride Conventional",
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:02+00:00",
        inference_manifest={"pipeline": "test"},
        resolved_config={"a": 1},
        params={"enable_gpu": False, "device_policy": "cpu"},
    )
    rec_dir = Path(capture.record_dir)
    assert rec_dir.exists()
    assert (rec_dir / "feedback_record.json").exists()
    payload = load_feedback_record(rec_dir)
    assert payload["feedback"]["rating"] == "unrated"

    writer.update_feedback(rec_dir, rating="thumbs_down", comment="over-segmentation")
    updated = load_feedback_record(rec_dir)
    assert updated["feedback"]["rating"] == "thumbs_down"
    assert updated["feedback"]["comment"] == "over-segmentation"

    writer.attach_corrected_mask(rec_dir, _mask(1))
    with_corr = load_feedback_record(rec_dir)
    assert with_corr["correction"]["has_corrected_mask"] is True
    assert str(with_corr["correction"]["corrected_mask_path"]).endswith("corrected_mask_indexed.png")
    assert (rec_dir / "corrected_mask_indexed.png").exists()

    writer.link_correction_export(rec_dir, correction_record_path="/tmp/correction_record.json")
    linked = load_feedback_record(rec_dir)
    assert linked["correction"]["correction_record_path"] == "/tmp/correction_record.json"


def test_phase28_feedback_bundle_and_ingest_with_dedup(tmp_path: Path) -> None:
    root = tmp_path / "feedback"
    writer = FeedbackArtifactWriter(
        FeedbackCaptureConfig(
            feedback_root=str(root),
            deployment_id="site_bundle",
            operator_id="operator_bundle",
            source="cli_infer",
        )
    )
    captures = []
    for idx in range(2):
        cap = writer.create_from_inference_arrays(
            run_id=f"r{idx}",
            image_path=str(tmp_path / f"img_{idx}.png"),
            input_image_rgb=_rgb(),
            predicted_mask_indexed=_mask(1),
            predicted_overlay_rgb=_rgb(),
            model_id="hydride_conventional",
            model_name="Hydride Conventional",
            started_utc="2026-01-01T00:00:00+00:00",
            finished_utc="2026-01-01T00:00:01+00:00",
            inference_manifest={"pipeline": "test"},
            resolved_config={"idx": idx},
            params={},
        )
        captures.append(cap)
    writer.update_feedback(captures[0].record_dir, rating="thumbs_down", comment="bad sample")
    writer.update_feedback(captures[1].record_dir, rating="thumbs_up", comment="good sample")

    bundle_result = export_feedback_bundle(
        FeedbackBundleConfig(
            feedback_root=str(root),
            output_dir=str(tmp_path / "bundles"),
            deployment_id="site_bundle",
            max_records=200,
            state_path=str(tmp_path / "bundle_state.json"),
        )
    )
    assert Path(bundle_result.bundle_zip_path).exists()
    assert bundle_result.selected_records == 2

    ingest_report = ingest_feedback_bundles(
        FeedbackIngestConfig(
            bundle_paths=(str(bundle_result.bundle_zip_path),),
            ingest_root=str(tmp_path / "lake"),
            output_path=str(tmp_path / "ingest" / "report.json"),
            dedup_index_path=str(tmp_path / "ingest" / "index.json"),
            review_queue_path=str(tmp_path / "ingest" / "review.jsonl"),
        )
    )
    assert ingest_report.accepted_records == 2
    assert ingest_report.duplicate_records == 0
    assert ingest_report.review_queue_records == 1

    ingest_report_dup = ingest_feedback_bundles(
        FeedbackIngestConfig(
            bundle_paths=(str(bundle_result.bundle_zip_path),),
            ingest_root=str(tmp_path / "lake"),
            output_path=str(tmp_path / "ingest" / "report_dup.json"),
            dedup_index_path=str(tmp_path / "ingest" / "index.json"),
            review_queue_path=str(tmp_path / "ingest" / "review.jsonl"),
        )
    )
    assert ingest_report_dup.accepted_records == 0
    assert ingest_report_dup.duplicate_records == 2


def test_phase28_feedback_dataset_builder_policy(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    writer = FeedbackArtifactWriter(
        FeedbackCaptureConfig(
            feedback_root=str(root),
            deployment_id="site_ds",
            operator_id="operator_ds",
            source="desktop_gui",
        )
    )
    cap_corr = writer.create_from_inference_arrays(
        run_id="corr",
        image_path=str(tmp_path / "c.png"),
        input_image_rgb=_rgb(),
        predicted_mask_indexed=_mask(1),
        predicted_overlay_rgb=_rgb(),
        model_id="hydride_ml",
        model_name="Hydride ML",
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:01+00:00",
    )
    writer.update_feedback(cap_corr.record_dir, rating="thumbs_down", comment="fixed")
    writer.attach_corrected_mask(cap_corr.record_dir, _mask(1))

    cap_up = writer.create_from_inference_arrays(
        run_id="up",
        image_path=str(tmp_path / "u.png"),
        input_image_rgb=_rgb(),
        predicted_mask_indexed=_mask(1),
        predicted_overlay_rgb=_rgb(),
        model_id="hydride_ml",
        model_name="Hydride ML",
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:01+00:00",
    )
    writer.update_feedback(cap_up.record_dir, rating="thumbs_up", comment="ok")

    cap_down = writer.create_from_inference_arrays(
        run_id="down",
        image_path=str(tmp_path / "d.png"),
        input_image_rgb=_rgb(),
        predicted_mask_indexed=_mask(1),
        predicted_overlay_rgb=_rgb(),
        model_id="hydride_ml",
        model_name="Hydride ML",
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:01+00:00",
    )
    writer.update_feedback(cap_down.record_dir, rating="thumbs_down", comment="no correction")

    result = build_feedback_training_dataset(
        FeedbackDatasetBuildConfig(
            feedback_root=str(root),
            output_dir=str(tmp_path / "dataset"),
            seed=11,
            thumbs_up_weight=0.2,
            corrected_weight=1.0,
        )
    )
    assert result.included_samples == 2
    assert result.corrected_samples == 1
    assert result.pseudo_labeled_samples == 1
    assert result.excluded_downvote_without_correction == 1
    assert Path(result.manifest_path).exists()
    weights = (Path(result.output_dir) / "sample_weights.csv").read_text(encoding="utf-8")
    assert "pseudo_accepted,0.200000" in weights
    assert "human_corrected,1.000000" in weights


def test_phase28_feedback_trigger_report(tmp_path: Path) -> None:
    root = tmp_path / "lake_trigger"
    writer = FeedbackArtifactWriter(
        FeedbackCaptureConfig(
            feedback_root=str(root),
            deployment_id="site_trigger",
            operator_id="operator_trigger",
            source="desktop_gui",
        )
    )
    cap = writer.create_from_inference_arrays(
        run_id="trigger",
        image_path=str(tmp_path / "t.png"),
        input_image_rgb=_rgb(),
        predicted_mask_indexed=_mask(1),
        predicted_overlay_rgb=_rgb(),
        model_id="hydride_ml",
        model_name="Hydride ML",
        started_utc="2026-01-01T00:00:00+00:00",
        finished_utc="2026-01-01T00:00:01+00:00",
    )
    writer.update_feedback(cap.record_dir, rating="thumbs_down", comment="corrected")
    writer.attach_corrected_mask(cap.record_dir, _mask(1))

    report = evaluate_feedback_train_trigger(
        FeedbackTrainTriggerConfig(
            feedback_root=str(root),
            output_path=str(tmp_path / "trigger" / "report.json"),
            state_path=str(tmp_path / "trigger" / "state.json"),
            corrected_threshold=1,
            max_days_since_last_trigger=14,
            execute=False,
        )
    )
    assert report.should_trigger is True
    assert report.corrected_records_since_last_trigger >= 1
    assert report.commands
    assert Path(report.report_path).exists()
    assert Path(report.dataset_manifest_path).exists()


def test_phase28_cli_infer_creates_feedback_record(tmp_path: Path) -> None:
    img_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(img_path)
    out_dir = tmp_path / "infer_out"
    feedback_root = tmp_path / "feedback_cli"

    cmd = [
        sys.executable,
        "scripts/microseg_cli.py",
        "infer",
        "--image",
        str(img_path),
        "--model-name",
        "Hydride Conventional",
        "--output-dir",
        str(out_dir),
        "--set",
        "include_analysis=false",
        "--set",
        f"feedback_root={feedback_root}",
        "--set",
        "deployment_id=site_cli_test",
        "--set",
        "operator_id=operator_cli",
        "--set",
        "capture_feedback=true",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout
    record_dirs = discover_feedback_record_dirs(feedback_root)
    assert len(record_dirs) == 1
    payload = load_feedback_record(record_dirs[0])
    assert payload["source"] == "cli_infer"
    assert payload["deployment_id"] == "site_cli_test"


def test_phase28_service_worker_captures_feedback(monkeypatch, tmp_path: Path) -> None:
    import src.microseg.deployment.service_worker as svc_mod

    model = tmp_path / "toy.joblib"
    model.write_bytes(b"dummy")
    pkg = create_deployment_package(
        DeploymentPackageConfig(
            model_path=str(model),
            output_dir=str(tmp_path / "deploy"),
            package_name="svc",
        )
    )
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, 12:] = [255, 255, 255]
    image_path = tmp_path / "svc_input.png"
    Image.fromarray(image).save(image_path)

    def _predictor_factory(_artifact, *, enable_gpu, device_policy):  # noqa: ANN001
        def _predict(image_rgb):
            h, w, _ = image_rgb.shape
            out = np.zeros((h, w), dtype=np.uint8)
            out[:, w // 2 :] = 1
            return out

        return _predict

    monkeypatch.setattr(svc_mod, "build_predictor_from_artifact", _predictor_factory)
    result = run_service_worker_batch(
        svc_mod.ServiceWorkerConfig(
            package_dir=str(pkg.package_dir),
            output_dir=str(tmp_path / "svc_out"),
            capture_feedback=True,
            feedback_root=str(tmp_path / "svc_feedback"),
            deployment_id="svc_site",
            operator_id="svc_operator",
        ),
        image_paths=(str(image_path),),
        await_completion=True,
    )
    assert result.completed == 1
    job = result.jobs[0]
    assert str(job.feedback_record_dir).strip()
    assert Path(job.feedback_record_dir).exists()
    payload = load_feedback_record(job.feedback_record_dir)
    assert payload["source"] == "service_worker"
    assert payload["deployment_id"] == "svc_site"
    assert payload["operator_id"] == "svc_operator"
