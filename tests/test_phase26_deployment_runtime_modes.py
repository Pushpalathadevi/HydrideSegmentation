"""Phase 26 tests for deployment service-mode worker, canary-shadow, and perf harness."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.deployment import (
    CanaryShadowConfig,
    DeploymentPackageConfig,
    DeploymentPerfConfig,
    ServiceWorkerConfig,
    create_deployment_package,
    run_canary_shadow_compare,
    run_deployment_perf,
    run_service_worker_batch,
)
from src.microseg.quality import SERVICE_QUEUE_FULL


def _write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _make_package(tmp_path: Path, *, package_name: str) -> Path:
    model = tmp_path / f"{package_name}.joblib"
    model.write_bytes(b"dummy-model")
    result = create_deployment_package(
        DeploymentPackageConfig(
            model_path=str(model),
            output_dir=str(tmp_path / "deployments"),
            package_name=package_name,
        )
    )
    return Path(result.package_dir)


def test_phase26_service_worker_queue_capacity(monkeypatch, tmp_path: Path) -> None:
    import src.microseg.deployment.service_worker as svc_mod

    pkg_dir = _make_package(tmp_path, package_name="svc_q")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:, 0] = 255
    image_path = tmp_path / "image.png"
    _write_rgb(image_path, image)

    def _slow_predictor(_artifact, *, enable_gpu, device_policy):  # noqa: ANN001
        def _predict(image_rgb):
            time.sleep(0.2)
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        return _predict

    monkeypatch.setattr(svc_mod, "build_predictor_from_artifact", _slow_predictor)

    worker = svc_mod.DeploymentServiceWorker(
        ServiceWorkerConfig(
            package_dir=str(pkg_dir),
            output_dir=str(tmp_path / "svc_out"),
            max_workers=1,
            max_queue_size=1,
        )
    )
    try:
        first = worker.submit(str(image_path))
        second = worker.submit(str(image_path))
        assert first.status in {"queued", "running"}
        assert second.status == "rejected"
        assert second.error_code == SERVICE_QUEUE_FULL
    finally:
        worker.shutdown(wait=True)


def test_phase26_service_worker_batch_success(monkeypatch, tmp_path: Path) -> None:
    import src.microseg.deployment.service_worker as svc_mod

    pkg_dir = _make_package(tmp_path, package_name="svc_batch")
    image_dir = tmp_path / "images"
    for idx in range(2):
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[:, 8:, 0] = 255
        _write_rgb(image_dir / f"s{idx}.png", arr)

    def _predictor_factory(_artifact, *, enable_gpu, device_policy):  # noqa: ANN001
        def _predict(image_rgb):
            h, w, _ = image_rgb.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:, w // 2 :] = 1
            return mask

        return _predict

    monkeypatch.setattr(svc_mod, "build_predictor_from_artifact", _predictor_factory)

    result = run_service_worker_batch(
        ServiceWorkerConfig(
            package_dir=str(pkg_dir),
            output_dir=str(tmp_path / "svc_out"),
            max_workers=2,
            max_queue_size=8,
        ),
        image_dir=str(image_dir),
        await_completion=True,
    )

    assert result.total_submitted == 2
    assert result.accepted == 2
    assert result.rejected == 0
    assert result.completed == 2
    assert result.failed == 0
    assert Path(result.report_path).exists()


def test_phase26_canary_shadow_reports_positive_gain(monkeypatch, tmp_path: Path) -> None:
    import src.microseg.deployment.canary_shadow as cs_mod

    baseline_pkg = _make_package(tmp_path, package_name="baseline_pkg")
    candidate_pkg = _make_package(tmp_path, package_name="candidate_pkg")

    image_dir = tmp_path / "imgs"
    mask_dir = tmp_path / "masks"
    arr = np.zeros((12, 12, 3), dtype=np.uint8)
    _write_rgb(image_dir / "a.png", arr)
    mask_dir.mkdir(parents=True, exist_ok=True)
    gt = np.zeros((12, 12), dtype=np.uint8)
    gt[:, 6:] = 255
    Image.fromarray(gt).save(mask_dir / "a.png")

    def _predictor_factory(artifact, *, enable_gpu, device_policy):  # noqa: ANN001
        a = str(artifact)

        def _predict(image_rgb):
            h, w, _ = image_rgb.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            if "candidate_pkg" in a:
                mask[:, w // 2 :] = 1
            return mask

        return _predict

    monkeypatch.setattr(cs_mod, "build_predictor_from_artifact", _predictor_factory)

    result = run_canary_shadow_compare(
        CanaryShadowConfig(
            baseline_package_dir=str(baseline_pkg),
            candidate_package_dir=str(candidate_pkg),
            output_dir=str(tmp_path / "canary"),
            image_dir=str(image_dir),
            mask_dir=str(mask_dir),
        )
    )

    payload = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert payload["failed_images"] == 0
    assert payload["mean_disagreement_fraction"] > 0.0
    assert payload["mean_candidate_iou_gain"] > 0.0


def test_phase26_deploy_perf_harness_outputs_metrics(monkeypatch, tmp_path: Path) -> None:
    import src.microseg.deployment.perf_benchmark as perf_mod

    pkg_dir = _make_package(tmp_path, package_name="perf_pkg")
    image_dir = tmp_path / "images"
    for idx in range(2):
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        _write_rgb(image_dir / f"i{idx}.png", arr)

    def _predictor_factory(_artifact, *, enable_gpu, device_policy):  # noqa: ANN001
        def _predict(image_rgb):
            time.sleep(0.005)
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        return _predict

    monkeypatch.setattr(perf_mod, "build_predictor_from_artifact", _predictor_factory)

    result = run_deployment_perf(
        DeploymentPerfConfig(
            package_dir=str(pkg_dir),
            output_dir=str(tmp_path / "perf"),
            image_dir=str(image_dir),
            warmup_runs=1,
            repeat=2,
            max_workers=2,
        )
    )

    payload = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert payload["total_requests"] == 4
    assert payload["failed_requests"] == 0
    assert payload["throughput_images_per_second"] > 0.0
    assert payload["latency_ms_p95"] >= 0.0
    assert Path(result.csv_path).exists()
