"""Phase 2 tests for desktop workflow manager and export packaging."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.microseg.app.desktop_workflow import DesktopWorkflowManager
from hydride_segmentation.microseg_adapter import resolve_gui_model_id
from src.microseg.inference.predictors import HydrideMLPredictor
from src.microseg.inference.trained_model_loader import InferenceModelReference, run_reference_inference


def _synthetic_image_a() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[10:80, 12:28] = 255
    arr[20:75, 55:75] = 255
    return arr


def _synthetic_image_b() -> np.ndarray:
    arr = np.zeros((96, 96), dtype=np.uint8)
    arr[8:70, 8:24] = 255
    arr[30:86, 50:85] = 255
    return arr


def _tmp_image(image: np.ndarray) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(f.name)
    return f.name


def test_phase2_model_registry_options_available() -> None:
    mgr = DesktopWorkflowManager()
    options = mgr.model_options()
    assert options
    assert options[0] == "Hydride ML (UNet)"
    assert options[-1] == "Hydride Conventional"
    assert any(resolve_gui_model_id(name) == "hydride_conventional" for name in options)
    assert any(resolve_gui_model_id(name) == "hydride_ml" for name in options)


def test_phase2_preferred_default_model_prefers_trained_ml_checkpoint() -> None:
    mgr = DesktopWorkflowManager()
    preferred = mgr.preferred_default_model_name()
    assert preferred
    assert resolve_gui_model_id(preferred) != "hydride_conventional"


def test_phase2_dynamic_gui_models_are_included(monkeypatch) -> None:
    fake_binding = SimpleNamespace(
        model_id="my_unet_v2",
        display_name="My Unet V2",
        description="Trained unet_binary model",
        details="architecture=unet_binary",
        reference=SimpleNamespace(
            reference_id="registry::my_unet_v2",
            display_name="Registry: my_unet_v2 (unet_binary)",
            source="registry",
            checkpoint_path="frozen_checkpoints/candidates/my_unet_v2/best_checkpoint.pt",
            architecture="unet_binary",
            backend_label="unet_binary",
        ),
    )

    monkeypatch.setattr(
        "src.microseg.inference.predictors.discover_dynamic_ml_model_bindings",
        lambda: ([fake_binding], []),
    )

    from hydride_segmentation.microseg_adapter import get_gui_model_options, get_gui_model_specs

    options = get_gui_model_options()
    specs = get_gui_model_specs()
    assert "My Unet V2" in options
    assert any(spec["display_name"] == "My Unet V2" for spec in specs)
    assert options[-1] == "Hydride Conventional"


def test_phase2_cli_infer_defaults_to_first_discovered_model(monkeypatch, tmp_path: Path) -> None:
    from scripts import microseg_cli

    image_path = tmp_path / "input.png"
    Image.fromarray(_synthetic_image_a()).save(image_path)

    captured: dict[str, object] = {}

    class _FakeWorkflow:
        def preferred_default_model_name(self) -> str:
            return "Hydride ML (UNet)"

    def _fake_collect_inference_images(*, image, image_dir, glob_patterns, recursive):
        _ = image_dir, glob_patterns, recursive
        assert image == str(image_path)
        return [image_path]

    def _fake_run_desktop_batch_job(**kwargs):
        captured["model_name"] = kwargs["model_name"]
        captured["params"] = kwargs["params"]
        return SimpleNamespace(
            batch_dir=tmp_path / "batch",
            summary_json_path=tmp_path / "batch" / "batch_results_summary.json",
            records=[SimpleNamespace()],
        )

    monkeypatch.setattr(microseg_cli, "resolve_config", lambda *_args, **_kwargs: {"image_path": str(image_path)})
    monkeypatch.setattr(microseg_cli, "collect_inference_images", _fake_collect_inference_images)
    monkeypatch.setattr(microseg_cli, "DesktopWorkflowManager", lambda: _FakeWorkflow())
    monkeypatch.setattr(microseg_cli, "run_desktop_batch_job", _fake_run_desktop_batch_job)

    args = SimpleNamespace(
        config=None,
        set=[],
        image=str(image_path),
        image_dir="",
        recursive=True,
        glob_patterns="*.png",
        model_name=None,
        output_dir=str(tmp_path / "out"),
        enable_gpu=False,
        device_policy="cpu",
        capture_feedback=False,
        feedback_root="",
        deployment_id="",
        operator_id="",
    )

    assert microseg_cli._infer(args) == 0
    assert captured["model_name"] == "Hydride ML (UNet)"
    gui_preprocess = captured["params"]["gui_preprocess"]
    assert gui_preprocess["target_long_side"] == 512
    assert gui_preprocess["auto_contrast_enabled"] is True
    assert gui_preprocess["contrast_mode"] == "histogram_stretch"


def test_phase2_cli_infer_conventional_model_skips_gui_preprocess(monkeypatch, tmp_path: Path) -> None:
    from scripts import microseg_cli

    image_path = tmp_path / "input.png"
    Image.fromarray(_synthetic_image_a()).save(image_path)

    captured: dict[str, object] = {}

    class _FakeWorkflow:
        def preferred_default_model_name(self) -> str:
            return "Hydride ML (UNet)"

    def _fake_collect_inference_images(*, image, image_dir, glob_patterns, recursive):
        _ = image_dir, glob_patterns, recursive
        assert image == str(image_path)
        return [image_path]

    def _fake_run_desktop_batch_job(**kwargs):
        captured["model_name"] = kwargs["model_name"]
        captured["params"] = kwargs["params"]
        return SimpleNamespace(
            batch_dir=tmp_path / "batch",
            summary_json_path=tmp_path / "batch" / "batch_results_summary.json",
            records=[SimpleNamespace()],
        )

    monkeypatch.setattr(microseg_cli, "resolve_config", lambda *_args, **_kwargs: {"image_path": str(image_path)})
    monkeypatch.setattr(microseg_cli, "collect_inference_images", _fake_collect_inference_images)
    monkeypatch.setattr(microseg_cli, "DesktopWorkflowManager", lambda: _FakeWorkflow())
    monkeypatch.setattr(microseg_cli, "run_desktop_batch_job", _fake_run_desktop_batch_job)

    args = SimpleNamespace(
        config=None,
        set=[],
        image=str(image_path),
        image_dir="",
        recursive=True,
        glob_patterns="*.png",
        model_name="Hydride Conventional",
        output_dir=str(tmp_path / "out"),
        enable_gpu=False,
        device_policy="cpu",
        capture_feedback=False,
        feedback_root="",
        deployment_id="",
        operator_id="",
    )

    assert microseg_cli._infer(args) == 0
    assert captured["model_name"] == "Hydride Conventional"
    assert "gui_preprocess" not in captured["params"]


def test_phase2_single_run_and_export_package() -> None:
    mgr = DesktopWorkflowManager()
    conv_name = next(name for name in mgr.model_options() if resolve_gui_model_id(name) == "hydride_conventional")

    p = _tmp_image(_synthetic_image_a())
    out_dir = tempfile.mkdtemp(prefix="phase2_export_")
    try:
        record = mgr.run_single(p, model_name=conv_name, params={"image_path": p}, include_analysis=True)
        run_dir = mgr.export_run(record, out_dir)
    finally:
        Path(p).unlink(missing_ok=True)

    assert run_dir.exists()
    assert (run_dir / "input.png").exists()
    assert (run_dir / "prediction.png").exists()
    assert (run_dir / "overlay.png").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()


def test_phase2_batch_runs_recorded_in_history() -> None:
    mgr = DesktopWorkflowManager(max_history=10)
    conv_name = next(name for name in mgr.model_options() if resolve_gui_model_id(name) == "hydride_conventional")

    p1 = _tmp_image(_synthetic_image_a())
    p2 = _tmp_image(_synthetic_image_b())
    try:
        records = mgr.run_batch(
            [p1, p2],
            model_name=conv_name,
            params={"image_path": p1},
            include_analysis=False,
        )
    finally:
        Path(p1).unlink(missing_ok=True)
        Path(p2).unlink(missing_ok=True)

    assert len(records) == 2
    assert len(mgr.history()) == 2
    assert mgr.latest() is not None


def test_phase2_hydride_ml_predictor_defaults_to_registry_checkpoint(monkeypatch) -> None:
    predictor = HydrideMLPredictor()
    calls: dict[str, object] = {}

    def _fake_load_reference_from_registry(model_id: str):
        calls["model_id"] = model_id
        return object()

    def _fake_run_reference_inference(image_path, ref, *, enable_gpu, device_policy, preprocess_config):
        calls["image_path"] = image_path
        calls["ref"] = ref
        return np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8), {"timing": {}}

    monkeypatch.setattr("src.microseg.inference.predictors.load_reference_from_registry", _fake_load_reference_from_registry)
    monkeypatch.setattr("src.microseg.inference.predictors.run_reference_inference", _fake_run_reference_inference)
    output = predictor.predict("synthetic.png", params={"device_policy": "cpu"})
    assert calls["model_id"] == "hydride_ml"
    assert calls["image_path"] == "synthetic.png"
    assert output.manifest["timing"]["model_resolution_seconds"] >= 0.0


def test_phase2_ml_preprocess_display_image_preserves_original_size(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "ml_input.png"
    Image.fromarray(_synthetic_image_a()).save(image_path)

    reference = InferenceModelReference(
        reference_id="registry::hydride_ml",
        display_name="Hydride ML (UNet)",
        source="registry",
        checkpoint_path="fake.ckpt",
        architecture="unet_binary",
        backend_label="torch",
    )

    def _fake_get_or_load_reference_bundle(_reference, *, enable_gpu: bool, device_policy: str):
        _ = enable_gpu, device_policy
        return ({"device": "cpu"}, True, 0.0)

    def _fake_predict_unet_binary_mask(image: np.ndarray, bundle: dict[str, object]) -> np.ndarray:
        _ = bundle
        return np.ones(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        "src.microseg.inference.trained_model_loader.get_or_load_reference_bundle",
        _fake_get_or_load_reference_bundle,
    )
    monkeypatch.setattr(
        "src.microseg.inference.trained_model_loader.predict_unet_binary_mask",
        _fake_predict_unet_binary_mask,
    )

    display, mask, manifest = run_reference_inference(
        str(image_path),
        reference,
        preprocess_config={
            "target_long_side": 64,
            "auto_contrast_enabled": True,
            "contrast_mode": "histogram_stretch",
        },
    )

    assert display.shape == _synthetic_image_a().shape
    assert mask.shape == _synthetic_image_a().shape
    assert manifest["preprocessing"]["applied"] is True
    assert manifest["preprocessing"]["rescaled_to_original"] is True
    assert manifest["preprocessing"]["preprocessed_size"]["width"] == 64
