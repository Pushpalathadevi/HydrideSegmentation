"""Phase 14 tests for checkpoint lifecycle and smoke artifact behavior."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

from src.microseg.inference.predictors import HydrideMLPredictor
from src.microseg.plugins import FrozenCheckpointRecord
from src.microseg.plugins.registry_validation import validate_frozen_registry


def test_phase14_ml_predictor_uses_registry_hint_when_weights_missing(tmp_path: Path, monkeypatch) -> None:
    hinted = tmp_path / "frozen_checkpoints" / "smoke" / "torch_pixel_smoke_random_v1.pth"
    hinted.parent.mkdir(parents=True, exist_ok=True)
    hinted.write_bytes(b"smoke")

    record = FrozenCheckpointRecord(
        model_id="hydride_ml",
        model_nickname="smoke",
        model_type="binary_unet",
        framework="pytorch",
        input_size="variable",
        input_dimensions="H x W x 3",
        checkpoint_path_hint="frozen_checkpoints/smoke/torch_pixel_smoke_random_v1.pth",
        application_remarks="smoke",
        classes=tuple(),
    )

    monkeypatch.setattr("src.microseg.inference.predictors.frozen_checkpoint_map", lambda: {"hydride_ml": record})
    monkeypatch.setattr("src.microseg.inference.predictors.find_repo_root", lambda _start=None: tmp_path)

    observed: dict[str, str] = {}
    fake_module = types.ModuleType("hydride_segmentation.inference")

    def _fake_run_model(image_path: str, params=None, weights_path: str = ""):  # noqa: ANN001
        observed["image_path"] = image_path
        observed["weights_path"] = weights_path
        return np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)

    fake_module.run_model = _fake_run_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydride_segmentation.inference", fake_module)

    out = HydrideMLPredictor().predict("dummy_input.png", params={})
    assert out.image.shape == (8, 8, 3)
    assert out.mask.shape == (8, 8)
    assert observed["weights_path"] == str(hinted.resolve())


def test_phase14_registry_validation_accepts_lifecycle_optional_fields(tmp_path: Path) -> None:
    reg = tmp_path / "model_registry.json"
    reg.write_text(
        """
{
  "schema_version": "microseg.frozen_checkpoint_registry.v1",
  "models": [
    {
      "model_id": "smoke_a",
      "model_nickname": "smoke_a_v1",
      "model_type": "pixel_linear_smoke",
      "framework": "pytorch",
      "input_size": "variable",
      "input_dimensions": "H x W x 3",
      "checkpoint_path_hint": "frozen_checkpoints/smoke/smoke_a.pth",
      "application_remarks": "smoke",
      "artifact_stage": "smoke",
      "source_run_manifest": "outputs/training/run_001/training_manifest.json",
      "quality_report_path": "outputs/evaluation/run_001/report.json",
      "file_sha256": "abc123",
      "file_size_bytes": 128
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = validate_frozen_registry(reg)
    assert report.ok is True
    assert not report.errors

