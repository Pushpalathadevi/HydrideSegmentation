"""Phase 14 tests for checkpoint lifecycle and smoke artifact behavior."""

from __future__ import annotations

from pathlib import Path
from src.microseg.inference.predictors import HydrideMLPredictor
from src.microseg.plugins import FrozenCheckpointRecord
from src.microseg.plugins.registry_validation import validate_frozen_registry


def test_phase14_ml_predictor_requires_explicit_unified_selector() -> None:
    predictor = HydrideMLPredictor()
    assert predictor.model_id == "hydride_ml"


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


def test_phase14_registry_validation_enforces_stage_hint_folder_consistency(tmp_path: Path) -> None:
    reg = tmp_path / "model_registry.json"
    reg.write_text(
        """
{
  "schema_version": "microseg.frozen_checkpoint_registry.v1",
  "models": [
    {
      "model_id": "bad_candidate",
      "model_nickname": "bad_candidate_v1",
      "model_type": "binary_unet",
      "framework": "pytorch",
      "input_size": "variable",
      "input_dimensions": "H x W x 3",
      "checkpoint_path_hint": "frozen_checkpoints/promoted/bad_candidate/model.pth",
      "application_remarks": "candidate",
      "artifact_stage": "candidate",
      "classes": [
        {
          "index": 0,
          "name": "background"
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = validate_frozen_registry(reg)
    assert report.ok is False
    assert any("artifact_stage='candidate'" in err for err in report.errors)


def test_phase14_registry_validation_rejects_absolute_canonical_checkpoint_hints(tmp_path: Path) -> None:
    reg = tmp_path / "model_registry.json"
    reg.write_text(
        """
{
  "schema_version": "microseg.frozen_checkpoint_registry.v1",
  "models": [
    {
      "model_id": "bad_absolute",
      "model_nickname": "bad_absolute_v1",
      "model_type": "unet_binary",
      "framework": "pytorch",
      "input_size": "variable",
      "input_dimensions": "H x W x 3",
      "checkpoint_path_hint": "C:/models/bad_absolute.pt",
      "application_remarks": "bad absolute path",
      "artifact_stage": "candidate",
      "classes": [
        {
          "index": 0,
          "name": "background"
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = validate_frozen_registry(reg)
    assert report.ok is False
    assert any("absolute checkpoint_path_hint" in err for err in report.errors)
