"""Phase 7 tests for frozen checkpoint metadata registry integration."""

from __future__ import annotations

import json

from hydride_segmentation.microseg_adapter import get_gui_model_specs
from src.microseg.plugins.frozen_checkpoints import load_frozen_checkpoint_records, registry_path


def test_phase7_registry_file_and_records_exist() -> None:
    reg_path = registry_path()
    assert reg_path.exists()

    records = load_frozen_checkpoint_records(reg_path)
    assert records
    assert any(rec.model_id == "hydride_ml" for rec in records)


def test_phase7_gui_specs_include_frozen_checkpoint_fields() -> None:
    specs = get_gui_model_specs()
    by_id = {spec["model_id"]: spec for spec in specs}
    assert "hydride_ml" in by_id

    ml = by_id["hydride_ml"]
    assert ml.get("model_nickname")
    assert ml.get("checkpoint_path_hint")
    assert ml.get("application_remarks")

    candidate = by_id.get("hydride_ml_Unet")
    assert candidate is not None
    assert candidate.get("file_sha256")
    assert candidate.get("file_size_bytes")


def test_phase7_local_registry_overlay_is_merged(tmp_path) -> None:
    reg_root = tmp_path / "frozen_checkpoints"
    reg_root.mkdir(parents=True, exist_ok=True)
    reg_path = reg_root / "model_registry.json"
    overlay_path = reg_root / "model_registry.local.json"

    reg_path.write_text(
        json.dumps(
            {
                "schema_version": "microseg.frozen_checkpoint_registry.v1",
                "updated_utc": "2026-04-08T00:00:00Z",
                "models": [
                    {
                        "model_id": "canonical_model",
                        "model_nickname": "canonical",
                        "model_type": "rule_based",
                        "framework": "opencv+numpy",
                        "input_size": "variable",
                        "input_dimensions": "H x W x 3",
                        "checkpoint_path_hint": "n/a",
                        "application_remarks": "canonical entry",
                        "short_description": "",
                        "detailed_description": "",
                        "artifact_stage": "builtin",
                        "source_run_manifest": "",
                        "quality_report_path": "",
                        "file_sha256": "",
                        "file_size_bytes": None,
                        "classes": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    overlay_path.write_text(
        json.dumps(
            {
                "schema_version": "microseg.frozen_checkpoint_registry.v1",
                "updated_utc": "2026-04-08T00:00:00Z",
                "models": [
                    {
                        "model_id": "local_overlay_model",
                        "model_nickname": "local_overlay",
                        "model_type": "unet_binary",
                        "framework": "pytorch",
                        "input_size": "variable",
                        "input_dimensions": "H x W x 3",
                        "checkpoint_path_hint": "frozen_checkpoints/candidates/local_overlay_model.pt",
                        "application_remarks": "local overlay entry",
                        "short_description": "local overlay",
                        "detailed_description": "local overlay",
                        "artifact_stage": "candidate",
                        "source_run_manifest": "",
                        "quality_report_path": "",
                        "file_sha256": "",
                        "file_size_bytes": 123,
                        "classes": [
                            {
                                "index": 0,
                                "name": "background",
                                "color_hex": "#000000",
                            }
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records = load_frozen_checkpoint_records(reg_path)
    ids = {rec.model_id for rec in records}
    assert "canonical_model" in ids
    assert "local_overlay_model" in ids
