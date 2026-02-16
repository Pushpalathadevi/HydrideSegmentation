"""Phase 7 tests for frozen checkpoint metadata registry integration."""

from __future__ import annotations

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
