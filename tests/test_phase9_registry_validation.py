"""Phase 9 tests for frozen registry strict validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.plugins.registry_validation import validate_frozen_registry


def test_phase9_registry_validation_passes_on_repo_registry() -> None:
    report = validate_frozen_registry("frozen_checkpoints/model_registry.json")
    assert report.ok is True
    assert not report.errors
    assert report.model_count >= 1


def test_phase9_registry_validation_detects_duplicate_ids(tmp_path: Path) -> None:
    bad = {
        "schema_version": "microseg.frozen_checkpoint_registry.v1",
        "models": [
            {
                "model_id": "x",
                "model_nickname": "nick1",
                "model_type": "binary_unet",
                "framework": "pytorch",
                "input_size": "variable",
                "input_dimensions": "H x W x 3",
                "checkpoint_path_hint": "frozen_checkpoints/x/model.pth",
                "application_remarks": "test",
                "classes": [{"index": 0, "name": "background"}],
            },
            {
                "model_id": "x",
                "model_nickname": "nick2",
                "model_type": "binary_unet",
                "framework": "pytorch",
                "input_size": "variable",
                "input_dimensions": "H x W x 3",
                "checkpoint_path_hint": "/abs/path/model.pth",
                "application_remarks": "test",
                "classes": [{"index": 1, "name": "feature"}],
            },
        ],
    }
    p = tmp_path / "bad_registry.json"
    p.write_text(json.dumps(bad), encoding="utf-8")

    report = validate_frozen_registry(p)
    assert report.ok is False
    assert any("duplicate model_id" in err for err in report.errors)
    assert any("absolute checkpoint_path_hint" in err for err in report.errors)
