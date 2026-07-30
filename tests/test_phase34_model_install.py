"""Phase 34 tests for local checkpoint installation and model availability."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.microseg.plugins import (
    ModelInstallRequest,
    inspect_checkpoint,
    install_model,
    local_registry_entries,
    model_catalog_status,
    uninstall_model,
    verify_checkpoint_runtime,
    write_install_report,
)
from src.microseg.plugins.frozen_checkpoints import REGISTRY_SCHEMA
from src.microseg.training.unet_binary import _build_binary_model


def _fake_repo(root: Path) -> Path:
    """Create a minimal repository layout that the installer can target."""

    (root / "frozen_checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("fake repo", encoding="utf-8")
    return root


def _write_checkpoint(path: Path, *, base_channels: int = 8, architecture: str = "unet_binary") -> Path:
    import torch

    model = _build_binary_model(
        architecture=architecture,
        base_channels=base_channels,
        transformer_depth=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        segformer_patch_size=4,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "microseg.torch_unet_binary.v1",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "epoch": 7,
            "best_val_loss": 0.125,
            "backend": architecture,
            "model_architecture": architecture,
            "model_state_dict": model.state_dict(),
            "config": {
                "model_architecture": architecture,
                "model_base_channels": base_channels,
                "input_hw": (256, 256),
            },
        },
        path,
    )
    return path


def _overlay(root: Path) -> Path:
    return root / "frozen_checkpoints" / "model_registry.local.json"


def test_inspect_checkpoint_recovers_architecture_and_provenance(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "source" / "best_model.pt")

    introspection = inspect_checkpoint(ckpt)

    assert introspection.architecture == "unet_binary"
    assert introspection.architecture_supported is True
    assert introspection.schema_version == "microseg.torch_unet_binary.v1"
    assert introspection.epoch == 7
    assert introspection.best_val_loss == pytest.approx(0.125)
    assert introspection.input_size == "256x256"
    assert introspection.parameter_count and introspection.parameter_count > 0
    assert len(introspection.file_sha256) == 64
    assert introspection.file_size_bytes == ckpt.stat().st_size
    assert introspection.suggested_model_id() == "best_model"
    assert introspection.warnings == ()


def test_inspect_checkpoint_rejects_non_checkpoint_payload(tmp_path: Path) -> None:
    import torch

    bogus = tmp_path / "bogus.pt"
    torch.save([1, 2, 3], bogus)

    with pytest.raises(ValueError):
        inspect_checkpoint(bogus)


def test_verify_checkpoint_runtime_loads_and_runs_forward_pass(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "best_model.pt")

    outcome = verify_checkpoint_runtime(ckpt)

    assert outcome["ok"] is True
    assert outcome["loaded"] is True
    assert outcome["forward_pass"] is True
    assert outcome["architecture"] == "unet_binary"


def test_install_model_copies_checkpoint_and_writes_overlay(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "my_best.pt")

    result = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="my_unet_v1",
            model_nickname="my_unet_v1_optical",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    assert result.ok is True, result.errors
    assert result.architecture == "unet_binary"
    assert result.checkpoint_path_hint == "frozen_checkpoints/candidates/my_unet_v1/my_best.pt"
    assert (root / result.checkpoint_path_hint).exists()
    assert ckpt.exists(), "the source checkpoint must not be moved"
    assert result.verification["forward_pass"] is True

    entries = local_registry_entries(_overlay(root))
    assert [item["model_id"] for item in entries] == ["my_unet_v1"]
    entry = entries[0]
    assert entry["model_type"] == "unet_binary"
    assert entry["artifact_stage"] == "candidate"
    assert entry["file_size_bytes"] == ckpt.stat().st_size
    assert entry["classes"][0]["index"] == 0
    assert "epoch 7" in entry["detailed_description"]

    payload = json.loads(_overlay(root).read_text(encoding="utf-8"))
    assert payload["schema_version"] == REGISTRY_SCHEMA
    assert payload["updated_utc"]


def test_install_model_uses_stage_folder_matching_artifact_stage(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "promoted_model.pt")

    result = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="promoted_unet",
            model_nickname="promoted_unet_v1",
            artifact_stage="promoted",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    assert result.ok is True, result.errors
    assert result.checkpoint_path_hint.startswith("frozen_checkpoints/promoted/")


def test_install_model_rejects_duplicate_id_without_overwrite(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "model.pt")
    request = ModelInstallRequest(
        checkpoint_path=str(ckpt),
        model_id="dup_model",
        model_nickname="dup_model_v1",
    )

    assert install_model(request, repo_root=root, overlay_path=_overlay(root)).ok is True

    repeated = install_model(request, repo_root=root, overlay_path=_overlay(root))
    assert repeated.ok is False
    assert any("already installed" in error for error in repeated.errors)

    overwritten = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="dup_model",
            model_nickname="dup_model_v2",
            overwrite=True,
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )
    assert overwritten.ok is True
    entries = local_registry_entries(_overlay(root))
    assert len(entries) == 1
    assert entries[0]["model_nickname"] == "dup_model_v2"


def test_install_model_rejects_reserved_and_malformed_identifiers(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "model.pt")

    reserved = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="hydride_conventional",
            model_nickname="attempted_override",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )
    assert reserved.ok is False
    assert any("reserved" in error for error in reserved.errors)

    malformed = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="Bad Model Id",
            model_nickname="bad_id",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )
    assert malformed.ok is False
    assert not _overlay(root).exists()


def test_install_model_rolls_back_when_verification_fails(tmp_path: Path) -> None:
    import torch

    root = _fake_repo(tmp_path / "repo")
    broken = tmp_path / "incoming" / "broken.pt"
    broken.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "microseg.torch_unet_binary.v1",
            "model_architecture": "unet_binary",
            "backend": "unet_binary",
            "config": {"model_architecture": "unet_binary", "model_base_channels": 8},
            "model_state_dict": {"not_a_real_layer.weight": torch.zeros(3, 3)},
        },
        broken,
    )

    result = install_model(
        ModelInstallRequest(
            checkpoint_path=str(broken),
            model_id="broken_model",
            model_nickname="broken_model_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    assert result.ok is False
    assert result.errors
    assert not (root / "frozen_checkpoints" / "candidates" / "broken_model").exists()
    assert not _overlay(root).exists()


def test_install_model_rejects_unsupported_architecture(tmp_path: Path) -> None:
    import torch

    root = _fake_repo(tmp_path / "repo")
    ckpt = tmp_path / "incoming" / "exotic.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "created_utc": "2026-01-01T00:00:00+00:00",
            "model_architecture": "totally_unknown_backbone",
            "model_state_dict": {"layer.weight": torch.zeros(2, 2)},
            "config": {},
        },
        ckpt,
    )

    introspection = inspect_checkpoint(ckpt)
    assert introspection.architecture_supported is False
    assert introspection.warnings

    result = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="exotic_model",
            model_nickname="exotic_model_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )
    assert result.ok is False
    assert any("not supported" in error for error in result.errors)


def test_uninstall_model_removes_entry_and_optionally_the_file(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "model.pt")
    installed = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="removable_model",
            model_nickname="removable_model_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )
    assert installed.ok is True, installed.errors
    installed_file = root / installed.checkpoint_path_hint

    result = uninstall_model(
        "removable_model",
        delete_checkpoint=True,
        repo_root=root,
        overlay_path=_overlay(root),
    )

    assert result.ok is True
    assert result.removed_registry_entry is True
    assert not installed_file.exists()
    assert local_registry_entries(_overlay(root)) == []


def test_uninstall_model_reports_unknown_identifier(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    _overlay(root).write_text(
        json.dumps({"schema_version": REGISTRY_SCHEMA, "models": []}, indent=2),
        encoding="utf-8",
    )

    result = uninstall_model("never_installed", repo_root=root, overlay_path=_overlay(root))

    assert result.ok is False
    assert any("not installed locally" in error for error in result.errors)


def test_model_catalog_status_flags_missing_checkpoints(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    canonical = root / "frozen_checkpoints" / "model_registry.json"
    canonical.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA,
                "models": [
                    {
                        "model_id": "absent_model",
                        "model_nickname": "absent_model_v1",
                        "model_type": "unet_binary",
                        "framework": "pytorch",
                        "input_size": "variable",
                        "input_dimensions": "H x W x 3",
                        "checkpoint_path_hint": "frozen_checkpoints/candidates/absent/model.pt",
                        "application_remarks": "missing on purpose",
                        "artifact_stage": "candidate",
                        "classes": [{"index": 0, "name": "background", "color_hex": "#000000"}],
                    },
                    {
                        "model_id": "rule_based_model",
                        "model_nickname": "rule_based_v1",
                        "model_type": "rule_based",
                        "framework": "opencv+numpy",
                        "input_size": "variable",
                        "input_dimensions": "H x W x 3",
                        "checkpoint_path_hint": "n/a (classical pipeline)",
                        "application_remarks": "builtin",
                        "artifact_stage": "builtin",
                        "classes": [{"index": 0, "name": "background", "color_hex": "#000000"}],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ckpt = _write_checkpoint(tmp_path / "incoming" / "present.pt")
    install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="present_model",
            model_nickname="present_model_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    catalog = {item.model_id: item for item in model_catalog_status(repo_root=root, overlay_path=_overlay(root))}

    assert catalog["absent_model"].status == "checkpoint_missing"
    assert catalog["absent_model"].locally_installed is False
    assert catalog["rule_based_model"].status == "no_checkpoint_required"
    assert catalog["present_model"].status == "ready"
    assert catalog["present_model"].locally_installed is True


def test_install_never_modifies_the_tracked_canonical_registry(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    canonical = root / "frozen_checkpoints" / "model_registry.json"
    canonical_payload = {
        "schema_version": REGISTRY_SCHEMA,
        "models": [
            {
                "model_id": "shipped_model",
                "model_nickname": "shipped_model_v1",
                "model_type": "unet_binary",
                "framework": "pytorch",
                "input_size": "variable",
                "input_dimensions": "H x W x 3",
                "checkpoint_path_hint": "frozen_checkpoints/candidates/shipped/model.pt",
                "application_remarks": "shipped entry",
                "artifact_stage": "candidate",
                "classes": [{"index": 0, "name": "background", "color_hex": "#000000"}],
            }
        ],
    }
    canonical.write_text(json.dumps(canonical_payload, indent=2), encoding="utf-8")
    before = canonical.read_text(encoding="utf-8")

    ckpt = _write_checkpoint(tmp_path / "incoming" / "model.pt")
    install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="local_only_model",
            model_nickname="local_only_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    assert canonical.read_text(encoding="utf-8") == before


def test_write_install_report_persists_machine_readable_outcome(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path / "repo")
    ckpt = _write_checkpoint(tmp_path / "incoming" / "model.pt")
    result = install_model(
        ModelInstallRequest(
            checkpoint_path=str(ckpt),
            model_id="reported_model",
            model_nickname="reported_model_v1",
        ),
        repo_root=root,
        overlay_path=_overlay(root),
    )

    report_path = write_install_report(result, tmp_path / "reports" / "install_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "microseg.model_install_report.v1"
    assert payload["ok"] is True
    assert payload["model_id"] == "reported_model"
    assert payload["verification"]["forward_pass"] is True
    assert payload["introspection"]["architecture"] == "unet_binary"


def test_gui_model_specs_expose_availability_for_the_selector() -> None:
    from hydride_segmentation.microseg_adapter import get_gui_model_specs, model_is_runnable

    specs = get_gui_model_specs()

    assert specs
    for spec in specs:
        assert "availability" in spec
        assert "availability_message" in spec

    conventional = next(spec for spec in specs if spec["model_id"] == "hydride_conventional")
    assert conventional["availability"] == "no_checkpoint_required"
    assert model_is_runnable(conventional) is True
    assert model_is_runnable({"availability": "checkpoint_missing"}) is False


def test_desktop_workflow_defaults_to_conventional_when_no_checkpoint_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.microseg.app.desktop_workflow import DesktopWorkflowManager

    manager = DesktopWorkflowManager()
    monkeypatch.setattr(
        manager,
        "model_specs",
        lambda: [
            {
                "model_id": "hydride_ml",
                "display_name": "Hydride ML (UNet)",
                "availability": "checkpoint_missing",
            },
            {
                "model_id": "hydride_conventional",
                "display_name": "Hydride Conventional",
                "availability": "no_checkpoint_required",
            },
        ],
    )

    assert manager.preferred_default_model_name() == "Hydride Conventional"


def test_qt_selector_disables_models_whose_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import QtSegmentationMainWindow

    QApplication.instance() or QApplication([])
    win = QtSegmentationMainWindow()
    try:
        monkeypatch.setattr(win, "_start_model_warm_load", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            win.workflow,
            "model_specs",
            lambda: [
                {
                    "model_id": "ghost_model",
                    "display_name": "Ghost Model",
                    "description": "",
                    "details": "",
                    "availability": "checkpoint_missing",
                    "availability_message": "Checkpoint file not found.",
                },
                {
                    "model_id": "hydride_conventional",
                    "display_name": "Hydride Conventional",
                    "description": "",
                    "details": "",
                    "availability": "no_checkpoint_required",
                    "availability_message": "",
                },
            ],
        )
        win._reload_model_catalog()

        assert win.model_combo.count() == 2
        assert win.model_combo.model().item(0).isEnabled() is False
        assert win.model_combo.model().item(1).isEnabled() is True
        assert win.model_combo.currentText() == "Hydride Conventional"
        assert win._selected_model_is_runnable() is True
        assert win._selected_model_is_runnable("Ghost Model") is False

        win._on_model_changed("Ghost Model")
        assert "Unavailable" in win.model_desc.text()
        assert "Installed Models" in win.model_desc.text()
    finally:
        win.close()


def test_qt_install_dialog_prefills_from_checkpoint_metadata(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from hydride_segmentation.qt.main_window import InstallModelDialog

    QApplication.instance() or QApplication([])
    ckpt = _write_checkpoint(tmp_path / "incoming" / "field_model.pt")

    dialog = InstallModelDialog()
    try:
        dialog.path_edit.setText(str(ckpt))
        dialog.inspect_selected_checkpoint()

        assert "unet_binary" in dialog.detected_label.text()
        assert "NOT SUPPORTED" not in dialog.detected_label.text()
        assert dialog.model_id_edit.text() == "field_model"
        assert dialog.nickname_edit.text() == "field_model_local"

        request = dialog.selected_request()
        assert request.model_id == "field_model"
        assert request.artifact_stage == "candidate"
        assert request.verify_forward_pass is True
        assert request.classes[0]["index"] == 0
        assert request.classes[1]["name"] == "hydride"
    finally:
        dialog.close()
