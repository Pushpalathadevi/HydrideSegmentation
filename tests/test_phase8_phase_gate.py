"""Phase 8 tests for phase-gate closeout validation tooling."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.deployment import DeploymentPackageConfig, create_deployment_package
from src.microseg.quality.phase_gate import PhaseGateConfig, run_phase_gate


def _make_min_repo(root: Path) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# repo\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (root / "docs" / "README.md").write_text("# docs\n", encoding="utf-8")
    (root / "docs" / "development_roadmap.md").write_text("# roadmap\n", encoding="utf-8")
    (root / "docs" / "current_state_gap_analysis.md").write_text("# gaps\n", encoding="utf-8")
    (root / "docs" / "development_workflow.md").write_text("# workflow\n", encoding="utf-8")


def _make_dummy_deployment_package(root: Path) -> Path:
    model = root / "dummy_model.joblib"
    model.write_bytes(b"dummy-model")
    result = create_deployment_package(
        DeploymentPackageConfig(
            model_path=str(model),
            output_dir=str(root / "outputs" / "deployments"),
            package_name="phase8_test",
        )
    )
    return Path(result.package_dir)


def test_phase8_phase_gate_pass(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_min_repo(repo)

    import src.microseg.quality.phase_gate as gate_mod

    monkeypatch.setattr(gate_mod, "_repo_root", lambda start=None: repo)
    monkeypatch.setattr(gate_mod, "_run_tests", lambda _repo_root: (True, 0, "pytest -q"))

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label="Phase 8",
            run_tests=True,
            output_dir="outputs/phase_gates",
        )
    )

    assert result.status == "pass"
    assert result.tests_passed is True
    assert not result.gaps

    json_path = Path(result.artifacts["json_report"])
    md_path = Path(result.artifacts["markdown_stocktake"])
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "microseg.phase_gate.v1"


def test_phase8_phase_gate_fails_on_absolute_md_refs(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_min_repo(repo)
    (repo / "docs" / "bad.md").write_text("See /Users/local/path/file.md\n", encoding="utf-8")

    import src.microseg.quality.phase_gate as gate_mod

    monkeypatch.setattr(gate_mod, "_repo_root", lambda start=None: repo)
    monkeypatch.setattr(gate_mod, "_run_tests", lambda _repo_root: (True, 0, "pytest -q"))

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label="Phase 8",
            run_tests=True,
            output_dir="outputs/phase_gates",
        )
    )
    assert result.status == "fail"
    assert result.absolute_md_refs
    assert any("Absolute markdown path references" in item for item in result.gaps)


def test_phase8_phase_gate_release_policy_and_deployment_checks(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_min_repo(repo)
    (repo / "docs" / "versioning_and_release_policy.md").write_text(
        "# Release\nRollback procedure and patch protocol are documented here.\n",
        encoding="utf-8",
    )
    package_dir = _make_dummy_deployment_package(repo)

    import src.microseg.quality.phase_gate as gate_mod

    monkeypatch.setattr(gate_mod, "_repo_root", lambda start=None: repo)
    monkeypatch.setattr(gate_mod, "_run_tests", lambda _repo_root: (True, 0, "pytest -q"))

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label="Phase 8",
            run_tests=True,
            output_dir="outputs/phase_gates",
            verify_release_policy=True,
            require_rollback_keywords=True,
            deployment_package_dirs=(str(package_dir),),
        )
    )
    assert result.status == "pass"
    assert result.release_policy_ok is True
    assert result.deployment_packages_checked
    assert result.deployment_packages_checked[0]["ok"] is True


def test_phase8_phase_gate_fails_release_policy_keyword_check(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_min_repo(repo)
    (repo / "docs" / "versioning_and_release_policy.md").write_text(
        "# Release\nThis file intentionally omits one required token.\n",
        encoding="utf-8",
    )

    import src.microseg.quality.phase_gate as gate_mod

    monkeypatch.setattr(gate_mod, "_repo_root", lambda start=None: repo)
    monkeypatch.setattr(gate_mod, "_run_tests", lambda _repo_root: (True, 0, "pytest -q"))

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label="Phase 8",
            run_tests=True,
            output_dir="outputs/phase_gates",
            verify_release_policy=True,
            require_rollback_keywords=True,
            rollback_keywords=("rollback", "patch", "release", "incident"),
        )
    )
    assert result.status == "fail"
    assert any("release policy missing required keywords" in gap for gap in result.gaps)
