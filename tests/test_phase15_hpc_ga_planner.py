"""Phase 15 tests for GA-based HPC bundle generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.app.hpc_ga import (
    HpcGaPlanConfig,
    generate_hpc_ga_bundle,
    parse_architectures,
    parse_batch_sizes,
    plan_hpc_ga_candidates,
)
from src.microseg.app.workflow_profiles import read_workflow_profile, write_workflow_profile


def test_phase15_parse_helpers() -> None:
    assert parse_architectures("unet_binary, torch_pixel") == ("unet_binary", "torch_pixel")
    assert parse_batch_sizes("4,8,16") == (4, 8, 16)


def test_phase15_ga_planner_and_bundle_generation(tmp_path: Path) -> None:
    out = tmp_path / "hpc_bundle"
    cfg = HpcGaPlanConfig(
        dataset_dir="outputs/prepared_dataset",
        output_dir=str(out),
        architectures=("unet_binary", "torch_pixel"),
        num_candidates=4,
        population_size=10,
        generations=3,
        seed=9,
        scheduler="slurm",
    )

    candidates = plan_hpc_ga_candidates(cfg)
    assert len(candidates) == 4
    assert all(c.backend in {"unet_binary", "torch_pixel"} for c in candidates)

    result = generate_hpc_ga_bundle(cfg)
    assert Path(result.bundle_dir).exists()
    assert Path(result.manifest_path).exists()
    assert Path(result.submit_script).exists()
    assert len(result.candidates) == 4

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "microseg.hpc_ga_bundle.v1"
    assert manifest["config"]["scheduler"] == "slurm"
    assert len(manifest["candidates"]) == 4

    submit_text = Path(result.submit_script).read_text(encoding="utf-8")
    assert "sbatch" in submit_text

    first_job = out / "jobs" / "cand_001.sh"
    assert first_job.exists()
    job_text = first_job.read_text(encoding="utf-8")
    assert "#SBATCH" in job_text
    assert "scripts/microseg_cli.py" in job_text
    assert "train" in job_text


def test_phase15_workflow_profile_roundtrip_hpc_ga_scope(tmp_path: Path) -> None:
    profile = tmp_path / "hpc_profile.yml"
    write_workflow_profile(
        profile,
        scope="hpc_ga",
        values={
            "dataset_dir": "outputs/prepared_dataset",
            "output_dir": "outputs/hpc_ga_bundle",
            "architectures": "unet_binary,torch_pixel",
            "num_candidates": 4,
            "scheduler": "slurm",
        },
    )
    loaded = read_workflow_profile(profile)
    assert loaded["scope"] == "hpc_ga"
    assert loaded["values"]["scheduler"] == "slurm"

