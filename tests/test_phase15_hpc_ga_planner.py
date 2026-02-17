"""Phase 15 tests for GA-based HPC bundle generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.microseg.app.hpc_ga import (
    HpcGaPlanConfig,
    generate_hpc_ga_bundle,
    load_feedback_samples,
    parse_architectures,
    parse_batch_sizes,
    parse_feedback_sources,
    parse_pretrained_model_map,
    plan_hpc_ga_candidates,
    summarize_feedback_sources,
)
from src.microseg.app.workflow_profiles import read_workflow_profile, write_workflow_profile


def test_phase15_parse_helpers() -> None:
    assert parse_architectures("unet_binary, torch_pixel") == ("unet_binary", "torch_pixel")
    assert parse_batch_sizes("4,8,16") == (4, 8, 16)
    assert parse_feedback_sources("a,b") == ("a", "b")
    assert parse_pretrained_model_map('{"hf_segformer_b0":"hf_segformer_b0_ade20k"}') == {
        "hf_segformer_b0": "hf_segformer_b0_ade20k"
    }
    assert parse_pretrained_model_map("hf_segformer_b2=hf_segformer_b2_ade20k") == {
        "hf_segformer_b2": "hf_segformer_b2_ade20k"
    }


def _write_feedback_sample(
    bundle_dir: Path,
    *,
    candidate_id: str,
    backend: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    max_samples: int,
    pixel_accuracy: float,
    macro_f1: float,
    mean_iou: float,
    runtime_seconds: float,
) -> None:
    candidates_dir = bundle_dir / "candidates"
    runs_dir = bundle_dir / "runs" / candidate_id
    candidates_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / f"{candidate_id}.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "backend": backend,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "max_samples": max_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (runs_dir / "eval_report.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "pixel_accuracy": pixel_accuracy,
                    "macro_f1": macro_f1,
                    "mean_iou": mean_iou,
                },
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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


def test_phase15_feedback_summary_and_hybrid_planning(tmp_path: Path) -> None:
    feedback_bundle = tmp_path / "prior_bundle"
    _write_feedback_sample(
        feedback_bundle,
        candidate_id="cand_001",
        backend="unet_binary",
        learning_rate=0.001,
        batch_size=8,
        epochs=12,
        weight_decay=1e-5,
        max_samples=60000,
        pixel_accuracy=0.94,
        macro_f1=0.89,
        mean_iou=0.87,
        runtime_seconds=120.0,
    )
    _write_feedback_sample(
        feedback_bundle,
        candidate_id="cand_002",
        backend="torch_pixel",
        learning_rate=0.0005,
        batch_size=16,
        epochs=18,
        weight_decay=1e-4,
        max_samples=70000,
        pixel_accuracy=0.88,
        macro_f1=0.83,
        mean_iou=0.79,
        runtime_seconds=95.0,
    )
    _write_feedback_sample(
        feedback_bundle,
        candidate_id="cand_003",
        backend="unet_binary",
        learning_rate=0.002,
        batch_size=4,
        epochs=10,
        weight_decay=2e-5,
        max_samples=50000,
        pixel_accuracy=0.90,
        macro_f1=0.84,
        mean_iou=0.81,
        runtime_seconds=140.0,
    )

    cfg = HpcGaPlanConfig(
        dataset_dir="outputs/prepared_dataset",
        output_dir=str(tmp_path / "bundle_out"),
        architectures=("unet_binary", "torch_pixel"),
        batch_size_choices=(4, 8, 16),
        num_candidates=4,
        population_size=10,
        generations=2,
        seed=11,
        fitness_mode="feedback_hybrid",
        feedback_sources=(str(feedback_bundle),),
        feedback_min_samples=2,
        feedback_k=2,
    )

    samples = load_feedback_samples(cfg.feedback_sources, cfg=cfg)
    assert len(samples) == 3
    assert all(s.fitness_score == s.fitness_score for s in samples)  # no NaN

    summary = summarize_feedback_sources(cfg.feedback_sources, cfg=cfg, top_k=2)
    assert summary["sample_count"] == 3
    assert len(summary["top_candidates"]) == 2

    candidates = plan_hpc_ga_candidates(cfg)
    assert len(candidates) == 4
    assert all(c.selection_score is not None for c in candidates)
    assert any(c.predicted_fitness is not None for c in candidates)

    result = generate_hpc_ga_bundle(cfg)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["feedback_summary"] is not None
    assert int(manifest["feedback_summary"]["sample_count"]) == 3


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
            "fitness_mode": "feedback_hybrid",
            "feedback_sources": "outputs/hpc_ga_bundle_a,outputs/hpc_ga_bundle_b",
            "feedback_report_output": "outputs/hpc_ga_feedback/feedback_report.json",
        },
    )
    loaded = read_workflow_profile(profile)
    assert loaded["scope"] == "hpc_ga"
    assert loaded["values"]["scheduler"] == "slurm"
    assert loaded["values"]["fitness_mode"] == "feedback_hybrid"


def test_phase15_hpc_job_script_includes_pretrained_overrides(tmp_path: Path) -> None:
    out = tmp_path / "hpc_bundle_pretrained"
    cfg = HpcGaPlanConfig(
        dataset_dir="outputs/prepared_dataset",
        output_dir=str(out),
        architectures=("hf_segformer_b0",),
        num_candidates=1,
        population_size=4,
        generations=1,
        seed=3,
        scheduler="local",
        pretrained_init_mode="local",
        pretrained_registry_path="pre_trained_weights/registry.json",
        pretrained_model_map={"hf_segformer_b0": "hf_segformer_b0_ade20k"},
        pretrained_verify_sha256=True,
        pretrained_ignore_mismatched_sizes=True,
        pretrained_strict=False,
    )

    generate_hpc_ga_bundle(cfg)
    job_text = (out / "jobs" / "cand_001.sh").read_text(encoding="utf-8")
    assert '"--set" "model_architecture=hf_segformer_b0"' in job_text
    assert '"--set" "pretrained_init_mode=local"' in job_text
    assert '"--set" "pretrained_model_id=hf_segformer_b0_ade20k"' in job_text
    assert '"--set" "pretrained_registry_path=pre_trained_weights/registry.json"' in job_text


def test_phase15_hpc_job_script_includes_pretrained_overrides_for_internal_backends(tmp_path: Path) -> None:
    cases = [
        ("unet_binary", "unet_binary_resnet18_imagenet_partial"),
        ("transunet_tiny", "transunet_tiny_vit_tiny_patch16_imagenet"),
        ("segformer_mini", "segformer_mini_vit_tiny_patch16_imagenet"),
    ]
    for idx, (backend, model_id) in enumerate(cases, start=1):
        out = tmp_path / f"hpc_bundle_pretrained_{idx}"
        cfg = HpcGaPlanConfig(
            dataset_dir="outputs/prepared_dataset",
            output_dir=str(out),
            architectures=(backend,),
            num_candidates=1,
            population_size=4,
            generations=1,
            seed=idx,
            scheduler="local",
            pretrained_init_mode="auto",
            pretrained_registry_path="pre_trained_weights/registry.json",
            pretrained_model_map={backend: model_id},
            pretrained_verify_sha256=True,
            pretrained_ignore_mismatched_sizes=True,
            pretrained_strict=False,
        )
        generate_hpc_ga_bundle(cfg)
        job_text = (out / "jobs" / "cand_001.sh").read_text(encoding="utf-8")
        assert f'"--set" "backend={backend}"' in job_text
        assert f'"--set" "model_architecture={backend}"' in job_text
        assert '"--set" "pretrained_init_mode=local"' in job_text
        assert f'"--set" "pretrained_model_id={model_id}"' in job_text


def test_phase15_hpc_local_pretrained_requires_mapping() -> None:
    cfg = HpcGaPlanConfig(
        dataset_dir="outputs/prepared_dataset",
        output_dir="outputs/hpc_ga_bundle",
        architectures=("hf_segformer_b0",),
        num_candidates=1,
        population_size=4,
        generations=1,
        pretrained_init_mode="local",
        pretrained_model_map={},
    )
    try:
        plan_hpc_ga_candidates(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pretrained_model_map" in str(exc)
