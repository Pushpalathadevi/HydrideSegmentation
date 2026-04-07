"""End-to-end raw `.oh5` benchmark workflow orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from src.microseg.data_preparation.oh5 import Oh5ExtractionConfig, extract_oh5_dataset
from src.microseg.dataops import DatasetPrepareConfig, DatasetQualityConfig, prepare_training_dataset_layout, run_dataset_quality_checks


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PhaseIdBenchmarkWorkflowConfig:
    """Configuration for raw `.oh5` to benchmark suite workflow."""

    raw_input_dir: str
    working_dir: str
    benchmark_template: str
    deck_title: str = "PhaseId Benchmark Lab Meeting"
    strict: bool = True
    python_executable: str = sys.executable
    node_executable: str = "node"
    benchmark_single_seed: bool = False
    benchmark_strict: bool = True
    benchmark_extra_args: tuple[str, ...] = field(default_factory=tuple)
    extraction: Oh5ExtractionConfig | None = None
    dataset_prepare: DatasetPrepareConfig | None = None
    dataset_qa: DatasetQualityConfig | None = None


def _run_command(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_extraction_config(config: PhaseIdBenchmarkWorkflowConfig) -> Oh5ExtractionConfig:
    if config.extraction is not None:
        return config.extraction
    work_dir = Path(config.working_dir)
    return Oh5ExtractionConfig(
        input_dir=config.raw_input_dir,
        output_dir=str((work_dir / "oh5_extracted").resolve()),
    )


def _resolve_prepare_config(config: PhaseIdBenchmarkWorkflowConfig, *, extracted_dir: str) -> DatasetPrepareConfig:
    if config.dataset_prepare is not None:
        return config.dataset_prepare
    work_dir = Path(config.working_dir)
    return DatasetPrepareConfig(
        dataset_dir=str(Path(extracted_dir).resolve()),
        output_dir=str((work_dir / "prepared_dataset").resolve()),
    )


def _resolve_qa_config(config: PhaseIdBenchmarkWorkflowConfig, *, prepared_dir: str) -> DatasetQualityConfig:
    if config.dataset_qa is not None:
        return config.dataset_qa
    work_dir = Path(config.working_dir)
    return DatasetQualityConfig(
        dataset_dir=str(Path(prepared_dir).resolve()),
        output_path=str((work_dir / "dataset_qa" / "dataset_qa_report.json").resolve()),
        strict=bool(config.strict),
    )


def _generate_suite_config(
    config: PhaseIdBenchmarkWorkflowConfig,
    *,
    prepared_dataset_dir: str,
) -> tuple[Path, dict[str, Any]]:
    repo_root = _repo_root()
    work_dir = Path(config.working_dir)
    template_path = Path(config.benchmark_template)
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark template must be a mapping: {template_path}")
    output_root = work_dir / "benchmarks"
    payload["dataset_dir"] = str(Path(prepared_dataset_dir).resolve())
    payload["output_root"] = str(output_root.resolve())
    payload["python_executable"] = str(config.python_executable)
    generated_path = work_dir / "generated_benchmark_suite.yml"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return generated_path, payload


def run_phaseid_benchmark_workflow(config: PhaseIdBenchmarkWorkflowConfig) -> dict[str, Any]:
    """Run raw-data extraction, dataset prep/QA, benchmark suite, and PPT generation."""

    repo_root = _repo_root()
    work_dir = Path(config.working_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    extraction_cfg = _resolve_extraction_config(config)
    extract_result = extract_oh5_dataset(extraction_cfg)

    prepare_cfg = _resolve_prepare_config(config, extracted_dir=extract_result.output_dir)
    prepared = prepare_training_dataset_layout(prepare_cfg)

    qa_cfg = _resolve_qa_config(config, prepared_dir=prepared.dataset_dir)
    qa_report = run_dataset_quality_checks(qa_cfg)
    if bool(config.strict) and not qa_report.ok:
        raise RuntimeError("dataset QA failed for phase-id workflow")

    suite_config_path, suite_payload = _generate_suite_config(config, prepared_dataset_dir=prepared.dataset_dir)
    benchmark_cmd = [
        str(config.python_executable),
        str((repo_root / "scripts" / "hydride_benchmark_suite.py").resolve()),
        "--config",
        str(suite_config_path),
    ]
    if bool(config.benchmark_single_seed):
        benchmark_cmd.append("--single-seed")
    if bool(config.benchmark_strict):
        benchmark_cmd.append("--strict")
    benchmark_cmd.extend(config.benchmark_extra_args)
    benchmark_proc = _run_command(benchmark_cmd, cwd=repo_root)
    if benchmark_proc.returncode != 0:
        raise RuntimeError(
            "benchmark suite failed.\n"
            f"stdout:\n{benchmark_proc.stdout}\n"
            f"stderr:\n{benchmark_proc.stderr}"
        )

    benchmark_output_root = Path(str(suite_payload["output_root"])).resolve()
    summary_json = benchmark_output_root / "benchmark_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"benchmark summary not found: {summary_json}")

    deck_output_dir = work_dir / "lab_meeting_deck"
    deck_cmd = [
        str(config.python_executable),
        str((repo_root / "scripts" / "generate_benchmark_lab_meeting_ppt.py").resolve()),
        "--summary-json",
        str(summary_json),
        "--output-dir",
        str(deck_output_dir),
        "--deck-title",
        str(config.deck_title),
        "--node-executable",
        str(config.node_executable),
    ]
    deck_proc = _run_command(deck_cmd, cwd=repo_root)
    if deck_proc.returncode != 0:
        raise RuntimeError(
            "PPT generation failed.\n"
            f"stdout:\n{deck_proc.stdout}\n"
            f"stderr:\n{deck_proc.stderr}"
        )

    report = {
        "schema_version": "microseg.phaseid_benchmark_workflow.v1",
        "config": asdict(config),
        "extract_report_path": extract_result.report_path,
        "prepared_dataset_dir": prepared.dataset_dir,
        "dataset_prepare_manifest_path": prepared.manifest_path,
        "dataset_qa_report_path": str(Path(qa_cfg.output_path).resolve()),
        "generated_benchmark_config_path": str(suite_config_path.resolve()),
        "benchmark_output_root": str(benchmark_output_root),
        "benchmark_summary_json": str(summary_json.resolve()),
        "benchmark_dashboard_html": str((benchmark_output_root / "benchmark_dashboard.html").resolve()),
        "deck_output_dir": str(deck_output_dir.resolve()),
        "workflow_stdout": {
            "benchmark": benchmark_proc.stdout,
            "deck": deck_proc.stdout,
        },
    }
    report_path = work_dir / "phaseid_benchmark_workflow_report.json"
    report["report_path"] = str(report_path.resolve())
    _write_json(report_path, report)
    return report
