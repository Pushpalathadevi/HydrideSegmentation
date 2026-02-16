"""Phase-gate validation and closeout reporting utilities."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ABSOLUTE_MD_PATTERNS = [
    re.compile(r"/Users/"),
    re.compile(r"[A-Za-z]:\\\\"),
    re.compile(r"file://"),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "README.md").exists() and (parent / "docs").exists():
            return parent
    raise FileNotFoundError("could not locate repository root")


@dataclass(frozen=True)
class PhaseGateConfig:
    """Configuration for phase-gate validation run."""

    phase_label: str
    run_tests: bool = True
    output_dir: str = "outputs/phase_gates"
    extra_notes: str = ""
    strict: bool = False


@dataclass
class PhaseGateResult:
    """Validation outcome payload for phase closeout."""

    schema_version: str
    created_utc: str
    phase_label: str
    repo_root: str
    status: str
    tests_passed: bool
    tests_command: str
    tests_return_code: int
    absolute_md_refs: list[str] = field(default_factory=list)
    required_docs_missing: list[str] = field(default_factory=list)
    docs_checked: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    notes: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)


def _check_absolute_md_refs(repo_root: Path) -> list[str]:
    findings: list[str] = []
    md_files = [p for p in repo_root.rglob("*.md") if ".git" not in p.parts]
    for path in md_files:
        rel = path.relative_to(repo_root).as_posix()
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in ABSOLUTE_MD_PATTERNS):
                findings.append(f"{rel}:{idx}")
    return findings


def _check_required_docs(repo_root: Path) -> tuple[list[str], list[str]]:
    required = [
        "README.md",
        "AGENTS.md",
        "docs/README.md",
        "docs/development_roadmap.md",
        "docs/current_state_gap_analysis.md",
        "docs/development_workflow.md",
    ]
    missing = [path for path in required if not (repo_root / path).exists()]
    return required, missing


def _run_tests(repo_root: Path) -> tuple[bool, int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{old_pythonpath}" if old_pythonpath else str(repo_root)
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    return completed.returncode == 0, int(completed.returncode), " ".join(cmd)


def _write_markdown_summary(result: PhaseGateResult, output_path: Path) -> None:
    lines = [
        f"# {result.phase_label} Closeout Stocktake",
        "",
        f"- Timestamp (UTC): `{result.created_utc}`",
        f"- Status: `{result.status}`",
        f"- Tests passed: `{result.tests_passed}`",
        f"- Test command: `{result.tests_command}`",
        "",
        "## Checks",
        "",
        f"- Absolute markdown path references: `{len(result.absolute_md_refs)}`",
        f"- Missing required docs: `{len(result.required_docs_missing)}`",
        "",
        "## Gaps",
        "",
    ]
    if result.gaps:
        lines.extend([f"- {item}" for item in result.gaps])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Documentation Coverage",
            "",
            "- Checked docs:",
        ]
    )
    lines.extend([f"- `{path}`" for path in result.docs_checked])
    if result.notes:
        lines.extend(["", "## Notes", "", result.notes])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_gate(config: PhaseGateConfig) -> PhaseGateResult:
    """Execute phase-gate checks and write closeout artifacts."""

    repo_root = _repo_root()
    output_dir = repo_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_refs = _check_absolute_md_refs(repo_root)
    docs_checked, missing_docs = _check_required_docs(repo_root)

    if config.run_tests:
        tests_passed, tests_return_code, tests_cmd = _run_tests(repo_root)
    else:
        tests_passed, tests_return_code, tests_cmd = True, 0, "skipped"

    gaps: list[str] = []
    if not tests_passed:
        gaps.append("Full test suite did not pass.")
    if absolute_refs:
        gaps.append("Absolute markdown path references were found.")
    if missing_docs:
        gaps.append("Required governance/workflow docs are missing.")

    status = "pass" if not gaps else "fail"
    if status == "fail" and config.strict:
        raise RuntimeError("phase gate failed in strict mode")

    stem = config.phase_label.lower().replace(" ", "_").replace("/", "_")
    json_path = output_dir / f"{stem}_phase_gate.json"
    md_path = output_dir / f"{stem}_stocktake.md"

    result = PhaseGateResult(
        schema_version="microseg.phase_gate.v1",
        created_utc=_utc_now(),
        phase_label=config.phase_label,
        repo_root=str(repo_root),
        status=status,
        tests_passed=tests_passed,
        tests_command=tests_cmd,
        tests_return_code=tests_return_code,
        absolute_md_refs=absolute_refs,
        required_docs_missing=missing_docs,
        docs_checked=docs_checked,
        gaps=gaps,
        notes=config.extra_notes,
        artifacts={
            "json_report": str(json_path),
            "markdown_stocktake": str(md_path),
        },
    )

    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    _write_markdown_summary(result, md_path)
    return result
