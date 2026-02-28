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
    verify_release_policy: bool = False
    release_policy_path: str = "docs/versioning_and_release_policy.md"
    require_rollback_keywords: bool = False
    rollback_keywords: tuple[str, ...] = ("rollback", "patch", "release")
    deployment_package_dirs: tuple[str, ...] = ()
    verify_deployment_sha256: bool = True


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
    release_policy_path: str = ""
    release_policy_ok: bool = True
    deployment_packages_checked: list[dict[str, object]] = field(default_factory=list)
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


def _resolve_path(raw: str | Path, *, repo_root: Path) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _check_release_policy(
    repo_root: Path,
    *,
    policy_path: str,
    require_rollback_keywords: bool,
    rollback_keywords: tuple[str, ...],
) -> tuple[bool, str, list[str]]:
    resolved = _resolve_path(policy_path, repo_root=repo_root)
    gaps: list[str] = []
    if not resolved.exists():
        return False, str(resolved), [f"release policy doc missing: {resolved}"]
    try:
        text = resolved.read_text(encoding="utf-8").lower()
    except Exception as exc:
        return False, str(resolved), [f"failed to read release policy doc: {exc}"]
    if require_rollback_keywords:
        missing = [kw for kw in rollback_keywords if kw and kw.lower() not in text]
        if missing:
            gaps.append(
                "release policy missing required keywords: "
                + ", ".join(sorted(set(str(kw) for kw in missing)))
            )
    return len(gaps) == 0, str(resolved), gaps


def _check_deployment_packages(
    repo_root: Path,
    *,
    package_dirs: tuple[str, ...],
    verify_sha256: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    from src.microseg.deployment import validate_deployment_package

    checked: list[dict[str, object]] = []
    gaps: list[str] = []
    for raw in package_dirs:
        if not str(raw).strip():
            continue
        resolved = _resolve_path(raw, repo_root=repo_root)
        report = validate_deployment_package(resolved, verify_sha256=verify_sha256)
        row = {
            "package_dir": str(resolved),
            "ok": bool(report.ok),
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "file_count": int(report.file_count),
        }
        checked.append(row)
        if not report.ok:
            reason = "; ".join(report.errors[:3]) if report.errors else "unknown validation error"
            gaps.append(f"deployment package invalid ({resolved}): {reason}")
    return checked, gaps


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
        f"- Release policy check: `{result.release_policy_ok}`",
        f"- Deployment packages checked: `{len(result.deployment_packages_checked)}`",
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
    release_policy_ok = True
    release_policy_path = ""
    if bool(config.verify_release_policy):
        release_policy_ok, release_policy_path, release_policy_gaps = _check_release_policy(
            repo_root,
            policy_path=str(config.release_policy_path),
            require_rollback_keywords=bool(config.require_rollback_keywords),
            rollback_keywords=tuple(str(v) for v in config.rollback_keywords),
        )
    else:
        release_policy_gaps = []
    deployment_packages_checked: list[dict[str, object]] = []
    if config.deployment_package_dirs:
        deployment_packages_checked, deployment_gaps = _check_deployment_packages(
            repo_root,
            package_dirs=tuple(config.deployment_package_dirs),
            verify_sha256=bool(config.verify_deployment_sha256),
        )
    else:
        deployment_gaps = []

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
    gaps.extend(release_policy_gaps)
    gaps.extend(deployment_gaps)

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
        release_policy_path=release_policy_path,
        release_policy_ok=release_policy_ok,
        deployment_packages_checked=deployment_packages_checked,
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
