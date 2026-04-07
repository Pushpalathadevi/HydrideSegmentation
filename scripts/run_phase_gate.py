"""CLI wrapper to execute mandatory phase-gate closeout checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.quality import PhaseGateConfig, run_phase_gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MicroSeg phase-gate checks and stocktake reporting")
    parser.add_argument("--phase-label", required=True, help="Phase label, e.g. 'Phase 8'")
    parser.add_argument("--output-dir", default="outputs/phase_gates", help="Output directory for reports")
    parser.add_argument("--notes", default="", help="Optional phase closeout notes")
    parser.add_argument("--skip-tests", action=argparse.BooleanOptionalAction, default=False, help="Skip full pytest execution")
    parser.add_argument("--verify-release-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release-policy-path", default="docs/versioning_and_release_policy.md")
    parser.add_argument("--require-rollback-keywords", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rollback-keywords", default="rollback,patch,release")
    parser.add_argument("--deployment-package-dirs", default="", help="Comma-separated package directories to validate")
    parser.add_argument("--verify-deployment-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when gate fails")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = run_phase_gate(
        PhaseGateConfig(
            phase_label=str(args.phase_label),
            run_tests=not bool(args.skip_tests),
            output_dir=str(args.output_dir),
            extra_notes=str(args.notes),
            strict=False,
            verify_release_policy=bool(args.verify_release_policy),
            release_policy_path=str(args.release_policy_path),
            require_rollback_keywords=bool(args.require_rollback_keywords),
            rollback_keywords=tuple(
                part.strip()
                for part in str(args.rollback_keywords).split(",")
                if part.strip()
            )
            or ("rollback", "patch", "release"),
            deployment_package_dirs=tuple(
                part.strip()
                for part in str(args.deployment_package_dirs).split(",")
                if part.strip()
            ),
            verify_deployment_sha256=bool(args.verify_deployment_sha256),
        )
    )

    print(json.dumps({"status": result.status, "artifacts": result.artifacts}, indent=2))
    if args.strict and result.status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
