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
    parser.add_argument("--skip-tests", action="store_true", help="Skip full pytest execution")
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
        )
    )

    print(json.dumps({"status": result.status, "artifacts": result.artifacts}, indent=2))
    if args.strict and result.status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
