"""Generate the constant-area-fraction synthetic benchmark for HCI validation."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src.microseg.evaluation.continuity.synthetic import (
    SyntheticBenchmarkConfig,
    generate_benchmark,
    render_overview,
    render_overview_from_manifest,
    write_benchmark,
)
from src.microseg.io.configuration import resolve_config

DEFAULT_CONFIG = "configs/hci_synthetic_benchmark.default.yml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--set", action="append", default=[], help="Dotted-key override, e.g. benchmark.replicates=3")
    return parser


def _code_version() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = resolve_config(args.config, args.set)
    cfg = SyntheticBenchmarkConfig.from_mapping(payload.get("benchmark", {}))
    out_dir = Path(payload.get("output_dir", "test_data/hci_synthetic_v1"))
    start = time.monotonic()

    def log(line: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] +{time.monotonic() - start:6.1f}s {line}", flush=True)

    samples = generate_benchmark(cfg, progress=log)
    manifest = write_benchmark(
        samples,
        out_dir,
        cfg,
        code_version=_code_version(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    log(f"wrote {len(samples)} samples; manifest {manifest}")
    if payload.get("render_overview", True):
        log(f"overview {render_overview(samples, out_dir / 'overview.png')}")
    if payload.get("docs_figure"):
        log(f"docs figure {render_overview_from_manifest(out_dir, payload['docs_figure'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
