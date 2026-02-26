"""CLI for segmentation dataset preparation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pipeline import DatasetPreparer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare segmentation datasets (binary masks).")
    parser.add_argument("--input", required=True, dest="input_dir")
    parser.add_argument("--output", required=True, dest="output_dir")
    parser.add_argument("--style", default="oxford,mado", help="Comma-separated styles: oxford,mado")
    parser.add_argument("--config", default=None, help="Optional YAML config. Defaults to configs/data_prep.default.yml when present.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pct", type=float, default=0.8)
    parser.add_argument("--val-pct", type=float, default=0.1)
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=100)
    parser.add_argument("--num-debug", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    default_config_path = Path(__file__).resolve().parents[3] / "configs" / "data_prep.default.yml"
    resolved_config_path: str | None = args.config
    if resolved_config_path is None and default_config_path.exists():
        resolved_config_path = str(default_config_path)
        logging.info("Using default prep config: %s", resolved_config_path)

    fallback = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "styles": [part.strip() for part in args.style.split(",") if part.strip()],
        "dry_run": args.dry_run,
        "seed": args.seed,
        "train_pct": args.train_pct,
        "val_pct": args.val_pct,
        "skip_sanity": args.skip_sanity,
        "debug": {
            "enabled": args.debug,
            "limit_pairs": args.debug_limit,
            "inspection_limit": args.num_debug,
        },
    }
    cfg = DatasetPrepConfig.from_yaml_or_default(resolved_config_path, fallback)
    result = DatasetPreparer(cfg).run()
    logging.info("Prepared dataset with %s pairs. Manifest: %s", result.total_pairs, result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
