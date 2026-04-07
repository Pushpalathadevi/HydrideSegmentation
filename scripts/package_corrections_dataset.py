#!/usr/bin/env python3
"""Package corrected sample exports into train/val/test dataset layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microseg.corrections import CorrectionDatasetPackager


def _collect_sample_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "correction_record.json").exists()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Package correction exports into dataset splits")
    parser.add_argument("--input-dir", required=True, help="Directory containing corrected sample folders")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    sample_dirs = _collect_sample_dirs(in_dir)
    if not sample_dirs:
        raise RuntimeError("No correction sample folders found")

    packager = CorrectionDatasetPackager(seed=args.seed)
    packaged = packager.package(
        [str(p) for p in sample_dirs],
        str(out_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Packaged dataset at: {packaged}")


if __name__ == "__main__":
    main()
