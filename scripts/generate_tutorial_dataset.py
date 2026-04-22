"""Generate the tutorial/smoke paired dataset used by docs and tests."""

from __future__ import annotations

import argparse

from src.microseg.data_preparation.tutorial_dataset import generate_tutorial_paired_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the beginner tutorial paired dataset from a bundled test image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/tutorial_demo/raw_pairs",
        help="Output folder for the generated image+mask pairs.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="test_data/3PB_SRT_data_generation_1817_OD_side1_8.png",
        help="Bundled source image used to generate the tutorial pseudo-mask.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_tutorial_paired_dataset(
        output_dir=args.output_dir,
        image_path=args.image_path,
    )
    print(f"tutorial dataset: {result.output_dir}")
    print(f"pairs: {result.pair_count}")
    print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
