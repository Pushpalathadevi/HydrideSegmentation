"""CLI for segmentation dataset preparation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pipeline import DatasetPreparer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare segmentation datasets (binary masks).")
    parser.add_argument("--input", "--input-dir", required=True, dest="input_dir")
    parser.add_argument("--output", "--output-dir", "--output-root", required=True, dest="output_dir")
    parser.add_argument("--style", default="oxford,mado", help="Comma-separated styles: oxford,mado")
    parser.add_argument("--config", default=None, help="Optional YAML config. Defaults to configs/data_prep.default.yml when present.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pct", "--train-frac", type=float, default=0.8)
    parser.add_argument("--val-pct", "--val-frac", type=float, default=0.1)
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=100)
    parser.add_argument("--num-debug", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--crop-train", choices=["center", "random"], default="random")
    parser.add_argument("--crop-eval", choices=["center", "random"], default="center")
    parser.add_argument("--mask-r-min", type=int, default=200)
    parser.add_argument("--mask-g-max", type=int, default=60)
    parser.add_argument("--mask-b-max", type=int, default=60)
    parser.add_argument("--allow-red-dominance-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-red-min-fallback", type=int, default=16)
    parser.add_argument("--mask-red-dominance-margin", type=int, default=8)
    parser.add_argument("--mask-red-dominance-ratio", type=float, default=1.5)
    parser.add_argument("--auto-otsu-for-noisy-grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noisy-grayscale-low-max", type=int, default=5)
    parser.add_argument("--noisy-grayscale-high-min", type=int, default=200)
    parser.add_argument("--noisy-grayscale-min-extreme-ratio", type=float, default=0.98)
    parser.add_argument("--empty-mask-action", choices=["warn", "error"], default="warn")
    parser.add_argument("--rgb-mask-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-log-interval", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
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
        "target_size": [args.target_size, args.target_size],
        "resize_policy": "short_side_to_target_crop",
        "crop_mode_train": args.crop_train,
        "crop_mode_eval": args.crop_eval,
        "rgb_mask_mode": bool(args.rgb_mask_mode),
        "mask_r_min": args.mask_r_min,
        "mask_g_max": args.mask_g_max,
        "mask_b_max": args.mask_b_max,
        "allow_red_dominance_fallback": bool(args.allow_red_dominance_fallback),
        "mask_red_min_fallback": args.mask_red_min_fallback,
        "mask_red_dominance_margin": args.mask_red_dominance_margin,
        "mask_red_dominance_ratio": args.mask_red_dominance_ratio,
        "auto_otsu_for_noisy_grayscale": bool(args.auto_otsu_for_noisy_grayscale),
        "noisy_grayscale_low_max": args.noisy_grayscale_low_max,
        "noisy_grayscale_high_min": args.noisy_grayscale_high_min,
        "noisy_grayscale_min_extreme_ratio": args.noisy_grayscale_min_extreme_ratio,
        "empty_mask_action": args.empty_mask_action,
        "progress_log_interval": args.progress_log_interval,
        "debug": {
            "enabled": args.debug,
            "limit_pairs": args.debug_limit,
            "inspection_limit": args.num_debug,
        },
    }
    cfg = DatasetPrepConfig.from_yaml_or_default(resolved_config_path, fallback)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "dataset_prepare.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    result = DatasetPreparer(cfg).run()
    logging.info("Prepared dataset with %s pairs. Manifest: %s", result.total_pairs, result.manifest_path)
    logging.info("Dataset directory for training: %s", Path(cfg.output_dir) / "mado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
