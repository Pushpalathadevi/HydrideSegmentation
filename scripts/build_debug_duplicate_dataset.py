"""Create tiny split dataset by duplicating one test image for pipeline debug runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _write_pair(
    *,
    images_dir: Path,
    masks_dir: Path,
    stem: str,
    rgb: np.ndarray,
    mask: np.ndarray,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(images_dir / f"{stem}.png")
    Image.fromarray(mask).save(masks_dir / f"{stem}.png")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create tiny duplicated train/val dataset from one image.")
    parser.add_argument("--image-path", type=str, default="test_data/syntheticHydrides.png")
    parser.add_argument("--output-dir", type=str, default="outputs/debug_pretrained_dataset")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Mask threshold on grayscale image (mask=1 where gray > threshold).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"source image not found: {image_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)
    mask = (gray > int(args.threshold)).astype(np.uint8)

    for idx in range(max(1, int(args.train_count))):
        _write_pair(
            images_dir=output_dir / "train" / "images",
            masks_dir=output_dir / "train" / "masks",
            stem=f"train_{idx:03d}",
            rgb=rgb,
            mask=mask,
        )
    for idx in range(max(1, int(args.val_count))):
        _write_pair(
            images_dir=output_dir / "val" / "images",
            masks_dir=output_dir / "val" / "masks",
            stem=f"val_{idx:03d}",
            rgb=rgb,
            mask=mask,
        )

    manifest = {
        "schema_version": "microseg.debug_dataset.v1",
        "source_image": str(image_path),
        "output_dir": str(output_dir),
        "threshold": int(args.threshold),
        "train_count": int(args.train_count),
        "val_count": int(args.val_count),
    }
    (output_dir / "debug_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"debug dataset: {output_dir}")
    print(f"source image: {image_path}")
    print(f"train samples: {int(args.train_count)}")
    print(f"val samples: {int(args.val_count)}")


if __name__ == "__main__":
    main()
