"""Tutorial/smoke dataset generator for beginner documentation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS
from hydride_segmentation.segmentation_mask_creation import run_model as run_conventional_mask


@dataclass(frozen=True)
class TutorialDatasetResult:
    """Summary of the generated tutorial dataset."""

    output_dir: Path
    manifest_path: Path
    source_image: Path
    pair_count: int


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[3] / path).resolve()


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8)).convert("L")


def generate_tutorial_paired_dataset(
    *,
    output_dir: str | Path,
    image_path: str | Path = "test_data/3PB_SRT_data_generation_1817_OD_side1_8.png",
) -> TutorialDatasetResult:
    """Generate a small paired dataset for docs and smoke tests."""

    resolved_output_dir = _resolve_repo_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_image_path = _resolve_repo_path(image_path)
    if not resolved_image_path.exists():
        raise FileNotFoundError(f"tutorial source image not found: {resolved_image_path}")

    base_image = Image.open(resolved_image_path).convert("RGB")
    grayscale_source, base_mask = run_conventional_mask(str(resolved_image_path), DEFAULT_CONVENTIONAL_PARAMS)
    if base_mask.ndim != 2:
        raise ValueError(f"conventional mask must be 2D, got shape={base_mask.shape}")
    if grayscale_source.ndim != 2:
        raise ValueError(f"conventional source image must be 2D, got shape={grayscale_source.shape}")

    base_rgb = Image.fromarray(np.stack([grayscale_source, grayscale_source, grayscale_source], axis=-1)).convert("RGB")
    mask_img = _mask_image(base_mask)

    families: dict[str, list[tuple[str, Image.Image, Image.Image]]] = {
        "hydride_plate_family_a": [
            ("aug00", base_rgb, mask_img),
            ("aug01", ImageOps.mirror(base_rgb), ImageOps.mirror(mask_img)),
        ],
        "hydride_plate_family_b": [
            ("aug00", base_rgb.rotate(90, expand=True), mask_img.rotate(90, expand=True)),
            ("aug01", base_rgb.rotate(270, expand=True), mask_img.rotate(270, expand=True)),
        ],
        "hydride_plate_family_c": [
            ("aug00", ImageOps.flip(base_rgb), ImageOps.flip(mask_img)),
            ("aug01", ImageOps.flip(ImageOps.mirror(base_rgb)), ImageOps.flip(ImageOps.mirror(mask_img))),
        ],
        "hydride_plate_family_d": [
            ("aug00", ImageEnhance.Brightness(base_image).enhance(0.92), mask_img),
            ("aug01", ImageEnhance.Brightness(base_image).enhance(1.08), mask_img),
        ],
        "hydride_plate_family_e": [
            ("aug00", ImageEnhance.Contrast(base_image).enhance(1.15), mask_img),
            ("aug01", ImageEnhance.Sharpness(base_image).enhance(1.35), mask_img),
        ],
        "hydride_plate_family_f": [
            ("aug00", ImageOps.autocontrast(base_image), mask_img),
            ("aug01", ImageEnhance.Color(base_image).enhance(0.0).convert("RGB"), mask_img),
        ],
    }

    manifest_rows: list[dict[str, object]] = []
    for family_name, variants in families.items():
        for variant_name, image, mask in variants:
            stem = f"{family_name}_{variant_name}"
            image_path_out = resolved_output_dir / f"{stem}.png"
            mask_path_out = resolved_output_dir / f"{stem}_mask.png"
            image.save(image_path_out)
            mask.save(mask_path_out)
            manifest_rows.append(
                {
                    "stem": stem,
                    "family": family_name,
                    "variant": variant_name,
                    "image_path": str(image_path_out),
                    "mask_path": str(mask_path_out),
                }
            )

    manifest_path = resolved_output_dir / "tutorial_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "microseg.tutorial_dataset.v1",
                "source_image": str(resolved_image_path),
                "pair_count": len(manifest_rows),
                "families": sorted(families.keys()),
                "pairs": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TutorialDatasetResult(
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        source_image=resolved_image_path,
        pair_count=len(manifest_rows),
    )
