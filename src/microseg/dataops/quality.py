"""Dataset quality checks for packaged segmentation datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetQualityConfig:
    """Configuration for dataset QA checks."""

    dataset_dir: str
    output_path: str = "outputs/dataops/dataset_qa_report.json"
    splits: tuple[str, ...] = ("train", "val", "test")
    imbalance_ratio_warn: float = 0.98
    strict: bool = False


@dataclass
class DatasetQualityReport:
    """Summary report for dataset QA checks."""

    schema_version: str
    created_utc: str
    config: dict
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    class_histogram: dict[str, int] = field(default_factory=dict)
    duplicate_files: list[str] = field(default_factory=list)
    dimension_mismatches: list[str] = field(default_factory=list)
    missing_pairs: list[str] = field(default_factory=list)


def _collect_pair_names(images_dir: Path, masks_dir: Path) -> tuple[set[str], set[str]]:
    image_names = {p.name for p in images_dir.glob("*") if p.is_file()}
    mask_names = {p.name for p in masks_dir.glob("*") if p.is_file()}
    return image_names, mask_names


def run_dataset_quality_checks(config: DatasetQualityConfig) -> DatasetQualityReport:
    """Run dataset quality checks and write JSON report."""

    root = Path(config.dataset_dir)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = DatasetQualityReport(
        schema_version="microseg.dataset_qa.v1",
        created_utc=_utc_now(),
        config=asdict(config),
        ok=False,
    )

    if not root.exists():
        report.errors.append(f"dataset root does not exist: {root}")
        output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        if config.strict:
            raise RuntimeError(report.errors[-1])
        return report

    file_hash_to_paths: dict[str, list[str]] = {}
    class_hist: dict[int, int] = {}

    for split in config.splits:
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            report.warnings.append(f"split '{split}' missing images/masks directories")
            report.split_counts[split] = 0
            continue

        image_names, mask_names = _collect_pair_names(images_dir, masks_dir)
        only_images = sorted(image_names - mask_names)
        only_masks = sorted(mask_names - image_names)
        for name in only_images:
            report.missing_pairs.append(f"{split}: missing mask for image {name}")
        for name in only_masks:
            report.missing_pairs.append(f"{split}: missing image for mask {name}")

        common = sorted(image_names & mask_names)
        report.split_counts[split] = len(common)
        for name in common:
            img_path = images_dir / name
            msk_path = masks_dir / name
            file_hash_to_paths.setdefault(_sha256(img_path), []).append(str(img_path))
            file_hash_to_paths.setdefault(_sha256(msk_path), []).append(str(msk_path))

            img = np.asarray(Image.open(img_path))
            msk = np.asarray(Image.open(msk_path))
            if img.ndim == 3:
                h_img, w_img = img.shape[:2]
            else:
                h_img, w_img = img.shape
            if msk.ndim == 3:
                h_msk, w_msk = msk.shape[:2]
            else:
                h_msk, w_msk = msk.shape
            if (h_img, w_img) != (h_msk, w_msk):
                report.dimension_mismatches.append(
                    f"{split}:{name} image({h_img}x{w_img}) != mask({h_msk}x{w_msk})"
                )

            vals, counts = np.unique(msk.reshape(-1), return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                class_hist[int(v)] = class_hist.get(int(v), 0) + int(c)

    duplicates: list[str] = []
    for paths in file_hash_to_paths.values():
        if len(paths) > 1:
            duplicates.extend(paths)
    if duplicates:
        report.duplicate_files = sorted(set(duplicates))
        report.warnings.append(f"duplicate file content detected: {len(report.duplicate_files)} files")

    if class_hist:
        total_pixels = sum(class_hist.values())
        major = max(class_hist.values())
        ratio = float(major / max(1, total_pixels))
        if ratio >= float(config.imbalance_ratio_warn):
            report.warnings.append(
                f"class imbalance detected: dominant class ratio {ratio:.4f} "
                f"(threshold {float(config.imbalance_ratio_warn):.4f})"
            )
        report.class_histogram = {str(k): int(v) for k, v in sorted(class_hist.items())}

    if report.missing_pairs:
        report.errors.append(f"missing image/mask pairs found: {len(report.missing_pairs)}")
    if report.dimension_mismatches:
        report.errors.append(f"image/mask dimension mismatches found: {len(report.dimension_mismatches)}")

    report.ok = len(report.errors) == 0
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    if config.strict and not report.ok:
        raise RuntimeError("dataset QA failed")
    return report
