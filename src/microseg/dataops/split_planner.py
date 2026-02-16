"""Leakage-aware deterministic split planning for correction exports."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_stem(path_text: str) -> str:
    text = str(path_text).strip()
    if not text:
        return ""
    return Path(text).stem


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CorrectionSample:
    """Sample descriptor parsed from correction export folder."""

    sample_id: str
    source_group: str
    root_dir: str
    image_path: str
    mask_path: str
    metadata_path: str


@dataclass(frozen=True)
class CorrectionSplitConfig:
    """Configuration for correction split planning and materialization."""

    input_dir: str
    output_dir: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    leakage_group: Literal["source_stem", "sample_id"] = "source_stem"


@dataclass
class CorrectionSplitResult:
    """Result summary for split planning and export."""

    schema_version: str
    created_utc: str
    config: dict
    total_samples: int
    leakage_groups: int
    split_counts: dict[str, int] = field(default_factory=dict)
    split_group_counts: dict[str, int] = field(default_factory=dict)
    output_dir: str = ""
    manifest_path: str = ""


def _collect_correction_samples(input_dir: Path, leakage_group: str) -> list[CorrectionSample]:
    samples: list[CorrectionSample] = []
    for folder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        rec_path = folder / "correction_record.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        sample_id = str(rec.get("sample_id", folder.name))
        files = rec.get("files", {})
        image_rel = files.get("input", "input.png")
        mask_key = "corrected_mask_indexed" if "corrected_mask_indexed" in files else "corrected_mask"
        if mask_key not in files:
            raise KeyError(f"missing corrected mask entry in {rec_path}")
        mask_rel = files[mask_key]

        image_path = folder / image_rel
        mask_path = folder / mask_rel
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"missing image/mask artifact for sample {sample_id}")

        if leakage_group == "source_stem":
            source_stem = _safe_stem(str(rec.get("source_image_path", "")))
            group = source_stem or sample_id
        else:
            group = sample_id

        samples.append(
            CorrectionSample(
                sample_id=sample_id,
                source_group=group,
                root_dir=str(folder),
                image_path=str(image_path),
                mask_path=str(mask_path),
                metadata_path=str(rec_path),
            )
        )
    if not samples:
        raise RuntimeError(f"no correction export sample folders found under {input_dir}")
    return samples


def _planned_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    n_train = min(n, int(round(n * train_ratio)))
    n_val = min(max(0, n - n_train), int(round(n * val_ratio)))
    return n_train, n_val


def _assign_groups(
    samples: list[CorrectionSample],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    grouped: dict[str, list[CorrectionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_group, []).append(sample)

    groups = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(grouped[g]), reverse=True)

    target_train, target_val = _planned_counts(len(samples), train_ratio, val_ratio)
    target_test = len(samples) - target_train - target_val
    current = {"train": 0, "val": 0, "test": 0}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    assignments: dict[str, str] = {}
    for group in groups:
        size = len(grouped[group])
        deficits = {split: targets[split] - current[split] for split in ["train", "val", "test"]}
        best_split = max(deficits.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if deficits[best_split] < 0:
            best_split = min(current.items(), key=lambda kv: kv[1])[0]
        assignments[group] = best_split
        current[best_split] += size
    return assignments


def _copy_to_dataset(
    samples: list[CorrectionSample],
    group_to_split: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    sample_to_split: dict[str, str] = {}

    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "metadata").mkdir(parents=True, exist_ok=True)

    for sample in samples:
        split = group_to_split[sample.source_group]
        root = output_dir / split
        shutil.copy2(sample.image_path, root / "images" / f"{sample.sample_id}.png")
        shutil.copy2(sample.mask_path, root / "masks" / f"{sample.sample_id}.png")
        shutil.copy2(sample.metadata_path, root / "metadata" / f"{sample.sample_id}.json")
        split_counts[split] += 1
        split_groups[split].add(sample.source_group)
        sample_to_split[sample.sample_id] = split

    split_group_counts = {split: len(groups) for split, groups in split_groups.items()}
    return split_counts, split_group_counts, sample_to_split


def plan_and_materialize_correction_splits(config: CorrectionSplitConfig) -> CorrectionSplitResult:
    """Build leakage-aware train/val/test splits from correction exports."""

    if config.train_ratio <= 0 or config.val_ratio < 0 or config.train_ratio + config.val_ratio >= 1:
        raise ValueError("invalid split ratios")

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_correction_samples(input_dir, config.leakage_group)
    group_to_split = _assign_groups(
        samples,
        train_ratio=float(config.train_ratio),
        val_ratio=float(config.val_ratio),
        seed=int(config.seed),
    )
    split_counts, split_group_counts, sample_to_split = _copy_to_dataset(samples, group_to_split, output_dir)

    manifest = {
        "schema_version": "microseg.dataset_split_manifest.v1",
        "created_utc": _utc_now(),
        "config": asdict(config),
        "total_samples": len(samples),
        "leakage_groups": len(group_to_split),
        "split_counts": split_counts,
        "split_group_counts": split_group_counts,
        "group_to_split": group_to_split,
        "sample_to_split": sample_to_split,
        "sample_hashes": {
            sample.sample_id: {
                "image_sha256": _file_sha256(Path(sample.image_path)),
                "mask_sha256": _file_sha256(Path(sample.mask_path)),
            }
            for sample in samples
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return CorrectionSplitResult(
        schema_version="microseg.dataset_split_result.v1",
        created_utc=_utc_now(),
        config=asdict(config),
        total_samples=len(samples),
        leakage_groups=len(group_to_split),
        split_counts=split_counts,
        split_group_counts=split_group_counts,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
    )
