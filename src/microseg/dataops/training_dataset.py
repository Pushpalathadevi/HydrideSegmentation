"""Training dataset layout preparation and auto-splitting utilities."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.microseg.corrections.classes import normalize_binary_index_mask
from src.microseg.corrections.classes import class_map_to_colormap, resolve_class_map
from src.microseg.data_preparation.augmentation import (
    AugmentationConfig,
    AugmentationDebugWriter,
    AugmentationRunner,
)


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_AUGMENTATION_SUFFIX_PATTERNS = [
    re.compile(r"(.+?)(?:[_-](?:aug|crop|tile|patch|view|variant)[_-]?\d+)$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-](?:flip|hflip|vflip|hvflip))$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-]rot(?:ation)?[_-]?\d+)$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-]r\d+)$", flags=re.IGNORECASE),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _collect_supported_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if _is_supported_image(p)])


def _count_pairs(split_dir: Path) -> int:
    images = split_dir / "images"
    masks = split_dir / "masks"
    if not images.exists() or not masks.exists():
        return 0
    image_names = {p.name for p in _collect_supported_files(images)}
    mask_names = {p.name for p in _collect_supported_files(masks)}
    return len(image_names & mask_names)


def _has_explicit_split_layout(root: Path) -> bool:
    required = [
        root / "train" / "images",
        root / "train" / "masks",
        root / "val" / "images",
        root / "val" / "masks",
        root / "test" / "images",
        root / "test" / "masks",
    ]
    return all(p.exists() for p in required)


def _find_unsplit_dirs(root: Path) -> tuple[Path, Path] | None:
    candidates = [
        (root / "source", root / "masks"),
        (root / "images", root / "masks"),
        (root / "data" / "source", root / "data" / "masks"),
        (root / "data" / "images", root / "data" / "masks"),
    ]
    for images_dir, masks_dir in candidates:
        if images_dir.exists() and masks_dir.exists():
            return images_dir, masks_dir
    return None


def generate_dataset_split_manifest_from_splits(
    dataset_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Generate ``dataset_manifest.json`` from existing train/val/test folders.

    Parameters
    ----------
    dataset_dir:
        Dataset root containing ``train/``, ``val/``, and ``test/`` split folders.
    output_path:
        Optional manifest output path. Defaults to ``<dataset_dir>/dataset_manifest.json``.

    Returns
    -------
    Path
        Path to the written manifest file.
    """

    root = Path(dataset_dir)
    if not _has_explicit_split_layout(root):
        raise FileNotFoundError(
            f"dataset layout at {root} does not contain train/val/test images+masks folders"
        )

    rows: list[tuple[str, str, Path, Path]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in ["train", "val", "test"]:
        split_pairs = _pairs_by_stem(root / split / "images", root / split / "masks")
        split_counts[split] = len(split_pairs)
        for stem, image_path, mask_path in split_pairs:
            rows.append((split, stem, image_path, mask_path))

    stem_counts = Counter(stem for _split, stem, _img, _msk in rows)
    sample_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    split_group_counts = {"train": 0, "val": 0, "test": 0}
    sample_hashes: dict[str, dict[str, str]] = {}

    for split, stem, image_path, mask_path in rows:
        sample_id = stem if stem_counts[stem] == 1 else f"{split}/{stem}"
        sample_to_split[sample_id] = split
        group_to_split[sample_id] = split
        split_group_counts[split] += 1
        sample_hashes[sample_id] = {
            "image_sha256": _file_sha256(image_path),
            "mask_sha256": _file_sha256(mask_path),
        }

    manifest = {
        "schema_version": "microseg.dataset_split_manifest.v1",
        "created_utc": _utc_now(),
        "config": {
            "dataset_dir": str(root),
            "source_layout": "split_layout",
            "generated_by": "generate_dataset_split_manifest_from_splits",
        },
        "total_samples": int(sum(split_counts.values())),
        "leakage_groups": int(sum(split_group_counts.values())),
        "split_counts": split_counts,
        "split_group_counts": split_group_counts,
        "group_to_split": group_to_split,
        "sample_to_split": sample_to_split,
        "sample_hashes": sample_hashes,
    }

    manifest_path = Path(output_path) if output_path is not None else (root / "dataset_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


@dataclass(frozen=True)
class DatasetPrepareConfig:
    """Configuration for training dataset layout preparation."""

    dataset_dir: str
    output_dir: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    id_width: int = 6
    split_strategy: Literal["leakage_aware", "random"] = "leakage_aware"
    leakage_group_mode: Literal["suffix_aware", "stem", "regex"] = "suffix_aware"
    leakage_group_regex: str = ""
    mask_input_type: Literal["indexed", "rgb_colormap", "auto"] = "indexed"
    mask_colormap: dict[str, object] = field(default_factory=dict)
    mask_colormap_strict: bool = True
    binary_mask_normalization: Literal["off", "two_value_zero_background", "nonzero_foreground"] = "off"
    class_map_path: str = ""
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class DatasetPrepareResult:
    """Result summary for dataset preparation."""

    schema_version: str
    created_utc: str
    dataset_dir: str
    output_dir: str
    used_existing_splits: bool
    prepared: bool
    split_counts: dict[str, int] = field(default_factory=dict)
    source_layout: str = ""
    manifest_path: str = ""


@dataclass
class DatasetPreparePreview:
    """Preview summary for dataset preparation without writing output files."""

    schema_version: str
    created_utc: str
    dataset_dir: str
    output_dir: str
    source_layout: str
    used_existing_splits: bool
    split_counts: dict[str, int] = field(default_factory=dict)
    total_pairs: int = 0
    leakage_groups: int = 0
    class_histogram: dict[str, int] = field(default_factory=dict)
    mapping: list[dict[str, object]] = field(default_factory=list)


def _pairs_by_stem(images_dir: Path, masks_dir: Path) -> list[tuple[str, Path, Path]]:
    images = _collect_supported_files(images_dir)
    masks = _collect_supported_files(masks_dir)
    image_by_stem: dict[str, Path] = {}
    mask_by_stem: dict[str, Path] = {}

    for p in images:
        if p.stem in image_by_stem:
            raise ValueError(f"duplicate image stem detected: {p.stem}")
        image_by_stem[p.stem] = p
    for p in masks:
        if p.stem in mask_by_stem:
            raise ValueError(f"duplicate mask stem detected: {p.stem}")
        mask_by_stem[p.stem] = p

    common = sorted(set(image_by_stem.keys()) & set(mask_by_stem.keys()))
    if not common:
        raise RuntimeError(f"no matching image/mask pairs found under {images_dir} and {masks_dir}")

    missing_images = sorted(set(mask_by_stem.keys()) - set(image_by_stem.keys()))
    missing_masks = sorted(set(image_by_stem.keys()) - set(mask_by_stem.keys()))
    if missing_images or missing_masks:
        problems = []
        if missing_images:
            problems.append(f"masks without images: {missing_images[:5]}")
        if missing_masks:
            problems.append(f"images without masks: {missing_masks[:5]}")
        raise ValueError("; ".join(problems))

    return [(stem, image_by_stem[stem], mask_by_stem[stem]) for stem in common]


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("invalid split ratios")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
    if n_train + n_val + n_test != n:
        n_train = min(n, n_train)
        n_val = min(max(0, n - n_train), n_val)
        n_test = max(0, n - n_train - n_val)
    return n_train, n_val, n_test


def _parse_rgb_text(text: str) -> tuple[int, int, int]:
    value = text.strip()
    if value.startswith("#"):
        if len(value) != 7:
            raise ValueError(f"invalid hex color '{text}'")
        try:
            return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
        except ValueError as exc:
            raise ValueError(f"invalid hex color '{text}'") from exc

    cleaned = value.strip("()[]")
    parts = [part.strip() for part in re.split(r"[,\s]+", cleaned) if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"RGB color must have 3 channels, got '{text}'")
    channels: list[int] = []
    for part in parts:
        try:
            channel = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid RGB channel '{part}' in '{text}'") from exc
        if channel < 0 or channel > 255:
            raise ValueError(f"RGB channel out of range in '{text}'")
        channels.append(channel)
    return (channels[0], channels[1], channels[2])


def _parse_index(value: object, *, context: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid class index for {context}: {value!r}") from exc
    if index < 0 or index > 255:
        raise ValueError(f"class index out of uint8 range for {context}: {index}")
    return index


def _normalize_rgb_colormap(raw: dict[str, object]) -> dict[tuple[int, int, int], int]:
    if not raw:
        raise ValueError("mask_colormap is required when mask_input_type is rgb_colormap")

    mapping: dict[tuple[int, int, int], int] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("empty key in mask_colormap")

        as_color_key = False
        if key_text.startswith("#"):
            as_color_key = True
        if "," in key_text or " " in key_text:
            as_color_key = True
        if key_text.startswith("(") or key_text.startswith("["):
            as_color_key = True

        if as_color_key:
            rgb = _parse_rgb_text(key_text)
            index = _parse_index(value, context=f"mask_colormap[{key_text}]")
        else:
            index = _parse_index(key_text, context=f"mask_colormap key '{key_text}'")
            if isinstance(value, str):
                rgb = _parse_rgb_text(value)
            elif isinstance(value, (list, tuple)):
                if len(value) != 3:
                    raise ValueError(f"mask_colormap[{key_text}] must be RGB triplet")
                rgb = (
                    _parse_index(value[0], context=f"mask_colormap[{key_text}][0]"),
                    _parse_index(value[1], context=f"mask_colormap[{key_text}][1]"),
                    _parse_index(value[2], context=f"mask_colormap[{key_text}][2]"),
                )
            else:
                raise ValueError(
                    f"mask_colormap[{key_text}] must be RGB triplet list/tuple/string, got {type(value).__name__}"
                )

        if rgb in mapping and mapping[rgb] != index:
            raise ValueError(
                f"conflicting class indices for RGB color {rgb}: {mapping[rgb]} vs {index}"
            )
        mapping[rgb] = index
    return mapping


def _rgb_mask_to_index(
    mask_rgb: np.ndarray,
    *,
    colormap: dict[tuple[int, int, int], int],
    strict: bool,
    mask_path: Path,
) -> np.ndarray:
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"RGB mask conversion requires 3-channel mask: {mask_path} (shape={mask_rgb.shape})")

    packed = (
        (mask_rgb[:, :, 0].astype(np.uint32) << 16)
        | (mask_rgb[:, :, 1].astype(np.uint32) << 8)
        | mask_rgb[:, :, 2].astype(np.uint32)
    )
    unique_colors = np.unique(packed)
    packed_to_index = {
        ((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]): int(idx)
        for rgb, idx in colormap.items()
    }
    missing = [int(color) for color in unique_colors if int(color) not in packed_to_index]
    if missing and strict:
        unknown_preview = []
        for packed_color in missing[:8]:
            r = (packed_color >> 16) & 0xFF
            g = (packed_color >> 8) & 0xFF
            b = packed_color & 0xFF
            unknown_preview.append(f"({r},{g},{b})")
        raise ValueError(
            f"RGB mask contains colors not present in mask_colormap for {mask_path}: {', '.join(unknown_preview)}"
        )

    out = np.zeros(packed.shape, dtype=np.uint8)
    for packed_color, class_index in packed_to_index.items():
        out[packed == packed_color] = np.uint8(class_index)
    return out


def _mask_to_index(mask_path: Path, config: DatasetPrepareConfig) -> np.ndarray:
    mask_raw = np.asarray(Image.open(mask_path))
    if mask_raw.ndim == 2:
        if config.mask_input_type == "rgb_colormap":
            raise ValueError(
                f"mask_input_type=rgb_colormap expects RGB masks, got 2D mask for {mask_path}"
            )
        return normalize_binary_index_mask(
            mask_raw.astype(np.uint8),
            mode=str(config.binary_mask_normalization),
        )

    if mask_raw.ndim == 3:
        if mask_raw.shape[2] == 4:
            mask_raw = mask_raw[:, :, :3]
        if mask_raw.shape[2] != 3:
            raise ValueError(f"unsupported mask channel count for {mask_path}: {mask_raw.shape}")
        if config.mask_input_type not in {"rgb_colormap", "auto"}:
            raise ValueError(
                f"RGB mask provided but mask_input_type is '{config.mask_input_type}'. "
                "Set mask_input_type=rgb_colormap and provide mask_colormap."
            )
        if config.mask_colormap:
            colormap = _normalize_rgb_colormap(config.mask_colormap)
        else:
            class_map, _ = resolve_class_map(config.class_map_path)
            colormap = _normalize_rgb_colormap(class_map_to_colormap(class_map))
        return _rgb_mask_to_index(
            mask_raw.astype(np.uint8),
            colormap=colormap,
            strict=bool(config.mask_colormap_strict),
            mask_path=mask_path,
        )

    raise ValueError(f"unsupported mask shape for {mask_path}: {mask_raw.shape}")


def _write_png_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path, config: DatasetPrepareConfig) -> None:
    image = Image.open(image_path).convert("RGB")
    mask_idx = _mask_to_index(mask_path, config)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image)
    Image.fromarray(mask_idx).save(out_mask)


def _load_png_pair(
    image_path: Path,
    mask_path: Path,
    config: DatasetPrepareConfig,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask_idx = _mask_to_index(mask_path, config)
    return image, mask_idx


def _write_png_arrays(image: np.ndarray, mask_idx: np.ndarray, out_image: Path, out_mask: Path) -> None:
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(out_image)
    Image.fromarray(mask_idx.astype(np.uint8)).save(out_mask)


def _resolved_augmentation_stage(config: DatasetPrepareConfig) -> str:
    return f"{config.augmentation.stage}:source_native"


def _serialize_augmentation_config(config: DatasetPrepareConfig) -> dict[str, Any]:
    return config.augmentation.to_dict()


def _debug_aug_name(new_name: str) -> str:
    stem = Path(new_name).stem
    return stem.replace(".", "_")


def _derive_source_group(stem: str, *, mode: str, regex: str) -> str:
    if mode == "stem":
        return stem
    if mode == "regex":
        if not regex.strip():
            raise ValueError("leakage_group_regex is required when leakage_group_mode=regex")
        m = re.search(regex, stem)
        if m is None:
            return stem
        if m.groups():
            candidate = str(m.group(1)).strip()
        else:
            candidate = str(m.group(0)).strip()
        return candidate or stem

    base = stem
    changed = True
    while changed:
        changed = False
        for pattern in _AUGMENTATION_SUFFIX_PATTERNS:
            match = pattern.fullmatch(base)
            if match:
                next_base = str(match.group(1)).strip(" _-")
                if next_base and next_base != base:
                    base = next_base
                    changed = True
    return base or stem


def _plan_split_assignments(
    pairs: list[tuple[str, Path, Path]],
    config: DatasetPrepareConfig,
) -> tuple[dict[int, str], dict[str, str], dict[int, str]]:
    n_train, n_val, n_test = _split_counts(
        len(pairs),
        float(config.train_ratio),
        float(config.val_ratio),
        float(config.test_ratio),
    )
    targets = {"train": n_train, "val": n_val, "test": n_test}

    grouped_indices: dict[str, list[int]] = {}
    source_group_for_index: dict[int, str] = {}
    if config.split_strategy == "random":
        for idx, (stem, _img, _msk) in enumerate(pairs):
            group = f"{stem}#{idx + 1}"
            grouped_indices[group] = [idx]
            source_group_for_index[idx] = group
    elif config.split_strategy == "leakage_aware":
        for idx, (stem, _img, _msk) in enumerate(pairs):
            group = _derive_source_group(
                stem,
                mode=config.leakage_group_mode,
                regex=config.leakage_group_regex,
            )
            grouped_indices.setdefault(group, []).append(idx)
            source_group_for_index[idx] = group
    else:
        raise ValueError(f"unsupported split_strategy: {config.split_strategy}")

    groups = list(grouped_indices.keys())
    rng = random.Random(int(config.seed))
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(grouped_indices[g]), reverse=True)

    current = {"train": 0, "val": 0, "test": 0}
    group_to_split: dict[str, str] = {}
    for group in groups:
        size = len(grouped_indices[group])
        deficits = {split: targets[split] - current[split] for split in ["train", "val", "test"]}
        best_split = max(deficits.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if deficits[best_split] < 0:
            best_split = min(current.items(), key=lambda kv: kv[1])[0]
        group_to_split[group] = best_split
        current[best_split] += size

    split_by_pair_index: dict[int, str] = {}
    for idx in range(len(pairs)):
        split_by_pair_index[idx] = group_to_split[source_group_for_index[idx]]
    return split_by_pair_index, group_to_split, source_group_for_index


def _mask_histogram(pairs: list[tuple[str, Path, Path]], config: DatasetPrepareConfig) -> dict[str, int]:
    hist: dict[int, int] = {}
    for _stem, _img_path, mask_path in pairs:
        mask_idx = _mask_to_index(mask_path, config)
        vals, counts = np.unique(mask_idx.reshape(-1), return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            hist[int(v)] = hist.get(int(v), 0) + int(c)
    return {str(k): int(v) for k, v in sorted(hist.items())}


def _append_augmented_preview_rows(
    mapping: list[dict[str, object]],
    *,
    split: str,
    stem: str,
    image_path: Path,
    mask_path: Path,
    global_id: str,
    source_group: str,
    augmentation_config: AugmentationConfig,
) -> int:
    if not augmentation_config.enabled:
        return 0
    if split not in set(augmentation_config.apply_splits):
        return 0
    if augmentation_config.variants_per_sample <= 0:
        return 0
    added = 0
    for variant_index in range(1, int(augmentation_config.variants_per_sample) + 1):
        mapping.append(
            {
                "id": f"{global_id}_aug{variant_index:03d}",
                "global_id": global_id,
                "original_stem": stem,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path),
                "source_group": source_group,
                "new_name": f"{stem}_{global_id}_aug{variant_index:03d}.png",
                "split": split,
                "is_augmented": True,
                "augmentation_variant_index": variant_index,
            }
        )
        added += 1
    return added


def preview_training_dataset_layout(config: DatasetPrepareConfig) -> DatasetPreparePreview:
    """Preview dataset preparation plan and class statistics without materialization."""

    root = Path(config.dataset_dir)

    if _has_explicit_split_layout(root):
        split_counts: dict[str, int] = {}
        mapping: list[dict[str, object]] = []
        class_hist: dict[str, int] = {}
        running_id = 0
        for split in ["train", "val", "test"]:
            images_dir = root / split / "images"
            masks_dir = root / split / "masks"
            if not images_dir.exists() or not masks_dir.exists():
                split_counts[split] = 0
                continue
            split_pairs = _pairs_by_stem(images_dir, masks_dir)
            split_counts[split] = len(split_pairs)
            split_hist = _mask_histogram(split_pairs, config)
            for key, value in split_hist.items():
                class_hist[key] = class_hist.get(key, 0) + value
            for stem, image_path, mask_path in split_pairs:
                running_id += 1
                global_id = f"{running_id:0{int(config.id_width)}d}"
                mapping.append(
                    {
                        "id": global_id,
                        "global_id": global_id,
                        "original_stem": stem,
                        "original_image_path": str(image_path),
                        "original_mask_path": str(mask_path),
                        "source_group": stem,
                        "new_name": f"{stem}_{global_id}.png",
                        "split": split,
                        "is_augmented": False,
                    }
                )
                split_counts[split] += _append_augmented_preview_rows(
                    mapping,
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    global_id=global_id,
                    source_group=stem,
                    augmentation_config=config.augmentation,
                )
        return DatasetPreparePreview(
            schema_version="microseg.dataset_prepare_preview.v1",
            created_utc=_utc_now(),
            dataset_dir=str(root),
            output_dir=str(Path(config.output_dir)),
            source_layout="split_layout",
            used_existing_splits=True,
            split_counts=split_counts,
            total_pairs=int(sum(split_counts.values())),
            leakage_groups=0,
            class_histogram={str(k): int(v) for k, v in sorted(class_hist.items(), key=lambda kv: int(kv[0]))},
            mapping=mapping,
        )

    pair_dirs = _find_unsplit_dirs(root)
    if pair_dirs is None:
        raise FileNotFoundError(
            f"dataset layout not recognized at {root}. "
            "Expected either explicit split folders or unsplit source/masks folders."
        )
    images_dir, masks_dir = pair_dirs
    pairs = _pairs_by_stem(images_dir, masks_dir)
    split_by_pair_index, group_to_split, source_group_for_index = _plan_split_assignments(pairs, config)
    split_counts = {"train": 0, "val": 0, "test": 0}
    mapping: list[dict[str, object]] = []
    for idx, (stem, image_path, mask_path) in enumerate(pairs, start=1):
        split = split_by_pair_index[idx - 1]
        global_id = f"{idx:0{int(config.id_width)}d}"
        split_counts[split] += 1
        mapping.append(
            {
                "id": global_id,
                "global_id": global_id,
                "original_stem": stem,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path),
                "source_group": source_group_for_index[idx - 1],
                "new_name": f"{stem}_{global_id}.png",
                "split": split,
                "is_augmented": False,
            }
        )
        split_counts[split] += _append_augmented_preview_rows(
            mapping,
            split=split,
            stem=stem,
            image_path=image_path,
            mask_path=mask_path,
            global_id=global_id,
            source_group=source_group_for_index[idx - 1],
            augmentation_config=config.augmentation,
        )

    return DatasetPreparePreview(
        schema_version="microseg.dataset_prepare_preview.v1",
        created_utc=_utc_now(),
        dataset_dir=str(root),
        output_dir=str(Path(config.output_dir)),
        source_layout=f"unsplit:{images_dir.relative_to(root)}+{masks_dir.relative_to(root)}",
        used_existing_splits=False,
        split_counts=split_counts,
        total_pairs=len(pairs),
        leakage_groups=len(group_to_split),
        class_histogram=_mask_histogram(pairs, config),
        mapping=mapping,
    )


def prepare_training_dataset_layout(config: DatasetPrepareConfig) -> DatasetPrepareResult:
    """Prepare training dataset layout, auto-splitting when split folders are absent."""

    root = Path(config.dataset_dir)
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    aug_runner = AugmentationRunner(config.augmentation)
    aug_debug_writer = AugmentationDebugWriter()
    aug_debug_written = 0
    augmentation_summary = {
        "enabled": bool(config.augmentation.enabled),
        "requested_stage": config.augmentation.stage,
        "resolved_stage": _resolved_augmentation_stage(config),
        "apply_splits": list(config.augmentation.apply_splits),
        "variants_per_sample": int(config.augmentation.variants_per_sample),
        "operations": [op.name for op in config.augmentation.operations if op.enabled],
        "generated_samples": 0,
        "debug_samples_written": 0,
    }

    if _has_explicit_split_layout(root):
        if not config.augmentation.enabled:
            return DatasetPrepareResult(
                schema_version="microseg.dataset_prepare.v1",
                created_utc=_utc_now(),
                dataset_dir=str(root),
                output_dir=str(out_root),
                used_existing_splits=True,
                prepared=False,
                split_counts={
                    "train": _count_pairs(root / "train"),
                    "val": _count_pairs(root / "val"),
                    "test": _count_pairs(root / "test"),
                },
                source_layout="split_layout",
                manifest_path="",
            )

        mapping: list[dict[str, Any]] = []
        split_counts = {"train": 0, "val": 0, "test": 0}
        group_to_split: dict[str, str] = {}
        running_id = 0
        for split in ["train", "val", "test"]:
            split_pairs = _pairs_by_stem(root / split / "images", root / split / "masks")
            for stem, image_path, mask_path in split_pairs:
                running_id += 1
                global_id = f"{running_id:0{int(config.id_width)}d}"
                group_to_split[stem] = split
                image_arr, mask_idx = _load_png_pair(image_path, mask_path, config)
                new_name = f"{stem}_{global_id}.png"
                out_img = out_root / split / "images" / new_name
                out_msk = out_root / split / "masks" / new_name
                _write_png_arrays(image_arr, mask_idx, out_img, out_msk)
                split_counts[split] += 1
                mapping.append(
                    {
                        "id": global_id,
                        "global_id": global_id,
                        "original_stem": stem,
                        "original_image_path": str(image_path),
                        "original_mask_path": str(mask_path),
                        "source_group": stem,
                        "new_name": new_name,
                        "split": split,
                        "is_augmented": False,
                    }
                )

                variants = aug_runner.generate_variants(
                    image=image_arr,
                    mask=mask_idx,
                    split=split,
                    source_name=new_name,
                    resolved_stage=_resolved_augmentation_stage(config),
                )
                for variant in variants:
                    aug_name = f"{Path(new_name).stem}_aug{variant.metadata.variant_index:03d}.png"
                    out_aug_img = out_root / split / "images" / aug_name
                    out_aug_msk = out_root / split / "masks" / aug_name
                    _write_png_arrays(variant.image, variant.mask, out_aug_img, out_aug_msk)
                    split_counts[split] += 1
                    augmentation_summary["generated_samples"] += 1
                    mapping.append(
                        {
                            "id": f"{global_id}_aug{variant.metadata.variant_index:03d}",
                            "global_id": global_id,
                            "original_stem": stem,
                            "original_image_path": str(image_path),
                            "original_mask_path": str(mask_path),
                            "source_group": stem,
                            "new_name": aug_name,
                            "split": split,
                            "is_augmented": True,
                            "augmentation_variant_index": variant.metadata.variant_index,
                            "augmentation": variant.metadata.to_dict(),
                        }
                    )
                    if config.augmentation.debug.enabled and aug_debug_written < int(config.augmentation.debug.max_samples):
                        aug_debug_writer.write(
                            debug_root=out_root / "debug_augmentation",
                            split=split,
                            stem=_debug_aug_name(aug_name),
                            base_image=image_arr,
                            augmented_image=variant.image,
                            mask=mask_idx,
                            metadata=variant.metadata,
                        )
                        aug_debug_written += 1

        augmentation_summary["debug_samples_written"] = aug_debug_written
        manifest = {
            "schema_version": "microseg.dataset_prepare_manifest.v1",
            "created_utc": _utc_now(),
            "config": asdict(config),
            "augmentation": augmentation_summary | {"config": _serialize_augmentation_config(config)},
            "source_layout": "split_layout",
            "supported_extensions": list(SUPPORTED_IMAGE_EXTENSIONS),
            "split_strategy": "existing_split_layout",
            "leakage_group_mode": "existing_split_layout",
            "leakage_group_regex": "",
            "leakage_groups": len(group_to_split),
            "group_to_split": group_to_split,
            "split_counts": split_counts,
            "mapping": mapping,
        }
        manifest_path = out_root / "dataset_prepare_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return DatasetPrepareResult(
            schema_version="microseg.dataset_prepare.v1",
            created_utc=_utc_now(),
            dataset_dir=str(out_root),
            output_dir=str(out_root),
            used_existing_splits=True,
            prepared=True,
            split_counts=split_counts,
            source_layout="split_layout",
            manifest_path=str(manifest_path),
        )

    pair_dirs = _find_unsplit_dirs(root)
    if pair_dirs is None:
        raise FileNotFoundError(
            f"dataset layout not recognized at {root}. "
            "Expected either explicit split folders or unsplit source/masks folders."
        )
    images_dir, masks_dir = pair_dirs
    pairs = _pairs_by_stem(images_dir, masks_dir)

    split_by_pair_index, group_to_split, source_group_for_index = _plan_split_assignments(pairs, config)

    mapping: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for idx, (stem, image_path, mask_path) in enumerate(pairs, start=1):
        split = split_by_pair_index[idx - 1]
        global_id = f"{idx:0{int(config.id_width)}d}"
        new_name = f"{stem}_{global_id}.png"
        out_img = out_root / split / "images" / new_name
        out_msk = out_root / split / "masks" / new_name
        image_arr, mask_idx = _load_png_pair(image_path, mask_path, config)
        _write_png_arrays(image_arr, mask_idx, out_img, out_msk)
        split_counts[split] += 1
        mapping.append(
            {
                "id": global_id,
                "global_id": global_id,
                "original_stem": stem,
                "original_image_path": str(image_path),
                "original_mask_path": str(mask_path),
                "source_group": source_group_for_index[idx - 1],
                "new_name": new_name,
                "split": split,
                "is_augmented": False,
            }
        )

        variants = aug_runner.generate_variants(
            image=image_arr,
            mask=mask_idx,
            split=split,
            source_name=new_name,
            resolved_stage=_resolved_augmentation_stage(config),
        )
        for variant in variants:
            aug_name = f"{Path(new_name).stem}_aug{variant.metadata.variant_index:03d}.png"
            out_aug_img = out_root / split / "images" / aug_name
            out_aug_msk = out_root / split / "masks" / aug_name
            _write_png_arrays(variant.image, variant.mask, out_aug_img, out_aug_msk)
            split_counts[split] += 1
            augmentation_summary["generated_samples"] += 1
            mapping.append(
                {
                    "id": f"{global_id}_aug{variant.metadata.variant_index:03d}",
                    "global_id": global_id,
                    "original_stem": stem,
                    "original_image_path": str(image_path),
                    "original_mask_path": str(mask_path),
                    "source_group": source_group_for_index[idx - 1],
                    "new_name": aug_name,
                    "split": split,
                    "is_augmented": True,
                    "augmentation_variant_index": variant.metadata.variant_index,
                    "augmentation": variant.metadata.to_dict(),
                }
            )
            if config.augmentation.debug.enabled and aug_debug_written < int(config.augmentation.debug.max_samples):
                aug_debug_writer.write(
                    debug_root=out_root / "debug_augmentation",
                    split=split,
                    stem=_debug_aug_name(aug_name),
                    base_image=image_arr,
                    augmented_image=variant.image,
                    mask=mask_idx,
                    metadata=variant.metadata,
                )
                aug_debug_written += 1

    augmentation_summary["debug_samples_written"] = aug_debug_written
    manifest = {
        "schema_version": "microseg.dataset_prepare_manifest.v1",
        "created_utc": _utc_now(),
        "config": asdict(config),
        "augmentation": augmentation_summary | {"config": _serialize_augmentation_config(config)},
        "source_layout": f"unsplit:{images_dir.relative_to(root)}+{masks_dir.relative_to(root)}",
        "supported_extensions": list(SUPPORTED_IMAGE_EXTENSIONS),
        "split_strategy": config.split_strategy,
        "leakage_group_mode": config.leakage_group_mode,
        "leakage_group_regex": config.leakage_group_regex,
        "leakage_groups": len(group_to_split),
        "group_to_split": group_to_split,
        "split_counts": split_counts,
        "mapping": mapping,
    }
    manifest_path = out_root / "dataset_prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return DatasetPrepareResult(
        schema_version="microseg.dataset_prepare.v1",
        created_utc=_utc_now(),
        dataset_dir=str(out_root),
        output_dir=str(out_root),
        used_existing_splits=False,
        prepared=True,
        split_counts=split_counts,
        source_layout=manifest["source_layout"],
        manifest_path=str(manifest_path),
    )
