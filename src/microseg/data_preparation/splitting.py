"""Split-planning helpers for paired dataset preparation."""

from __future__ import annotations

import random
import re
from pathlib import Path

from src.microseg.data_preparation.pairing import ImageMaskPair


_AUGMENTATION_SUFFIX_PATTERNS = [
    re.compile(r"(.+?)(?:[_-](?:aug|crop|tile|patch|view|variant)[_-]?\d+)$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-](?:flip|hflip|vflip|hvflip))$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-]rot(?:ation)?[_-]?\d+)$", flags=re.IGNORECASE),
    re.compile(r"(.+?)(?:[_-]r\d+)$", flags=re.IGNORECASE),
]


def derive_source_group(stem: str, *, mode: str, regex: str) -> str:
    """Derive the leakage-aware grouping key for a sample stem."""

    if mode == "stem":
        return stem
    if mode == "regex":
        if not regex.strip():
            raise ValueError("leakage_group_regex is required when leakage_group_mode=regex")
        match = re.search(regex, stem)
        if match is None:
            return stem
        if match.groups():
            candidate = str(match.group(1)).strip()
        else:
            candidate = str(match.group(0)).strip()
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


def build_split_map(
    pairs: list[ImageMaskPair],
    *,
    train_pct: float,
    val_pct: float,
    max_val_examples: int | None,
    max_test_examples: int | None,
    seed: int,
    split_strategy: str,
    leakage_group_mode: str,
    leakage_group_regex: str,
) -> tuple[dict[str, list[int]], dict[int, str], dict[str, str]]:
    """Plan train/val/test assignments for paired-folder input."""

    n_pairs = len(pairs)
    if n_pairs <= 0:
        return {"train": [], "val": [], "test": []}, {}, {}

    if split_strategy == "random":
        grouped_indices = {f"{pair.stem}#{idx + 1}": [idx] for idx, pair in enumerate(pairs)}
        source_group_for_index = {idx: f"{pair.stem}#{idx + 1}" for idx, pair in enumerate(pairs)}
    elif split_strategy == "leakage_aware":
        grouped_indices: dict[str, list[int]] = {}
        source_group_for_index = {}
        for idx, pair in enumerate(pairs):
            group = derive_source_group(
                pair.stem,
                mode=leakage_group_mode,
                regex=leakage_group_regex,
            )
            grouped_indices.setdefault(group, []).append(idx)
            source_group_for_index[idx] = group
    else:
        raise ValueError(f"unsupported split_strategy: {split_strategy}")

    indices = list(range(n_pairs))
    random.Random(seed).shuffle(indices)
    n_trainval_ratio = int(round(n_pairs * train_pct))
    n_trainval_ratio = max(0, min(n_pairs, n_trainval_ratio))
    n_test_ratio = n_pairs - n_trainval_ratio
    n_val_ratio = int(round(n_trainval_ratio * val_pct))
    n_val_ratio = max(0, min(n_trainval_ratio, n_val_ratio))

    n_test = n_test_ratio
    if max_test_examples is not None:
        n_test = min(n_test, int(max_test_examples))
    n_test = max(0, min(n_pairs, n_test))

    n_remaining = n_pairs - n_test
    n_val = min(n_val_ratio, n_remaining)
    if max_val_examples is not None:
        n_val = min(n_val, int(max_val_examples))
    n_val = max(0, min(n_remaining, n_val))

    targets = {"train": n_remaining - n_val, "val": n_val, "test": n_test}
    groups = list(grouped_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: len(grouped_indices[group]), reverse=True)

    current = {"train": 0, "val": 0, "test": 0}
    group_to_split: dict[str, str] = {}
    for group in groups:
        size = len(grouped_indices[group])
        deficits = {split: targets[split] - current[split] for split in ("train", "val", "test")}
        best_split = max(deficits.items(), key=lambda item: (item[1], item[0]))[0]
        if deficits[best_split] < 0:
            best_split = min(current.items(), key=lambda item: item[1])[0]
        group_to_split[group] = best_split
        current[best_split] += size

    split_map = {"train": [], "val": [], "test": []}
    for idx in range(n_pairs):
        split_map[group_to_split[source_group_for_index[idx]]].append(idx)
    for split in split_map:
        split_map[split] = sorted(split_map[split], key=lambda idx: Path(pairs[idx].image_path).name)
    return split_map, source_group_for_index, group_to_split
