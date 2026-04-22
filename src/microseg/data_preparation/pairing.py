"""Image and mask pairing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageMaskPair:
    """Pair record for a source image and matching mask."""

    stem: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PairCollectionReport:
    """Pairing diagnostics for dataset ingestion."""

    total_images: int
    total_masks: int
    pair_count: int
    missing_masks: list[str]
    missing_images: list[str]


class PairCollector:
    """Collect paired segmentation image/mask files from an input folder."""

    def __init__(
        self,
        *,
        image_extensions: list[str],
        mask_extensions: list[str],
        mask_name_patterns: list[str],
        same_stem_pairing_enabled: bool = False,
        same_stem_image_extensions: list[str] | None = None,
        same_stem_mask_extensions: list[str] | None = None,
        strict: bool = True,
    ) -> None:
        self.image_extensions = {ext.lower() for ext in image_extensions}
        self.mask_extensions = {ext.lower() for ext in mask_extensions}
        self.mask_name_patterns = mask_name_patterns
        self.same_stem_pairing_enabled = bool(same_stem_pairing_enabled)
        self.same_stem_image_extensions = {ext.lower() for ext in (same_stem_image_extensions or [])}
        self.same_stem_mask_extensions = {ext.lower() for ext in (same_stem_mask_extensions or [])}
        self.strict = strict

    def collect(self, input_dir: Path) -> list[ImageMaskPair]:
        pairs, _ = self.collect_with_report(input_dir)
        return pairs

    def collect_with_report(self, input_dir: Path) -> tuple[list[ImageMaskPair], PairCollectionReport]:
        files = [p for p in input_dir.iterdir() if p.is_file()]
        files_by_stem: dict[str, list[Path]] = {}
        for path in files:
            files_by_stem.setdefault(path.stem, []).append(path)
        images = sorted([p for p in files if self._is_image_candidate(p, files_by_stem=files_by_stem)])
        all_masks = sorted([p for p in files if p.suffix.lower() in self.mask_extensions])
        image_stems = {p.stem for p in images}
        pairs: list[ImageMaskPair] = []
        missing_masks: list[str] = []
        for image in images:
            stem = image.stem
            found = None
            for pattern in self.mask_name_patterns:
                candidate = image.with_name(pattern.format(stem=stem))
                if candidate.exists() and candidate.resolve() == image.resolve():
                    continue
                if candidate.exists() and candidate.suffix.lower() in self.mask_extensions:
                    found = candidate
                    break
            if found is None:
                found = self._find_same_stem_mask(image, files_by_stem=files_by_stem)
            if found is None:
                missing_masks.append(stem)
                continue
            pairs.append(ImageMaskPair(stem=stem, image_path=image, mask_path=found))

        normalized_mask_stems = set()
        for mask_path in all_masks:
            normalized = self._normalize_mask_stem(mask_path.name, image_stems=image_stems)
            normalized_mask_stems.add(normalized if normalized else mask_path.stem)
        missing_images = sorted(normalized_mask_stems - image_stems)
        report = PairCollectionReport(
            total_images=len(images),
            total_masks=len(all_masks),
            pair_count=len(pairs),
            missing_masks=sorted(missing_masks),
            missing_images=missing_images,
        )

        if not pairs:
            raise RuntimeError(f"no image/mask pairs found in {input_dir}")
        if self.strict and (missing_masks or missing_images):
            raise ValueError(
                "pairing mismatch detected: "
                f"missing_masks={missing_masks[:10]} missing_images={missing_images[:10]}"
            )
        return sorted(pairs, key=lambda p: p.stem), report

    def _is_image_candidate(self, path: Path, *, files_by_stem: dict[str, list[Path]]) -> bool:
        if path.suffix.lower() not in self.image_extensions or "_mask" in path.stem:
            return False
        if not self.same_stem_pairing_enabled:
            return True
        suffix = path.suffix.lower()
        if suffix not in self.same_stem_mask_extensions:
            return True
        if not self.same_stem_image_extensions:
            return True
        siblings = files_by_stem.get(path.stem, [])
        has_role_image = any(sibling.suffix.lower() in self.same_stem_image_extensions for sibling in siblings)
        return not has_role_image

    def _find_same_stem_mask(self, image: Path, *, files_by_stem: dict[str, list[Path]]) -> Path | None:
        if not self.same_stem_pairing_enabled:
            return None
        if self.same_stem_image_extensions and image.suffix.lower() not in self.same_stem_image_extensions:
            return None
        siblings = files_by_stem.get(image.stem, [])
        candidates = sorted(
            sibling
            for sibling in siblings
            if sibling.resolve() != image.resolve() and sibling.suffix.lower() in self.same_stem_mask_extensions
        )
        return candidates[0] if candidates else None

    def _normalize_mask_stem(self, filename: str, *, image_stems: set[str]) -> str:
        fallback = ""
        for pattern in self.mask_name_patterns:
            if "{stem}" not in pattern:
                continue
            prefix, suffix = pattern.split("{stem}", maxsplit=1)
            if not (filename.startswith(prefix) and filename.endswith(suffix)):
                continue
            end = len(filename) - len(suffix) if suffix else len(filename)
            stem = filename[len(prefix):end]
            if stem:
                if stem in image_stems:
                    return stem
                if not fallback:
                    fallback = stem
        return fallback
