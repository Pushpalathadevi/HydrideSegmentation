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


class PairCollector:
    """Collect paired segmentation image/mask files from an input folder."""

    def __init__(self, *, image_extensions: list[str], mask_extensions: list[str], mask_name_patterns: list[str], strict: bool = True) -> None:
        self.image_extensions = {ext.lower() for ext in image_extensions}
        self.mask_extensions = {ext.lower() for ext in mask_extensions}
        self.mask_name_patterns = mask_name_patterns
        self.strict = strict

    def collect(self, input_dir: Path) -> list[ImageMaskPair]:
        images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in self.image_extensions and "_mask" not in p.stem])
        pairs: list[ImageMaskPair] = []
        missing: list[str] = []
        for image in images:
            stem = image.stem
            found = None
            for pattern in self.mask_name_patterns:
                candidate = image.with_name(pattern.format(stem=stem))
                if candidate.exists() and candidate.suffix.lower() in self.mask_extensions:
                    found = candidate
                    break
            if found is None:
                missing.append(stem)
                continue
            pairs.append(ImageMaskPair(stem=stem, image_path=image, mask_path=found))

        if not pairs:
            raise RuntimeError(f"no image/mask pairs found in {input_dir}")
        if missing and self.strict:
            raise ValueError(f"missing masks for stems: {missing[:10]}")
        return sorted(pairs, key=lambda p: p.stem)
