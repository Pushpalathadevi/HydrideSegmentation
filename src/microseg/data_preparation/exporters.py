"""Dataset exporters for Oxford-like and MaDo-like layouts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image



def write_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        if arr.ndim == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(path)
        else:
            Image.fromarray(arr).save(path)
        return
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"failed to write file: {path}")


class OxfordExporter:
    """Write Oxford-IIIT-Pet-like dataset layout."""

    name = "oxford"

    def export(self, root: Path, records: list[dict[str, object]], dry_run: bool = False) -> None:
        if dry_run:
            return
        images_dir = root / "images"
        trimaps_dir = root / "annotations" / "trimaps"
        ann_dir = root / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        trimaps_dir.mkdir(parents=True, exist_ok=True)

        split_stems: dict[str, list[str]] = {"train": [], "val": [], "test": [], "trainval": []}
        for rec in records:
            split = str(rec["split"])
            stem = str(rec["stem"])
            split_stems[split].append(stem)
            if split in {"train", "val"}:
                split_stems["trainval"].append(stem)
            write_image(images_dir / str(rec["image_file_name"]), rec["image_out"])
            write_image(trimaps_dir / str(rec["mask_file_name"]), rec["mask_out"])

        for split_name in ["train", "val", "test", "trainval"]:
            lines = [f"{stem} 1 1 1\n" for stem in sorted(split_stems[split_name])]
            (ann_dir / f"{split_name}.txt").write_text("".join(lines), encoding="utf-8")


class MadoExporter:
    """Write MaDo-style train|val|test/{images,masks} layout."""

    name = "mado"

    def export(self, root: Path, records: list[dict[str, object]], dry_run: bool = False) -> None:
        if dry_run:
            return
        for rec in records:
            split = str(rec["split"])
            img_path = root / split / "images" / str(rec["image_file_name"])
            mask_path = root / split / "masks" / str(rec["mask_file_name"])
            write_image(img_path, rec["image_out"])
            write_image(mask_path, rec["mask_out"])
