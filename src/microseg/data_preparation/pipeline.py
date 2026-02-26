"""Orchestrator for segmentation dataset preparation."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.microseg.data_preparation.binarization import MaskBinarizer
from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.debug import DebugInspector
from src.microseg.data_preparation.exporters import MadoExporter, OxfordExporter
from src.microseg.data_preparation.manifest import ManifestWriter
from src.microseg.data_preparation.pairing import PairCollector
from src.microseg.data_preparation.resizing import Resizer


@dataclass
class DatasetPrepareResult:
    manifest_path: Path
    split_counts: dict[str, int]
    total_pairs: int


class DatasetPreparer:
    """End-to-end dataset preparation pipeline."""

    def __init__(self, cfg: DatasetPrepConfig) -> None:
        self.cfg = cfg
        self.log = logging.getLogger("microseg.data_preparation")
        self.rng = random.Random(cfg.seed)
        self.collector = PairCollector(
            image_extensions=cfg.image_extensions,
            mask_extensions=cfg.mask_extensions,
            mask_name_patterns=cfg.mask_name_patterns,
            strict=cfg.strict_pairing,
        )
        self.binarizer = MaskBinarizer(cfg)
        self.resizer = Resizer(cfg)
        self.inspector = DebugInspector()
        self.manifest_writer = ManifestWriter()
        self.exporters = {"oxford": OxfordExporter(), "mado": MadoExporter()}

    def run(self) -> DatasetPrepareResult:
        input_dir = Path(self.cfg.input_dir)
        output_dir = Path(self.cfg.output_dir)
        pairs = self.collector.collect(input_dir)
        if self.cfg.debug.enabled:
            pairs = pairs[: self.cfg.debug.limit_pairs]

        if not self.cfg.skip_sanity and not (self.cfg.debug.enabled and self.cfg.debug.skip_sanity_checks):
            self._sanity_check(pairs)

        split_map = self._build_splits(len(pairs))
        split_counts = {k: len(v) for k, v in split_map.items()}
        warnings_all: list[str] = []
        split_for_index = {idx: split for split, idxs in split_map.items() for idx in idxs}
        records: list[dict[str, Any]] = []

        debug_written = 0
        for idx, pair in enumerate(pairs):
            split = split_for_index[idx]
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
            mask_raw = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask_raw is None:
                raise ValueError(f"failed to read pair: {pair}")
            mask_binary, mask_stats = self.binarizer.apply(mask_raw)
            image_out, mask_out, resize_warnings = self.resizer.apply(image, mask_binary)
            warnings_all.extend(resize_warnings)

            mask_export = (mask_out * self.cfg.mask_foreground_value).astype(np.uint8)
            image_file_name = f"{pair.stem}{self.cfg.image_ext}"
            mask_file_name = f"{pair.stem}{self.cfg.mask_ext}"
            rec = {
                "stem": pair.stem,
                "split": split,
                "source_image_path": self._fmt_path(pair.image_path, output_dir),
                "source_mask_path": self._fmt_path(pair.mask_path, output_dir),
                "image_file_name": image_file_name,
                "mask_file_name": mask_file_name,
                "original_shape": list(image.shape[:2]),
                "output_shape": list(image_out.shape[:2]),
                "mask_stats": mask_stats,
                "warnings": resize_warnings,
                "image_out": image_out,
                "mask_out": mask_export,
            }
            records.append(rec)

            if self.cfg.debug.enabled and debug_written < self.cfg.debug.inspection_limit:
                annotation = (
                    f"{pair.stem} split={split} src={image.shape}/{mask_raw.shape} out={image_out.shape}/{mask_out.shape} "
                    f"dtype={mask_out.dtype} uniq={sorted(np.unique(mask_out).tolist())} mode={self.cfg.binarization_mode}"
                )
                self.inspector.write(
                    debug_root=output_dir / "debug_inspection",
                    split=split,
                    stem=pair.stem,
                    image_raw=image_out,
                    mask_raw=mask_raw,
                    mask_binary=mask_out,
                    show=self.cfg.debug.show_plots,
                    ext=self.cfg.debug_ext,
                    draw_contours=self.cfg.debug.draw_contours,
                    annotation=annotation,
                )
                debug_written += 1

        manifest_records = self._serialize_records(records, output_dir)

        if not self.cfg.debug.enabled or not self.cfg.debug.minimal_exports:
            for style in self.cfg.styles:
                exporter = self.exporters[style]
                export_root = output_dir / style
                exporter.export(export_root, records, dry_run=self.cfg.dry_run)
        else:
            for style in self.cfg.styles:
                exporter = self.exporters[style]
                export_root = output_dir / style
                exporter.export(export_root, records[: min(10, len(records))], dry_run=self.cfg.dry_run)

        manifest_path = self.manifest_writer.write(
            output_root=output_dir,
            config=self.cfg.to_dict(),
            input_dir=input_dir,
            records=manifest_records,
            split_counts=split_counts,
            warnings=warnings_all,
        )
        return DatasetPrepareResult(manifest_path=manifest_path, split_counts=split_counts, total_pairs=len(pairs))

    def _serialize_records(self, records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for rec in sorted(records, key=lambda r: str(r["stem"])):
            row = {k: v for k, v in rec.items() if k not in {"image_out", "mask_out"}}
            row["exports"] = {}
            for style in self.cfg.styles:
                if style == "oxford":
                    row["exports"]["oxford"] = {
                        "image_path": self._fmt_path(output_dir / "oxford" / "images" / str(rec["image_file_name"]), output_dir),
                        "mask_path": self._fmt_path(output_dir / "oxford" / "annotations" / "trimaps" / str(rec["mask_file_name"]), output_dir),
                    }
                elif style == "mado":
                    split = str(rec["split"])
                    row["exports"]["mado"] = {
                        "image_path": self._fmt_path(output_dir / "mado" / split / "images" / str(rec["image_file_name"]), output_dir),
                        "mask_path": self._fmt_path(output_dir / "mado" / split / "masks" / str(rec["mask_file_name"]), output_dir),
                    }
            serialized.append(row)
        return serialized

    def _fmt_path(self, path: Path, output_dir: Path) -> str:
        if self.cfg.path_mode == "absolute":
            return str(path.resolve())
        return str(path.resolve().relative_to(output_dir.resolve().parent)) if path.is_absolute() else str(path)

    def _build_splits(self, n: int) -> dict[str, list[int]]:
        idx = list(range(n))
        self.rng.shuffle(idx)
        n_trainval = int(round(n * self.cfg.train_pct))
        trainval = idx[:n_trainval]
        test = idx[n_trainval:]
        n_val = int(round(len(trainval) * self.cfg.val_pct))
        val = trainval[:n_val]
        train = trainval[n_val:]
        return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    def _sanity_check(self, pairs: list[Any]) -> None:
        for pair in pairs:
            img = cv2.imread(str(pair.image_path), cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
            if img is None or msk is None:
                raise ValueError(f"unreadable pair: {pair}")
            if img.shape[:2] != msk.shape[:2]:
                raise ValueError(f"shape mismatch for {pair.stem}: {img.shape} vs {msk.shape}")
