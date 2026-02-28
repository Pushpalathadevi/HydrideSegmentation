"""Orchestrator for segmentation dataset preparation."""

from __future__ import annotations

import json
import logging
import random
import time
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
from src.microseg.data_preparation.pairing import PairCollectionReport, PairCollector
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
        run_start = time.perf_counter()
        input_dir = Path(self.cfg.input_dir)
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs, pairing_report = self.collector.collect_with_report(input_dir)
        self._log_pairing_report(pairing_report)
        if self.cfg.debug.enabled:
            pairs = pairs[: self.cfg.debug.limit_pairs]

        if not self.cfg.skip_sanity and not (self.cfg.debug.enabled and self.cfg.debug.skip_sanity_checks):
            self._sanity_check(pairs)

        split_map = self._build_splits(len(pairs))
        split_counts = {k: len(v) for k, v in split_map.items()}
        warnings_all: list[str] = []
        split_for_index = {idx: split for split, idxs in split_map.items() for idx in idxs}
        records: list[dict[str, Any]] = []
        read_failures: list[str] = []
        empty_output_masks: list[str] = []
        debug_written = 0

        stage_start = time.perf_counter()
        for idx, pair in enumerate(pairs):
            split = split_for_index[idx]
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
            mask_raw = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask_raw is None:
                message = f"failed to read pair: {pair.stem}"
                read_failures.append(message)
                if self.cfg.strict_pairing:
                    raise ValueError(message)
                self.log.warning(message)
                continue
            mask_binary, mask_stats = self.binarizer.apply(mask_raw)
            image_out, mask_out, resize_warnings = self.resizer.apply(
                image,
                mask_binary,
                split=split,
                sample_seed=self.cfg.seed + idx,
            )
            if int(np.count_nonzero(mask_out)) == 0:
                empty_message = "output mask is all zeros after preprocessing"
                empty_output_masks.append(pair.stem)
                mask_stats["all_zero_output_mask"] = True
                if self.cfg.empty_mask_action == "error":
                    raise ValueError(f"{pair.stem}: {empty_message}")
                mask_stats.setdefault("warnings", [])
                mask_stats["warnings"].append(empty_message)
            mask_warnings = [str(w) for w in mask_stats.get("warnings", [])]
            pair_warnings = [f"{pair.stem}: {w}" for w in [*mask_warnings, *resize_warnings]]
            for warning in pair_warnings:
                self.log.warning(warning)
            warnings_all.extend(pair_warnings)

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
                "warnings": [*mask_warnings, *resize_warnings],
                "image_out": image_out,
                "mask_out": mask_export,
            }
            records.append(rec)

            if self.cfg.debug.enabled and debug_written < self.cfg.debug.inspection_limit:
                debug_criteria = self._debug_criteria(mask_stats)
                annotation = (
                    f"{pair.stem} split={split} src={image.shape}/{mask_raw.shape} out={image_out.shape}/{mask_out.shape} "
                    f"dtype={mask_out.dtype} uniq={sorted(np.unique(mask_out).tolist())} mode={debug_criteria.get('mode')}"
                )
                self.inspector.write(
                    debug_root=output_dir / "debug_inspection",
                    split=split,
                    stem=pair.stem,
                    image_input=image,
                    image_output=image_out,
                    mask_input=mask_raw,
                    mask_processed=mask_out,
                    criteria=debug_criteria,
                    show=self.cfg.debug.show_plots,
                    ext=self.cfg.debug_ext,
                    draw_contours=self.cfg.debug.draw_contours,
                    annotation=annotation,
                )
                debug_written += 1
            self._log_progress(idx + 1, len(pairs), stage_start)

        if read_failures:
            warnings_all.extend(read_failures)

        manifest_records = self._serialize_records(records, output_dir)
        self._export_styles(records, output_dir)

        manifest_path = self.manifest_writer.write(
            output_root=output_dir,
            config=self.cfg.to_dict(),
            input_dir=input_dir,
            records=manifest_records,
            split_counts=split_counts,
            warnings=warnings_all,
        )
        self._write_qa_reports(
            output_dir=output_dir,
            records=records,
            split_counts=split_counts,
            pairing_report=pairing_report,
            read_failures=read_failures,
            empty_output_masks=empty_output_masks,
            elapsed_seconds=time.perf_counter() - run_start,
        )
        return DatasetPrepareResult(manifest_path=manifest_path, split_counts=split_counts, total_pairs=len(records))

    def _export_styles(self, records: list[dict[str, Any]], output_dir: Path) -> None:
        if not self.cfg.debug.enabled or not self.cfg.debug.minimal_exports:
            rows = records
        else:
            rows = records[: min(10, len(records))]
        for style in self.cfg.styles:
            exporter = self.exporters[style]
            export_root = output_dir / style
            exporter.export(export_root, rows, dry_run=self.cfg.dry_run)

    def _write_qa_reports(
        self,
        *,
        output_dir: Path,
        records: list[dict[str, Any]],
        split_counts: dict[str, int],
        pairing_report: PairCollectionReport,
        read_failures: list[str],
        empty_output_masks: list[str],
        elapsed_seconds: float,
    ) -> None:
        fg_ratios = [float(rec["mask_stats"].get("fg_ratio", 0.0)) for rec in records]
        binary_sets = [tuple(rec["mask_stats"].get("unique_binary_values", [])) for rec in records]
        qa = {
            "schema_version": "microseg.dataset_prep_qa.v1",
            "pairing": {
                "total_images": pairing_report.total_images,
                "total_masks": pairing_report.total_masks,
                "pair_count": pairing_report.pair_count,
                "missing_masks": pairing_report.missing_masks,
                "missing_images": pairing_report.missing_images,
            },
            "split_counts": split_counts,
            "processed_pairs": len(records),
            "read_failures": read_failures,
            "empty_output_masks": {
                "count": len(empty_output_masks),
                "stems": sorted(empty_output_masks),
                "action": self.cfg.empty_mask_action,
            },
            "mask": {
                "foreground_ratio_mean": float(np.mean(fg_ratios)) if fg_ratios else 0.0,
                "foreground_ratio_min": float(np.min(fg_ratios)) if fg_ratios else 0.0,
                "foreground_ratio_max": float(np.max(fg_ratios)) if fg_ratios else 0.0,
                "binary_value_sets": sorted({str(v) for v in binary_sets}),
            },
            "timing": {
                "elapsed_seconds": elapsed_seconds,
            },
            "dry_run": bool(self.cfg.dry_run),
        }
        qa_path = output_dir / self.cfg.qa_report_name
        qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
        html_path = output_dir / self.cfg.html_report_name
        html_path.write_text(
            "<html><body><h1>Dataset Preparation QA</h1>"
            f"<p>processed_pairs={len(records)}</p>"
            f"<p>split_counts={split_counts}</p>"
            f"<p>missing_masks={len(pairing_report.missing_masks)}</p>"
            f"<p>missing_images={len(pairing_report.missing_images)}</p>"
            f"<p>empty_output_masks={len(empty_output_masks)}</p>"
            f"<p>elapsed_seconds={elapsed_seconds:.2f}</p>"
            "</body></html>",
            encoding="utf-8",
        )

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

    def _log_pairing_report(self, report: PairCollectionReport) -> None:
        self.log.info(
            "pairing summary images=%d masks=%d paired=%d missing_masks=%d missing_images=%d",
            report.total_images,
            report.total_masks,
            report.pair_count,
            len(report.missing_masks),
            len(report.missing_images),
        )

    def _log_progress(self, done: int, total: int, start: float) -> None:
        interval = max(1, int(self.cfg.progress_log_interval))
        if done % interval != 0 and done != total:
            return
        elapsed = time.perf_counter() - start
        per_item = elapsed / done if done else 0.0
        eta = per_item * max(0, total - done)
        pct = (100.0 * done / total) if total else 100.0
        self.log.info("progress %d/%d (%.1f%%) elapsed=%.1fs eta=%.1fs", done, total, pct, elapsed, eta)

    @staticmethod
    def _debug_criteria(mask_stats: dict[str, Any]) -> dict[str, Any]:
        threshold = (
            mask_stats.get("otsu_threshold")
            if "otsu_threshold" in mask_stats
            else mask_stats.get("percentile_threshold")
        )
        return {
            "mode": mask_stats.get("mode"),
            "threshold": threshold,
            "thresholds": mask_stats.get("thresholds", {}),
            "auto_otsu_applied": bool(mask_stats.get("auto_otsu_applied", False)),
            "fg_pixel_count": mask_stats.get("fg_pixel_count"),
            "fg_ratio": mask_stats.get("fg_ratio"),
            "unique_binary_values": mask_stats.get("unique_binary_values"),
            "warnings": mask_stats.get("warnings", []),
        }
