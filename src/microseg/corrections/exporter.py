"""Export utilities for corrected masks and training dataset packaging."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.microseg.app.desktop_workflow import DesktopRunRecord
from src.microseg.corrections.classes import (
    DEFAULT_CLASS_MAP,
    SegmentationClassMap,
    colorize_index_mask,
    to_index_mask,
)
from src.microseg.domain.corrections import CorrectionExportRecord
from src.microseg.utils import mask_overlay, to_rgb


SCHEMA_VERSION = "microseg.correction.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorrectionExporter:
    """Export corrected samples using a versioned schema."""

    def export_sample(
        self,
        run: DesktopRunRecord,
        corrected_mask: np.ndarray,
        output_dir: str | Path,
        *,
        annotator: str = "unknown",
        notes: str = "",
        class_map: SegmentationClassMap | None = None,
        formats: set[str] | None = None,
    ) -> Path:
        """Export corrected sample artifacts and metadata.

        Supported formats:
        - ``indexed_png``: indexed masks in PNG format
        - ``color_png``: colorized masks using class map colors
        - ``numpy_npy``: corrected indexed mask in NPY format
        """

        root = Path(output_dir)
        sample_id = f"{Path(run.image_name).stem}_{run.run_id}"
        sample_dir = root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        class_map = class_map or DEFAULT_CLASS_MAP
        formats = formats or {"indexed_png", "color_png"}

        input_path = sample_dir / "input.png"
        predicted_mask_path = sample_dir / "predicted_mask_indexed.png"
        corrected_mask_path = sample_dir / "corrected_mask_indexed.png"
        corrected_overlay_path = sample_dir / "corrected_overlay.png"
        predicted_color_path = sample_dir / "predicted_mask_color.png"
        corrected_color_path = sample_dir / "corrected_mask_color.png"
        corrected_npy_path = sample_dir / "corrected_mask.npy"

        run.input_image.save(input_path)
        pred_idx = to_index_mask(np.array(run.mask_image))
        corr_idx = to_index_mask(corrected_mask)

        files_payload: dict[str, str] = {"input": input_path.name}

        if "indexed_png" in formats:
            Image.fromarray(pred_idx).save(predicted_mask_path)
            Image.fromarray(corr_idx).save(corrected_mask_path)
            files_payload["predicted_mask_indexed"] = predicted_mask_path.name
            files_payload["corrected_mask_indexed"] = corrected_mask_path.name
            files_payload["predicted_mask"] = predicted_mask_path.name
            files_payload["corrected_mask"] = corrected_mask_path.name

        if "color_png" in formats:
            Image.fromarray(colorize_index_mask(pred_idx, class_map)).save(predicted_color_path)
            Image.fromarray(colorize_index_mask(corr_idx, class_map)).save(corrected_color_path)
            files_payload["predicted_mask_color"] = predicted_color_path.name
            files_payload["corrected_mask_color"] = corrected_color_path.name

        if "numpy_npy" in formats:
            np.save(corrected_npy_path, corr_idx)
            files_payload["corrected_mask_numpy"] = corrected_npy_path.name

        overlay = mask_overlay(np.array(run.input_image), (corr_idx > 0).astype(np.uint8) * 255)
        Image.fromarray(overlay).save(corrected_overlay_path)
        files_payload["corrected_overlay"] = corrected_overlay_path.name

        record = CorrectionExportRecord(
            schema_version=SCHEMA_VERSION,
            sample_id=sample_id,
            source_image_path=run.image_path,
            model_id=run.model_id,
            model_name=run.model_name,
            run_id=run.run_id,
            created_utc=_utc_now(),
            annotator=annotator,
            notes=notes,
            files=files_payload,
            metrics=run.metrics,
        )

        metadata = asdict(record)
        metadata["class_map"] = class_map.as_dict()
        metadata["export_formats"] = sorted(list(formats))
        metadata["correction_foreground_pixels"] = int(np.count_nonzero(corr_idx > 0))
        metadata["predicted_foreground_pixels"] = int(np.count_nonzero(pred_idx > 0))

        (sample_dir / "correction_record.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return sample_dir


class CorrectionDatasetPackager:
    """Package corrected samples into train/val/test folder layout."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def package(
        self,
        sample_dirs: list[str | Path],
        output_dir: str | Path,
        *,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Path:
        """Build dataset layout from correction sample directories."""

        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("Invalid split ratios")

        samples = [Path(p) for p in sample_dirs]
        for p in samples:
            if not (p / "correction_record.json").exists():
                raise FileNotFoundError(f"Missing correction_record.json in {p}")

        rng = random.Random(self.seed)
        order = list(samples)
        rng.shuffle(order)

        n = len(order)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))

        train = order[:n_train]
        val = order[n_train:n_train + n_val]
        test = order[n_train + n_val:]

        out = Path(output_dir)
        for split, items in {"train": train, "val": val, "test": test}.items():
            img_dir = out / split / "images"
            msk_dir = out / split / "masks"
            meta_dir = out / split / "metadata"
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            for sample_dir in items:
                rec = json.loads((sample_dir / "correction_record.json").read_text(encoding="utf-8"))
                sid = rec["sample_id"]
                shutil.copy2(sample_dir / rec["files"]["input"], img_dir / f"{sid}.png")
                key = "corrected_mask_indexed" if "corrected_mask_indexed" in rec["files"] else "corrected_mask"
                if key not in rec["files"]:
                    raise KeyError("missing corrected mask entry in correction export record")
                shutil.copy2(sample_dir / rec["files"][key], msk_dir / f"{sid}.png")
                shutil.copy2(sample_dir / "correction_record.json", meta_dir / f"{sid}.json")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "seed": self.seed,
            "splits": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "source_samples": [str(p) for p in samples],
        }
        (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out
