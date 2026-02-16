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
    ) -> Path:
        """Export corrected sample artifacts and metadata."""

        root = Path(output_dir)
        sample_id = f"{Path(run.image_name).stem}_{run.run_id}"
        sample_dir = root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        input_path = sample_dir / "input.png"
        predicted_mask_path = sample_dir / "predicted_mask.png"
        corrected_mask_path = sample_dir / "corrected_mask.png"
        corrected_overlay_path = sample_dir / "corrected_overlay.png"

        run.input_image.save(input_path)
        run.mask_image.save(predicted_mask_path)

        cmask = (corrected_mask > 0).astype(np.uint8) * 255
        Image.fromarray(cmask).save(corrected_mask_path)

        overlay = mask_overlay(np.array(run.input_image), cmask)
        Image.fromarray(overlay).save(corrected_overlay_path)

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
            files={
                "input": input_path.name,
                "predicted_mask": predicted_mask_path.name,
                "corrected_mask": corrected_mask_path.name,
                "corrected_overlay": corrected_overlay_path.name,
            },
            metrics=run.metrics,
        )

        metadata = asdict(record)
        metadata["correction_foreground_pixels"] = int(np.count_nonzero(cmask))
        metadata["predicted_foreground_pixels"] = int(np.count_nonzero(np.array(run.mask_image)))

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
                shutil.copy2(sample_dir / rec["files"]["corrected_mask"], msk_dir / f"{sid}.png")
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
