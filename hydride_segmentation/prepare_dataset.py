#!/usr/bin/env python3
"""
Prepare two dataset layouts *simultaneously* from paired image / mask files:

1. **Oxford-IIIT-Pet style** (images/ + annotations/trimaps/ + *.txt splits)
2. **MaDo-UNet style**      (train|val|test/{images,masks}/)

Masks are normalised to binary {0,1} uint8 before saving; for MaDo-UNet the
mask is rescaled to 0/255 because that repository treats any non–zero pixel
as foreground and many medical datasets store masks that way.

---------------------------------------------------------------------------
Expected input file naming
---------------------------------------------------------------------------
<stem>.jpg / <stem>.png              ← *image*   (grayscale or RGB)
<stem>.png  / <stem>_mask.png        ← *mask*    (same stem, any extension)

Adjust `_collect_pairs()` if your naming differs (e.g. *_merged.png etc.).
"""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DatasetPreparer:
    # --------------------------------------------------------------- init --
    def __init__(self, settings: Dict):
        """
        Required keys in *settings*:
            input_dir   : folder with paired image / mask files
            output_dir  : destination folder (both datasets written here)
            train_pct   : fraction → train+val (e.g. 0.9)
            val_pct     : fraction of trainval → val (e.g. 0.1)
            seed        : RNG seed (int)
            debug       : bool, plot examples if True
            num_debug   : int, #examples per split to plot
            skip_sanity : bool, skip sanity checks if True
        """
        self.cfg = settings
        self.log = self._setup_logger()
        self.rng = random.Random(self.cfg.get("seed", 42))

    # --------------------------------------------------------- logger setup
    @staticmethod
    def _setup_logger() -> logging.Logger:
        log = logging.getLogger("DatasetPreparer")
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            log.addHandler(h)
        return log

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _mask_raw_ok(mask: np.ndarray) -> bool:
        """Accept masks encoded with {0,1} or {0,255} (any channel count)."""
        allow = ({0, 1}, {0, 255}, {0}, {1}, {255})
        return set(np.unique(mask)) in allow

    @staticmethod
    def _read_mask_binary(path: Path) -> np.ndarray:
        """
        Load *path*, convert RGB→gray if needed, return **binary** uint8 {0,1}.
        """
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError(f"Cannot read mask {path}")
        if m.ndim == 3:  # BGR → gray
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return (m > 0).astype("uint8")

    # ------------------------------------------------------------- workflow
    def run(self):
        pairs = self._collect_pairs()
        if not self.cfg.get("skip_sanity", False):
            self._sanity_checks(pairs)
        splits = self._split_indices(len(pairs))
        root = Path(self.cfg["output_dir"]).expanduser()
        if self.cfg.get('export_oxford_style', True):
            self._export_oxford_style(pairs, splits, root / "oxford-iiit-pet")
        else:
            self.log.warning("Skipping the oxford style data export as flag was set false ")
        if self.cfg.get('export_mado_style', True):
            self._export_mado_style(pairs, splits, root / "mado_style")
        else:
            self.log.warning("Skipping the modo style data export as flag was set false ")
        if self.cfg.get("debug", False):
            self._plot_examples(pairs, splits, root)
        self.log.info("Dataset preparation completed.")

    # -------------------------------------------------------- collect pairs
    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Scan `input_dir` for images; infer mask path by *stem* equality.
        Accepts .jpg / .png images with mask as either <stem>.png or
        <stem>_mask.png.
        """
        src = Path(self.cfg["input_dir"]).expanduser()
        self.log.info(f"Scanning {src} for images …")
        pairs: List[Tuple[Path, Path]] = []
        for img in sorted(src.glob("*.*")):
            if img.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            stem = img.stem.replace("_mask", "")  # robustness
            # preferred mask names in order
            for cand in (img.with_name(f"{stem}.png"),
                         img.with_name(f"{stem}.png")):
                if cand.exists():
                    pairs.append((img, cand))
                    break
        if not pairs:
            raise RuntimeError("No valid image/mask pairs found!")
        if self.cfg.get("debug", False):
            # deterministic shuffle for reproducibility
            self.rng.shuffle(pairs)
            pairs = pairs[:100]
            self.log.warning("DEBUG mode active – using only 100 image/mask pairs")
        self.log.info(f"Found {len(pairs)} pairs.")
        return pairs

    # --------------------------------------------------------- sanity checks
    def _sanity_checks(self, pairs):
        self.log.info("Running sanity checks …")
        for img_p, mask_p in pairs:
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            raw = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
            if img is None or raw is None:
                raise ValueError(f"Unreadable: {img_p} / {mask_p}")
            if not self._mask_raw_ok(raw):
                self.log.warning(
                    f"Mask {mask_p.name} is not binary; will be binarised.")
            if img.shape != self._read_mask_binary(mask_p).shape:
                raise ValueError(f"Shape mismatch {img_p.name} vs {mask_p.name}")
        self.log.info("All sanity checks passed.")

    # ------------------------------------------------------ split generator
    def _split_indices(self, n: int) -> Dict[str, List[int]]:
        idx = list(range(n))
        self.rng.shuffle(idx)
        n_test = int(round(n * (1 - self.cfg["train_pct"])))
        test = idx[:n_test]
        trainval = idx[n_test:]
        n_val = int(round(len(trainval) * self.cfg["val_pct"]))
        val = trainval[:n_val]
        train = trainval[n_val:]
        self.log.info(f"Split sizes — train:{len(train)}  "
                      f"val:{len(val)}  test:{len(test)}")
        return {"trainval": trainval, "train": train, "val": val, "test": test}

    # --------------------------------------------------- export: Oxford style
    def _export_oxford_style(self, pairs, splits, root: Path):
        self.log.info(f"Writing Oxford-style dataset → {root}")
        img_dir = root / "images"
        msk_dir = root / "annotations" / "trimaps"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        # copy images & masks once
        for img_src, mask_src in pairs:
            shutil.copy2(img_src, img_dir / img_src.name)
            mask_bin = self._read_mask_binary(mask_src)
            cv2.imwrite(str(msk_dir / mask_src.name), mask_bin)

        # write split text files
        ann_root = root / "annotations"
        ann_root.mkdir(exist_ok=True)
        for key, idxs in splits.items():
            with (ann_root / f"{key}.txt").open("w") as fh:
                fh.writelines(f"{pairs[i][0].stem} 1 1 1\n" for i in idxs)

    # --------------------------------------------------- export: MaDo style
    def _export_mado_style(self, pairs, splits, root: Path):
        self.log.info(f"Writing MaDo-UNet dataset → {root}")
        for split in ("train", "val", "test"):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "masks").mkdir(parents=True, exist_ok=True)

        for split_name, idxs in splits.items():
            if split_name == "trainval":  # not needed here
                continue
            for i in idxs:
                img_src, mask_src = pairs[i]
                dst_img = root / split_name / "images" / img_src.name
                dst_mask = root / split_name / "masks" / img_src.name
                shutil.copy2(img_src, dst_img)
                mask_bin = self._read_mask_binary(mask_src) * 255  # 0/255
                cv2.imwrite(str(dst_mask), mask_bin)

    # ------------------------------------------------------- debug plotting
    def _plot_examples(self, pairs, splits, root: Path):
        self.log.info("Plotting debug examples …")
        num_each = int(self.cfg.get("num_debug", 2))
        for split, idxs in splits.items():
            for i in self.rng.sample(idxs, min(num_each, len(idxs))):
                img_p, mask_p = pairs[i]
                img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                mask = self._read_mask_binary(mask_p)
                overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                overlay[mask == 1] = [255, 0, 0]
                blended = cv2.addWeighted(
                    overlay, 0.4,
                    cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0.6, 0)
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img, cmap="gray");
                axs[0].set_title("source")
                axs[1].imshow(mask, cmap="gray");
                axs[1].set_title("mask (0/1)")
                axs[2].imshow(blended);
                axs[2].set_title("overlay")
                for ax in axs:
                    ax.axis("off")
                fig.suptitle(f"{img_p.stem}  |  split: {split}")
                plt.tight_layout()
                plt.show()


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    SETTINGS = {
        "input_dir": r"V:\maniBackUp\HydrideSegmentation\inputForHydrideData6.0",
        "output_dir": r"V:\maniBackUp\HydrideSegmentation\HydrideData6.0",
        "train_pct": 0.95,
        "val_pct": 0.05,
        "seed": 123,
        "debug": False,
        "num_debug": 2,
        "skip_sanity": True,  # set False to enable sanity checks
        "export_oxford_style": True,
        "export_modo_style": True,
    }
    DatasetPreparer(SETTINGS).run()
