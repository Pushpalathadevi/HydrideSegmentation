"""
Image-merging utility
---------------------
* imgA  : matrix (background)      → **any format** (comes from `matrix_dir`)
* imgB  : hydride tile / patch     → **JPEG only**  (comes from `hydride_dir`)
* mask  : binary mask for imgB     → **same stem, .png extension**

Example pair
------------
…/Mask/3PB_SRT_7293_ASh_3.jpg      ← imgB   (hydride tile)
…/Mask/3PB_SRT_7293_ASh_3.png      ← mask   (binary mask)
"""

# --------------------------------------------------------- imports -------
import csv
import glob
import logging
import os
import random
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ========================================================= ImageMerger ==
class ImageMerger:
    # ---------------------------------------------------- initialisation --
    def __init__(self, settings: Dict):
        self.settings = settings
        self.logger   = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ImageMerger")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(ch)
        return logger

    # ------------------------------------------------------- file helpers --
    @staticmethod
    def _derive_mask_path(imageB_path: str) -> str:
        """…/foo.jpg  →  …/foo.png (same folder, same stem)"""
        folder, fname = os.path.split(imageB_path)
        stem, _       = os.path.splitext(fname)
        return os.path.join(folder, f"{stem}.png")

    def load_image(self, path: str, *, grayscale: bool = False) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img  = cv2.imread(path, flag)
        if img is None:
            self.logger.error(f"Cannot read {path}")
            raise FileNotFoundError(path)
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.logger.debug(f"Loaded {path}  shape={img.shape}")
        return img

    # ---------------------------------------------------- tiny utilities --
    @staticmethod
    def _resize(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, (hw[1], hw[0]))

    @staticmethod
    def _binarise(mask: np.ndarray) -> np.ndarray:
        _, m = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m

    # ----------------------------------------- EXTRA augmentation helpers --
    @staticmethod
    def _adjust_brightness(img: np.ndarray, delta: int) -> np.ndarray:
        return np.clip(img.astype(np.int16) + delta, 0, 255).astype("uint8")

    @staticmethod
    def _adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        mean = 128
        out  = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255)
        return out.astype("uint8")

    @staticmethod
    def _add_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        out   = np.clip(img.astype(np.float32) + noise, 0, 255)
        return out.astype("uint8")

    def _random_crop(
        self, img: np.ndarray, mask: np.ndarray, ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w   = img.shape[:2]
        new_h  = int(h * ratio)
        new_w  = int(w * ratio)
        y0     = random.randint(0, h - new_h)
        x0     = random.randint(0, w - new_w)
        img_cp = img[y0 : y0 + new_h, x0 : x0 + new_w]
        msk_cp = mask[y0 : y0 + new_h, x0 : x0 + new_w]
        img_cp = cv2.resize(img_cp, (w, h), interpolation=cv2.INTER_LINEAR)
        msk_cp = cv2.resize(msk_cp, (w, h), interpolation=cv2.INTER_NEAREST)
        return img_cp, msk_cp

    # ------------------------------------------- augmentation main entry --
    def _apply_augmentations(
        self, img: np.ndarray, mask: np.ndarray, cfg: Dict
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        h, w       = img.shape[:2]
        centre     = (w / 2, h / 2)
        angle      = 0.0
        scale      = 1.0
        desc_parts = []

        # ----- geometric augments ---------------------------------------
        if random.random() < cfg.get("rotate", {}).get("prob", 0):
            angle = random.uniform(*cfg["rotate"].get("angle_range", (-10, 10)))
            desc_parts.append(f"rot={angle:.1f}°")

        if random.random() < cfg.get("scale", {}).get("prob", 0):
            scale = random.uniform(*cfg["scale"].get("scale_range", (0.9, 1.1)))
            desc_parts.append(f"scale={scale:.2f}")

        # apply affine
        M        = cv2.getRotationMatrix2D(centre, angle, scale)
        img_aug  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
        mask_aug = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # flips
        if random.random() < cfg.get("flipH", {}).get("prob", 0):
            img_aug  = cv2.flip(img_aug, 1)
            mask_aug = cv2.flip(mask_aug, 1)
            desc_parts.append("flipH")
        if random.random() < cfg.get("flipV", {}).get("prob", 0):
            img_aug  = cv2.flip(img_aug, 0)
            mask_aug = cv2.flip(mask_aug, 0)
            desc_parts.append("flipV")

        # crop
        if random.random() < cfg.get("crop", {}).get("prob", 0):
            ratio_rng = cfg["crop"].get("ratio_range", (0.8, 1.0))
            ratio     = random.uniform(*ratio_rng)
            img_aug, mask_aug = self._random_crop(img_aug, mask_aug, ratio)
            desc_parts.append(f"crop={ratio:.2f}")

        # photometric (image only)
        if random.random() < cfg.get("brightness", {}).get("prob", 0):
            delta_rng = cfg["brightness"].get("delta_range", (-20, 20))
            delta     = random.randint(*delta_rng)
            img_aug   = self._adjust_brightness(img_aug, delta)
            desc_parts.append(f"ΔB={delta:+d}")

        if random.random() < cfg.get("contrast", {}).get("prob", 0):
            f_rng  = cfg["contrast"].get("factor_range", (0.8, 1.2))
            factor = random.uniform(*f_rng)
            img_aug = self._adjust_contrast(img_aug, factor)
            desc_parts.append(f"Cx={factor:.2f}")

        if random.random() < cfg.get("noise", {}).get("prob", 0):
            std_rng = cfg["noise"].get("std_range", (3, 10))
            std     = random.uniform(*std_rng)
            img_aug = self._add_gaussian_noise(img_aug, std)
            desc_parts.append(f"noiseσ={std:.1f}")

        mask_aug = self._binarise(mask_aug)  # after any interpolation

        desc = ", ".join(desc_parts) if desc_parts else "none"
        self.logger.debug(f"Applied augmentations: {desc}")
        return img_aug, mask_aug, desc

    # ---------------------------------------------------- merge pipeline --
    def apply_mask_and_merge(self, imgA: np.ndarray, imgB: np.ndarray, mask: np.ndarray):
        imgA, b_desc    = self._maybe_brightness(imgA)
        a_desc_matrix   = "none"
        a_desc_hydride  = "none"

        if self.settings.get("matrix_augmentations"):
            imgA, _, a_desc_matrix = self._apply_augmentations(
                imgA, np.zeros_like(imgA), self.settings["matrix_augmentations"])

        if self.settings.get("hydride_augmentations"):
            imgB, mask, a_desc_hydride = self._apply_augmentations(
                imgB, mask, self.settings["hydride_augmentations"])

        imgB = self._resize(imgB, imgA.shape[:2])
        mask = self._binarise(self._resize(mask, imgA.shape[:2]))

        merged          = imgA.copy()
        merged[mask > 0] = imgB[mask > 0]

        return merged, mask, {
            "hydride_aug": a_desc_hydride,
            "matrix_aug":  a_desc_matrix,
            "brightness":  b_desc,
        }

    # --------------------------------------------------- misc helpers ----
    def _maybe_brightness(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        cfg = self.settings.get("brightness", {})
        if not cfg.get("enable", False):
            return img, "none"
        delta = int(cfg.get("delta", 0))
        if delta == 0:
            return img, "none"
        img_mod = self._adjust_brightness(img, delta)
        return img_mod, f"ΔB={delta:+d}"

    def export_images(self, merged: np.ndarray, mask: np.ndarray, imgA_path: str, imgB_path: str):
        out_dir = self.settings["output_path"]
        os.makedirs(out_dir, exist_ok=True)

        matrix_name  = os.path.splitext(os.path.basename(imgA_path))[0]
        hydride_name = os.path.splitext(os.path.basename(imgB_path))[0]
        out_stem     = f"{matrix_name}__with__{hydride_name}"

        merged_path  = os.path.join(out_dir, f"{out_stem}_merged.jpg")
        mask_path    = os.path.join(out_dir, f"{out_stem}_merged.png")

        cv2.imwrite(merged_path, merged)
        red = np.zeros((*mask.shape, 3), dtype=np.uint8)
        red[mask > 0] = (0, 0, 255)
        cv2.imwrite(mask_path, red)

        return merged_path, mask_path

    def run(self, imgA_path: str, imgB_path: str):
        """Merge a hydride patch onto a matrix image and save outputs."""
        mask_path = self._derive_mask_path(imgB_path)

        imgA = self.load_image(imgA_path, grayscale=True)
        imgB = self.load_image(imgB_path, grayscale=True)
        mask = self.load_image(mask_path, grayscale=True)

        merged, final_mask, desc = self.apply_mask_and_merge(imgA, imgB, mask)
        merged_path, mask_path   = self.export_images(merged, final_mask, imgA_path, imgB_path)

        return {
            "Matrix":     os.path.splitext(os.path.basename(imgA_path))[0],
            "Hydride":    os.path.splitext(os.path.basename(imgB_path))[0],
            "MatrixAug":  desc["matrix_aug"],
            "HydrideAug": desc["hydride_aug"],
            "Brightness": desc["brightness"],
            "MergedFile": merged_path,
            "MaskFile":   mask_path,
        }


# =============================================================== main ===
if __name__ == "__main__":

    # ------------------ FULL BATCH  PATHS --------------------- #
    matrix_dir = r"L:\maniBackUp\HydrideSegmentation\MatrixAlone"
    hydride_dir = r"L:\maniBackUp\HydrideSegmentation\HydrideMasks"
    output_dir = r"L:\maniBackUp\HydrideSegmentation\Merged4.0"
    # ----------------------------------------------------------- #

    settings = {
        "debug": False,
        "output_path": output_dir,           # <── uses the variable just set

        # ------------ matrix image augmentations -----------
        "matrix_augmentations": {
            "rotate":     {"prob": 0.4, "angle_range": (-10, 10)},
            "scale":      {"prob": 0.3, "scale_range": (0.9, 2.5)},
            "flipH":      {"prob": 0.2},
            "brightness": {"prob": 0.4, "delta_range": (-50, 50)},
            "contrast":   {"prob": 0.4, "factor_range": (0.75, 1.25)},
            "crop":       {"prob": 0.3, "ratio_range": (0.85, 1.0)},
            "noise":      {"prob": 0.3, "std_range": (4, 10)},
        },

        # ------------ hydride patch + mask augmentations --
        "hydride_augmentations": {
            "rotate": {"prob": 0.5, "angle_range": (-10, 10)},
            "scale":  {"prob": 0.5, "scale_range": (0.8, 1.2)},
            "flipH":  {"prob": 0.4},
            "flipV":  {"prob": 0.4},
        },

        # optional global brightness tweak for *matrix* only
        "brightness": {"enable": False, "delta": 10},
    }

    merger = ImageMerger(settings)

    # ------------------ file discovery --------------------- #
    matrix_images  = glob.glob(os.path.join(matrix_dir, "*.*"))
    hydride_inputs = (
        glob.glob(os.path.join(hydride_dir, "*.jpg"))  +
        glob.glob(os.path.join(hydride_dir, "*.jpeg")) +
        glob.glob(os.path.join(hydride_dir, "*.JPG"))  +
        glob.glob(os.path.join(hydride_dir, "*.JPEG"))
    )

    log_entries = []
    os.makedirs(output_dir, exist_ok=True)

    for imgA_path in matrix_images:
        for imgB_path in hydride_inputs:
            try:
                print(f"🔄  {os.path.basename(imgB_path)} → {os.path.basename(imgA_path)}")
                log_entries.append(merger.run(imgA_path, imgB_path))
            except Exception as e:
                print(f"❌  {imgB_path} onto {imgA_path} → {e}")

    csv_path = os.path.join(output_dir, "merge_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Matrix", "Hydride", "MatrixAug", "HydrideAug",
                "Brightness", "MergedFile", "MaskFile",
            ],
        )
        writer.writeheader()
        writer.writerows(log_entries)

    print(f"✅ CSV log saved: {csv_path}")
