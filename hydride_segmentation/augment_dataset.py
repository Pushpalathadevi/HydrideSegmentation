#!/usr/bin/env python3
"""
hydride_batch_augmenter.py

Walk an input folder, find every (image, mask) pair, and create N
Albumentations-based augmentations for each pair.  All augmented
images/masks are written to a user-chosen output folder.

Author : ChatGPT (o3) – June 2025
"""

import os
import cv2
import uuid
import logging
import numpy as np
from typing import Dict, List, Tuple
import albumentations as A
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM NOISE TRANSFORMS & FALLBACKS
# ──────────────────────────────────────────────────────────────────────────────
class PoissonNoise(A.ImageOnlyTransform):
    """Pure shot-noise (Poisson) — good for low-count imaging."""
    def __init__(self, lam_range=(20., 60.), *, p: float = 0.25,
                 always_apply: bool = False):
        super().__init__(always_apply=always_apply, p=p)
        self.lam_range = lam_range

    def apply(self, img, **params):
        lam = np.random.uniform(*self.lam_range)
        noisy = np.random.poisson(img.astype(np.float32) / 255. * lam) / lam
        return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)


# try native SaltOrPepperNoise (Albumentations ≥1.4.0); otherwise shim
try:
    _SaltPepper = A.SaltOrPepperNoise
except AttributeError:
    class _SaltPepper(A.ImageOnlyTransform):
        def __init__(self, *, amount=0.01, salt_vs_pepper=0.5,
                     p: float = 0.25, always_apply: bool = False):
            """
            amount           – fraction of pixels to corrupt (0–1)
            salt_vs_pepper   – 0→all pepper, 1→all salt
            """
            super().__init__(always_apply=always_apply, p=p)
            self.amount = amount
            self.svp    = salt_vs_pepper

        def apply(self, img, **params):
            img = img.copy()
            h, w = img.shape[:2]
            n_pix = int(self.amount * h * w)
            ys = np.random.randint(0, h, n_pix)
            xs = np.random.randint(0, w, n_pix)
            split = int(n_pix * self.svp)
            # salt
            img[ys[:split],  xs[:split]]  = 255
            # pepper
            img[ys[split:], xs[split:]] = 0
            return img
# ──────────────────────────────────────────────────────────────────────────────
#                               LOGGER
# ──────────────────────────────────────────────────────────────────────────────
def setup_logger(out_dir: str) -> logging.Logger:
    log_file = os.path.join(out_dir, "augmentation.log")
    logger = logging.getLogger("HydrideAugmenter")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:                         # avoid duplicates
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
#                          AUGMENTATION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def build_transform(cfg: Dict) -> A.Compose:
    aug  = cfg.get("augmentations", {})
    t: List[A.BasicTransform] = []

    # ▸ Geometry / photometric -------------------------------------------------
    if aug.get("rotate", {}).get("enable", False):
        rc = aug["rotate"]
        t.append(A.Rotate(limit=10, border_mode=0, crop_border=True,
                          p=rc.get("prob", 0.9)))

    if aug.get("random_crop", {}).get("enable", False):
        rc = aug["random_crop"]
        t.append(A.RandomCrop(height=rc.get("size", (256, 256))[0],
                              width= rc.get("size", (256, 256))[1],
                              p=rc.get("prob", 0.9)))

    if aug.get("brightness_contrast", {}).get("enable", False):
        bc = aug["brightness_contrast"]
        t.append(A.RandomBrightnessContrast(
            brightness_limit=bc.get("brightness_limit", (-0.2, 0.2)),
            contrast_limit=  bc.get("contrast_limit",   (-0.2, 0.2)),
            p=bc.get("prob", 0.8)))

    if aug.get("h_flip", {}).get("enable", False):
        t.append(A.HorizontalFlip(p=aug["h_flip"].get("prob", 0.5)))

    if aug.get("v_flip", {}).get("enable", False):
        t.append(A.VerticalFlip(p=aug["v_flip"].get("prob", 0.5)))

    if aug.get("random_rotate90", {}).get("enable", False):
        t.append(A.RandomRotate90(p=aug["random_rotate90"].get("prob", 0.5)))

    # ▸ Noise transforms  (each with its own probability) ----------------------
    if aug.get("gaussian_noise", {}).get("enable", False):
        gn = aug["gaussian_noise"]
        t.append(A.GaussNoise(var_limit=gn.get("var_limit", (10., 50.)),
                              p=gn.get("prob", 0.3)))

    if aug.get("iso_noise", {}).get("enable", False):
        iso = aug["iso_noise"]
        t.append(A.ISONoise(color_shift=iso.get("color_shift", (0.9, 0.9)),
                            intensity=iso.get("intensity", (0.8, 0.9)),
                            p=iso.get("prob", 0.2)))

    if aug.get("multiplicative_noise", {}).get("enable", False):
        mn = aug["multiplicative_noise"]
        t.append(A.MultiplicativeNoise(multiplier=mn.get("multiplier", (0.9, 1.1)),
                                       per_channel=mn.get("per_channel", True),
                                       elementwise=mn.get("elementwise", True),
                                       p=mn.get("prob", 0.2)))

    # if aug.get("salt_pepper_noise", {}).get("enable", False):
    #     sp = aug["salt_pepper_noise"]
    #     t.append(_SaltPepper(amount=sp.get("amount", sp.get("prob", 0.01)),
    #                          salt_vs_pepper=sp.get("salt_vs_pepper", 0.5),
    #                          p=sp.get("prob_apply", 0.25)))
    #
    # if aug.get("poisson_noise", {}).get("enable", False):
    #     pn = aug["poisson_noise"]
    #     t.append(PoissonNoise(lam_range=pn.get("lam_range", (20., 60.)),
    #                           p=pn.get("prob", 0.25)))

    return A.Compose(t, additional_targets={"mask": "mask"})


# ──────────────────────────────────────────────────────────────────────────────
#                       UTILS – FINDING PAIRS
# ──────────────────────────────────────────────────────────────────────────────
def find_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """Return [(image_path, mask_path)] where stem.jpg & stem.png both exist."""
    stems = {}
    for fn in os.listdir(input_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in {".jpg", ".jpeg", ".png"}:
            stems.setdefault(stem, {})[ext.lower()] = os.path.join(input_dir, fn)
    return [(v[".jpg"], v[".png"]) for v in stems.values()
            if ".jpg" in v and ".png" in v]


# ──────────────────────────────────────────────────────────────────────────────
#                              CORE WORK
# ──────────────────────────────────────────────────────────────────────────────
def augment_batch(cfg: Dict) -> None:
    in_dir, out_dir = cfg["input_dir"], cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    log = setup_logger(out_dir)
    log.info("Scanning input dir: %s", in_dir)

    pairs = find_pairs(in_dir)
    if not pairs:
        log.error("No (.jpg + .png) pairs found.")
        return
    log.info("Found %d image–mask pairs", len(pairs))

    N     = int(cfg.get("num_augments", 5))
    debug = bool(cfg.get("debug", False))
    trans = build_transform(cfg)

    g_imgs, g_msks = [], []

    for idx, (ipath, mpath) in enumerate(pairs, 1):
        if cfg.get("debug",False):
            if idx>3:
                log.warning(f"debug mode active and hence skipping processing beyond idx=3")
                break

        img = cv2.imread(ipath);
        #msk = cv2.imread(mpath) ##, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)[:, :, 3]
        if img is None or msk is None:
            log.warning("Skipping unreadable pair: %s / %s", ipath, mpath); continue
        msk = (msk > 0).astype(np.uint8)
        log.info("[%d/%d] %s", idx, len(pairs), os.path.basename(ipath))

        stem = os.path.splitext(os.path.basename(ipath))[0]
        for k in range(N):
            aug = trans(image=img, mask=msk); a_img, a_msk = aug["image"], aug["mask"]
            #uid = uuid.uuid4().hex[:8]
            uid = ""
            #cv2.imwrite(os.path.join(out_dir, f"{stem}_{k:02d}_{uid}.jpg"), a_img)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{k:02d}{uid}.jpg"), cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{k:02d}{uid}.png"), a_msk)
            if debug:
                g_imgs.append(a_img[..., ::-1]); g_msks.append(a_msk)

    log.info("Finished. Augmented data in %s", out_dir)

    if debug and g_imgs:

        rows = len(g_imgs)
        if rows>9:
            rows = 3 ### just to plot opnly first three rows of images
        fig, ax = plt.subplots(rows, 2, figsize=(6, 3*rows), squeeze=False)
        for r in range(rows):
            ax[r, 0].imshow(g_imgs[r]); ax[r, 0].axis('off')
            ax[r, 1].imshow(g_msks[r], cmap='gray'); ax[r, 1].axis('off')
        plt.tight_layout(); plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#                               DEMO CONFIG
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_config = {
        "input_dir":  r"L:\maniBackUp\HydrideSegmentation\HydrideMask_Manual\dataset_raw",
        "output_dir": r"L:\maniBackUp\HydrideSegmentation\HydrideMask_Manual\Hydride_dataset_5.0",
        "num_augments": 20,
        "debug": False,

        "augmentations": {
            # geometry / photometric
    #if aug.get("rotate", {}).get("enable", False):
            "rotate":        {"enable": True,  "prob": 0.6},
            "random_crop":        {"enable": True,  "size": (320, 320), "prob": 0.7},
            "brightness_contrast":{"enable": True,  "brightness_limit": (-0.5, 0.5),
                                 "contrast_limit":   (-0.2, 0.2), "prob": 0.},
            "h_flip":             {"enable": True,  "prob": 0.5},
            "v_flip":             {"enable": True, "prob": 0.5},
            "random_rotate90":    {"enable": True,  "prob": 0.5},

            # noise family (each has its own p → any combination possible)
            "gaussian_noise":       {"enable": True,  "var_limit": (200, 300), "prob": 0.4},
            "iso_noise":            {"enable": True,  "color_shift": (0.6,1.0),
                                                     "intensity": (0.6,1.0), "prob": 0.3},
            "multiplicative_noise": {"enable": True,  "multiplier": (0.2,2),
                                                     "per_channel": True,
                                                     "elementwise": True,
                                                     "prob": 0.3},
            "salt_pepper_noise":    {"enable": True,  "amount": 0.01,
                                                     "salt_vs_pepper": 0.5,
                                                     "prob_apply": 0.25},
            "poisson_noise":        {"enable": True,  "lam_range": (20, 60), "prob": 0.25}
        }
    }

    augment_batch(example_config)
