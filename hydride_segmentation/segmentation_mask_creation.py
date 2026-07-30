# hydride_segmentation.py
import logging
import os
import shutil
import zipfile
from collections.abc import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class HydrideSegmentation:
    """
    Segment elongated zirconium-hydride plates, produce
      • ORA file with two layers (background & mask)
      • stand-alone PNG mask at user-defined location
    """
    def __init__(self, settings: dict):
        self.settings          = settings
        self.image_path        = settings['image_path']
        self.output_path       = settings.get('output_path', 'GUI')
        self.mask_output_path  = settings.get('mask_output_path', self.output_path)
        self.debug             = settings.get('debug', False)
        self.plot              = settings.get('plot',  False)
        self.crop              = settings.get('crop', False)
        self.crop_percent      = settings.get('crop_percent', 0)
        self.logger            = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("HydrideSegmentation")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(ch)
        return logger

    def load_image(self):
        full_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.original_image = full_image.copy()  # Save original for plotting
        if self.crop:
            crop_rows = int(full_image.shape[0] * self.crop_percent / 100)
            self.crop_line_y = full_image.shape[0] - crop_rows
            full_image = full_image[:self.crop_line_y, :]
            self.logger.info(f"Image cropped from bottom by {self.crop_percent}% → {full_image.shape}")
        else:
            self.crop_line_y = None
        self.image  = full_image
        self.height, self.width = self.image.shape
        self.logger.info(f"Loaded image: {self.image_path}  →  {self.image.shape}")

    def enhance_contrast(self):
        clahe = cv2.createCLAHE(
            clipLimit    = self.settings['clahe']['clip_limit'],
            tileGridSize = tuple(self.settings['clahe']['tile_grid_size'])
        )
        self.enhanced_img = clahe.apply(self.image)
        self.logger.debug("Contrast enhanced with CLAHE")

    def threshold_image(self):
        blur  = cv2.GaussianBlur(self.enhanced_img, (5, 5), 0)
        blk   = self.settings['adaptive']['block_size']
        C     = self.settings['adaptive']['C']
        self.thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, blk, C
        )
        self.logger.debug(f"Adaptive thresholding (block={blk}, C={C})")

    def close_holes(self):
        ksz        = tuple(self.settings['morph']['kernel_size'])
        iterations = self.settings['morph']['iterations']
        kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, ksz)
        self.closed_img = cv2.morphologyEx(
            self.thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        self.logger.debug(f"Morphological closing (kernel={ksz}, its={iterations})")

    def filter_regions(self):
        area_th = self.settings['area_threshold']
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
            self.closed_img, connectivity=8
        )
        self.mask = np.zeros_like(self.closed_img)
        keep = 0
        for lab in range(1, n_lbl):
            if stats[lab, cv2.CC_STAT_AREA] >= area_th:
                self.mask[lbl == lab] = 255
                keep += 1
        self.logger.info(f"Regions kept: {keep}, area ≥ {area_th}px")

    def plot_intermediate_results(self):
        if not self.debug:
            return

        if self.crop:
            fig, ax = plt.subplots(2, 3, figsize=(18, 10))
            ax = ax.flatten()

            # Original image with RED horizontal line
            orig_disp = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_GRAY2BGR)
            if self.crop_line_y:
                cv2.line(orig_disp, (0, self.crop_line_y), (self.width, self.crop_line_y), (255, 0, 0), 2)  # Red line
            ax[0].imshow(orig_disp)
            ax[0].set_title('Original (Uncropped) with crop line')
            ax[0].axis('off')

            # Cropped image with annotation
            ax[1].imshow(self.image, cmap='gray')
            ax[1].set_title(f'Cropped Image\n({self.crop_percent}% from bottom)')
            ax[1].axis('off')

            ax[2].imshow(self.enhanced_img, cmap='gray');
            ax[2].set_title('CLAHE');
            ax[2].axis('off')
            ax[3].imshow(self.thresh, cmap='gray');
            ax[3].set_title('Adaptive Mask');
            ax[3].axis('off')
            ax[4].imshow(self.closed_img, cmap='gray');
            ax[4].set_title('After Closing');
            ax[4].axis('off')
            ax[5].imshow(self.mask, cmap='gray');
            ax[5].set_title(f'Filtered ≥{self.settings["area_threshold"]}');
            ax[5].axis('off')

        else:
            fig, ax = plt.subplots(2, 3, figsize=(18, 8))
            ax = ax.flatten()

            ax[0].imshow(self.image, cmap='gray');
            ax[0].set_title('Original');
            ax[0].axis('off')
            ax[1].imshow(self.enhanced_img, cmap='gray');
            ax[1].set_title('CLAHE');
            ax[1].axis('off')
            ax[2].imshow(self.thresh, cmap='gray');
            ax[2].set_title('Adaptive Mask');
            ax[2].axis('off')
            ax[3].imshow(self.closed_img, cmap='gray');
            ax[3].set_title('After Closing');
            ax[3].axis('off')
            ax[4].imshow(self.mask, cmap='gray');
            ax[4].set_title(f'Filtered ≥{self.settings["area_threshold"]}');
            ax[4].axis('off')
            fig.delaxes(ax[5])  # 6th subplot not needed

        plt.tight_layout()
        plt.show()

    def visualize(self):
        if not self.plot: return
        rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(self.image, cmap='gray'); ax[0].set_title('Cropped Image');   ax[0].axis('off')
        ax[1].imshow(self.mask,  cmap='gray'); ax[1].set_title('Hydride Mask');    ax[1].axis('off')
        ax[2].imshow(rgb);                     ax[2].set_title('Overlay (Red)');   ax[2].axis('off')
        plt.tight_layout(); plt.show()

    def export_mask_png(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        png_out   = os.path.join(self.mask_output_path, f"{base_name}_mask.png")
        os.makedirs(self.mask_output_path, exist_ok=True)
        cv2.imwrite(png_out, self.mask)
        self.logger.info(f"✅  Mask PNG written → {png_out}")

    def export_to_ora(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        ora_out   = os.path.join(self.output_path, f"{base_name}.ora")

        os.makedirs('temp/layerstack', exist_ok=True)
        Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)).save('temp/layerstack/0000.png')

        rgba = np.zeros((self.height, self.width, 4), np.uint8)
        rgba[..., 0] = 255; rgba[..., 3] = self.mask
        Image.fromarray(rgba).save('temp/layerstack/0001.png')

        stack_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<image w="{self.width}" h="{self.height}" version="0.0.1">
  <stack>
    <layer name="hydride_mask" src="layerstack/0001.png"/>
    <layer name="background"  src="layerstack/0000.png"/>
  </stack>
</image>"""
        with open('temp/stack.xml', 'w') as f: f.write(stack_xml)

        with zipfile.ZipFile(ora_out, 'w') as ora:
            ora.write('temp/stack.xml',           'stack.xml')
            ora.write('temp/layerstack/0000.png', 'layerstack/0000.png')
            ora.write('temp/layerstack/0001.png', 'layerstack/0001.png')

        shutil.rmtree('temp')
        self.logger.info(f"✅  ORA file written → {ora_out}")

    def save_input_image(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        out_path = os.path.join(self.mask_output_path, base + "_input.png")
        cv2.imwrite(out_path, self.image)
        self.logger.info(f"✅ Input image saved → {out_path}")

    def run(self):
        self.load_image()
        self.enhance_contrast()
        self.threshold_image()
        self.close_holes()
        self.filter_regions()
        self.plot_intermediate_results()
        self.visualize()
        #self.export_mask_png()
        self.export_to_ora()
        #self.save_input_image()

# ------------------------------------------------------------- SETTINGS

def run_model(image_path, params):
    settings = {
        'image_path': image_path,
        'clahe': params['clahe'],
        'adaptive': params['adaptive'],
        'morph': params['morph'],
        'area_threshold': params['area_threshold'],
        'crop': params['crop'],
        'crop_percent': params['crop_percent'],
        'debug': False,
        'plot': False,
        'output_path': '.',
        'mask_output_path': '.'
    }

    segmenter = HydrideSegmentation(settings)
    segmenter.load_image()
    segmenter.enhance_contrast()
    segmenter.threshold_image()
    segmenter.close_holes()
    segmenter.filter_regions()

    return segmenter.image, segmenter.mask


def run_model_array(
    image: np.ndarray,
    params: dict,
    *,
    progress_hook: Callable[[str, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run conventional hydride segmentation entirely in memory.

    Parameters
    ----------
    image:
        Grayscale or RGB image array. RGB channel order is expected for
        three-channel input.
    params:
        Conventional segmentation parameters using the same structure as
        :func:`run_model`.
    progress_hook:
        Optional callback receiving ``stage``, integer ``percent``, and a
        user-facing message.

    Returns
    -------
    tuple of numpy.ndarray
        The processed grayscale image and binary uint8 mask.
    """

    def emit(stage: str, percent: int, message: str) -> None:
        if progress_hook is not None:
            progress_hook(stage, int(percent), message)

    arr = np.asarray(image)
    if arr.ndim == 2:
        grayscale = arr.astype(np.uint8, copy=True)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        grayscale = arr[:, :, 0].astype(np.uint8, copy=True)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        grayscale = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"unsupported in-memory image shape: {arr.shape!r}")

    emit("preprocessing", 18, "Converted the image to an 8-bit grayscale analysis array.")
    if bool(params.get("crop", False)):
        crop_percent = float(params.get("crop_percent", 0.0))
        crop_rows = int(grayscale.shape[0] * crop_percent / 100.0)
        if crop_rows > 0:
            grayscale = grayscale[: grayscale.shape[0] - crop_rows, :]
        emit("preprocessing", 24, f"Applied the configured {crop_percent:g}% bottom crop.")
    else:
        emit("preprocessing", 24, "No crop was requested.")

    clahe_cfg = params["clahe"]
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_cfg["clip_limit"]),
        tileGridSize=tuple(int(v) for v in clahe_cfg["tile_grid_size"]),
    )
    enhanced = clahe.apply(grayscale)
    emit("segmentation", 35, "Enhanced local contrast with CLAHE.")

    adaptive_cfg = params["adaptive"]
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        int(adaptive_cfg["block_size"]),
        int(adaptive_cfg["C"]),
    )
    emit("segmentation", 48, "Applied adaptive local thresholding.")

    morph_cfg = params["morph"]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        tuple(int(v) for v in morph_cfg["kernel_size"]),
    )
    closed = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=int(morph_cfg["iterations"]),
    )
    emit("postprocessing", 58, "Closed short gaps using the configured morphology settings.")

    area_threshold = int(params["area_threshold"])
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8,
    )
    mask = np.zeros_like(closed, dtype=np.uint8)
    kept = 0
    for label_id in range(1, int(n_labels)):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= area_threshold:
            mask[labels == label_id] = 255
            kept += 1
    emit(
        "postprocessing",
        68,
        f"Retained {kept} connected feature(s) at or above {area_threshold} pixels.",
    )
    return grayscale.astype(np.uint8, copy=False), mask

