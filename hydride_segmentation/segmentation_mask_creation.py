# hydride_segmentation.py
import logging
import os
import shutil
import zipfile

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
if __name__ == "__main__":
    settings = {
        'image_path': r'L:\maniBackUp\HydrideSegmentation\HydridedImagesForTraining\3PB_SRT_7293_ASh_ASH_from_3PB_2.png',

        'area_threshold': 150,
        'clahe':    {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
        'adaptive': {'block_size': 13, 'C': 20},
        'morph':    {'kernel_size': [5, 5], 'iterations': 0},

        'plot':  True,
        'debug': True,

        'output_path': r'L:\maniBackUp\HydrideSegmentation\HydridedImagesForTraining',
        'mask_output_path': r'L:\maniBackUp\HydrideSegmentation\HydrideMask',

        'crop': False,
        'crop_percent': 10  # Crops 10% from bottom, changeable anytime
    }

    HydrideSegmentation(settings).run()

# Add this at the END of segmentationMaskCreation.py

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

