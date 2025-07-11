import argparse
import logging
import random
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon, rectangle
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SegmentationEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.image_size = config.get("image_size", (100, 200))
        self.simulate = config.get("simulate", True)
        self.plot = config.get("plot", True)
        self.tamper_ratio = config.get("tamper_ratio", 0.2)
        self.report_path = config.get("report_path", "segmentation_report.txt")

        self.input_image = None
        self.gt_mask = None
        self.pred_mask = None

    def run(self):
        """Run data generation/loading, metric calculation and reporting."""
        logging.info("Starting segmentation evaluation.")
        if self.simulate:
            logging.info("Generating simulated data.")
            self.generate_simulated_data()
        else:
            logging.info("Loading real input data.")
            self.load_real_data()

        logging.info("Computing metrics.")
        metrics = self.compute_metrics()

        logging.info("Writing results to report file.")
        self.write_report(metrics)

        if self.plot:
            logging.info("Generating visualizations.")
            self.plot_results(metrics)

        return metrics

    def generate_simulated_data(self):
        """Create a set of toy rectangles/triangles for testing."""
        h, w = self.image_size
        self.input_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        self.gt_mask = np.zeros((h, w), dtype=np.uint8)

        for _ in range(5):
            shape_type = random.choice(['rectangle', 'triangle'])
            x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
            x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)

            if shape_type == 'rectangle':
                rr, cc = rectangle(start=(y1, x1), end=(y2, x2))
            else:
                r = np.array([y1, y2, y1])
                c = np.array([x1, x1 + (x2 - x1) // 2, x2])
                rr, cc = polygon(r, c)

            rr, cc = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
            self.input_image[rr, cc] = (0, 0, 255)
            self.gt_mask[rr, cc] = 1

        self.pred_mask = self.tamper_mask(self.gt_mask, self.tamper_ratio)

    def tamper_mask(self, mask: np.ndarray, ratio: float) -> np.ndarray:
        """Flip a fraction of pixels to mimic imperfect predictions."""
        tampered = mask.copy()
        h, w = mask.shape
        num_pixels = int(h * w * ratio)
        indices = np.unravel_index(np.random.choice(h * w, num_pixels, replace=False), (h, w))
        tampered[indices] = 1 - tampered[indices]
        return tampered

    def load_real_data(self):
        """Read images and masks from disk for evaluation."""
        self.input_image = cv2.imread(self.config['input_image_path'])
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        self.gt_mask = cv2.imread(self.config['ground_truth_path'], cv2.IMREAD_GRAYSCALE)
        self.pred_mask = cv2.imread(self.config['predicted_mask_path'], cv2.IMREAD_GRAYSCALE)
        self.gt_mask = (self.gt_mask > 0).astype(np.uint8)
        if np.sum(self.gt_mask) == 0:
            logging.warning(f"looks like ground truth mask is not scaled to 255 hence performing thrsholding >0!!")
            self.gt_mask = (self.gt_mask > 0).astype(np.uint8)
        self.pred_mask = (self.pred_mask > 127).astype(np.uint8)

    def compute_metrics(self) -> Dict:
        """Return IoU, Dice and other metrics as a dictionary."""
        gt = self.gt_mask.flatten()
        pred = self.pred_mask.flatten()

        iou = np.sum((gt & pred)) / np.sum((gt | pred)) if np.sum((gt | pred)) > 0 else 1.0
        precision = precision_score(gt, pred, zero_division=1)
        recall = recall_score(gt, pred, zero_division=1)
        f1 = f1_score(gt, pred, zero_division=1)
        dice = 2 * np.sum(gt * pred) / (np.sum(gt) + np.sum(pred)) if (np.sum(gt) + np.sum(pred)) > 0 else 1.0
        mse = mean_squared_error(gt, pred)
        accuracy = np.mean(gt == pred)

        return {
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Dice Coefficient": dice,
            "MSE": mse,
            "Accuracy": accuracy
        }

    def write_report(self, metrics: Dict):
        """Write metrics to a text report file."""
        with open(self.report_path, 'w') as f:
            f.write("Segmentation Evaluation Report\n")
            f.write("==============================\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        logging.info(f"Report written to {self.report_path}")

    def plot_results(self, metrics: Dict):
        """Visualize ground truth, prediction and overlay with scores."""
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))

        axs[0].imshow(self.input_image)
        axs[0].set_title("Input Image")

        axs[1].imshow(self.gt_mask, cmap='gray')
        axs[1].set_title("Ground Truth")

        axs[2].imshow(self.pred_mask, cmap='gray')
        axs[2].set_title("Prediction")

        overlay = self.input_image.copy()
        overlay[self.gt_mask == 1] = [0, 255, 0]  # Green for ground truth
        overlay[self.pred_mask == 1] = [255, 0, 0]  # Blue for prediction
        axs[3].imshow(overlay)
        axs[3].set_title("Overlay")

        metric_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        axs[3].text(1.05, 0.5, metric_text, transform=axs[3].transAxes,
                    fontsize=12, verticalalignment='center', bbox=dict(boxstyle="round", facecolor='white'))

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def main() -> None:
    """Entry point allowing simulation for remote development."""
    p = argparse.ArgumentParser(description="Evaluate segmentation masks")
    p.add_argument("--simulate", action="store_true",
                   help="use generated shapes instead of real files")
    p.add_argument("--plot", action="store_true", help="display result figures")
    args = p.parse_args()

    cfg = {
        "simulate": args.simulate,
        "plot": args.plot,
        "image_size": (100, 200),
        "tamper_ratio": 0.2,
        "input_image_path": "input.png",
        "ground_truth_path": "gt.png",
        "predicted_mask_path": "pred.png",
        "report_path": "segmentation_report.txt",
    }

    evaluator = SegmentationEvaluator(cfg)
    results = evaluator.run()
    logging.info("Computed Metrics: %s", results)


if __name__ == "__main__":
    main()
