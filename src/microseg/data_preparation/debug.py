"""Debug inspection artifact generation for dataset preparation."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.microseg.data_preparation.exporters import write_image

matplotlib.use("Agg")


class DebugInspector:
    """Write inspection panels and optional interactive plots."""

    def write(
        self,
        *,
        debug_root: Path,
        split: str,
        stem: str,
        image_raw: np.ndarray,
        mask_raw: np.ndarray,
        mask_binary: np.ndarray,
        show: bool,
        ext: str,
        draw_contours: bool,
        annotation: str,
    ) -> None:
        debug_root.mkdir(parents=True, exist_ok=True)
        raw_vis = self._to_display(mask_raw)
        bin_vis = (mask_binary.astype(np.uint8) * 255)
        overlay = self._overlay(image_raw, mask_binary, draw_contours=draw_contours)

        base = debug_root / split / stem
        write_image(base.with_name(f"{stem}_image{ext}"), image_raw)
        write_image(base.with_name(f"{stem}_mask_raw{ext}"), raw_vis)
        write_image(base.with_name(f"{stem}_mask_binary{ext}"), bin_vis)
        write_image(base.with_name(f"{stem}_overlay{ext}"), overlay)

        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        axs[0].imshow(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB) if image_raw.ndim == 3 else image_raw, cmap="gray")
        axs[0].set_title("original")
        axs[1].imshow(raw_vis, cmap="gray")
        axs[1].set_title("raw mask")
        axs[2].imshow(bin_vis, cmap="gray")
        axs[2].set_title("binarized")
        axs[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[3].set_title("overlay")
        for ax in axs:
            ax.axis("off")
        fig.suptitle(annotation)
        fig.tight_layout()
        panel_path = base.with_name(f"{stem}_panel{ext}")
        fig.savefig(panel_path)
        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _to_display(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            return mask
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _overlay(image: np.ndarray, mask: np.ndarray, *, draw_contours: bool) -> np.ndarray:
        base = image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        color = np.zeros_like(base)
        color[:, :, 2] = (mask > 0).astype(np.uint8) * 255
        blended = cv2.addWeighted(base, 0.7, color, 0.3, 0)
        if draw_contours:
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)
        return blended
