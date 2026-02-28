"""Debug inspection artifact generation for dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        image_input: np.ndarray,
        image_output: np.ndarray,
        mask_input: np.ndarray,
        mask_processed: np.ndarray,
        criteria: dict[str, Any],
        show: bool,
        ext: str,
        draw_contours: bool,
        annotation: str,
    ) -> None:
        debug_root.mkdir(parents=True, exist_ok=True)
        input_mask_vis = self._to_display(mask_input)
        processed_mask_vis = (mask_processed.astype(np.uint8) * 255)
        input_mask_for_diff = input_mask_vis
        if input_mask_for_diff.shape != processed_mask_vis.shape:
            input_mask_for_diff = cv2.resize(
                input_mask_for_diff,
                (processed_mask_vis.shape[1], processed_mask_vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        diff_vis = cv2.absdiff(input_mask_for_diff.astype(np.uint8), processed_mask_vis.astype(np.uint8))
        overlay = self._overlay(image_output, mask_processed, draw_contours=draw_contours)
        output_image_vis = (
            image_output
            if image_output.shape[:2] == processed_mask_vis.shape[:2]
            else cv2.resize(image_output, (processed_mask_vis.shape[1], processed_mask_vis.shape[0]), interpolation=cv2.INTER_AREA)
        )

        base = debug_root / split / stem
        write_image(base.with_name(f"{stem}_image_input{ext}"), image_input)
        write_image(base.with_name(f"{stem}_image_output{ext}"), output_image_vis)
        write_image(base.with_name(f"{stem}_image{ext}"), output_image_vis)
        write_image(base.with_name(f"{stem}_mask_input{ext}"), input_mask_vis)
        write_image(base.with_name(f"{stem}_mask_raw{ext}"), input_mask_vis)
        write_image(base.with_name(f"{stem}_mask_processed{ext}"), processed_mask_vis)
        write_image(base.with_name(f"{stem}_mask_binary{ext}"), processed_mask_vis)
        write_image(base.with_name(f"{stem}_mask_difference{ext}"), diff_vis)
        write_image(base.with_name(f"{stem}_overlay{ext}"), overlay)
        criteria_path = base.with_name(f"{stem}_criteria.json")
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_text(json.dumps(self._json_safe(criteria), indent=2, sort_keys=True), encoding="utf-8")

        fig, axs = plt.subplots(1, 6, figsize=(23, 4))
        axs[0].imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.ndim == 3 else image_input, cmap="gray")
        axs[0].set_title("input image")
        axs[1].imshow(input_mask_vis, cmap="gray")
        axs[1].set_title("input mask")
        axs[2].imshow(cv2.cvtColor(output_image_vis, cv2.COLOR_BGR2RGB) if output_image_vis.ndim == 3 else output_image_vis, cmap="gray")
        axs[2].set_title("output image")
        axs[3].imshow(processed_mask_vis, cmap="gray")
        axs[3].set_title("processed mask")
        axs[4].imshow(diff_vis, cmap="inferno")
        axs[4].set_title("mask diff")
        axs[5].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[5].set_title("overlay")
        for ax in axs:
            ax.axis("off")
        fig.suptitle(f"{annotation}\ncriteria={self._criteria_text(criteria)}")
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
        if mask.shape[2] == 4:
            return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
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

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): DebugInspector._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DebugInspector._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _criteria_text(criteria: dict[str, Any]) -> str:
        safe = DebugInspector._json_safe(criteria)
        mode = str(safe.get("mode", "unknown"))
        threshold = safe.get("threshold")
        fg_ratio = safe.get("fg_ratio")
        if threshold is None:
            threshold = safe.get("thresholds", {})
        return f"mode={mode} threshold={threshold} fg_ratio={fg_ratio}"
