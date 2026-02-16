"""High-level segmentation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from hydride_segmentation.core.utils import image_to_png_base64

from src.microseg.domain import PipelineResult, SegmentationRequest, utc_timestamp
from src.microseg.evaluation import HydrideAnalyzer
from src.microseg.inference import build_hydride_registry
from src.microseg.plugins import ModelRegistry
from src.microseg.utils import mask_overlay


@dataclass
class SegmentationPipeline:
    """Pipeline that routes requests through registry-backed predictors."""

    registry: ModelRegistry

    @classmethod
    def with_hydride_defaults(cls) -> "SegmentationPipeline":
        return cls(registry=build_hydride_registry())

    def run(self, request: SegmentationRequest) -> PipelineResult:
        predictor = self.registry.build(request.model_id)
        output = predictor.predict(request.image_path, request.params)

        overlay = mask_overlay(output.image, output.mask)
        images_b64 = {
            "input_png_b64": image_to_png_base64(output.image),
            "mask_png_b64": image_to_png_base64(output.mask),
            "overlay_png_b64": image_to_png_base64(overlay),
        }
        metrics: dict[str, float | int] = {}

        if request.include_analysis:
            analyzer = HydrideAnalyzer()
            report = analyzer.analyze(output.image, output.mask)
            metrics = report.metrics
            images_b64.update(report.analysis_images_b64)

        manifest = {
            "pipeline": "microseg.segmentation",
            "version": "phase1",
            "timestamp_utc": utc_timestamp(),
            "model_id": request.model_id,
            "include_analysis": request.include_analysis,
        }

        return PipelineResult(
            ok=True,
            model_id=request.model_id,
            image=output.image,
            mask=output.mask,
            metrics=metrics,
            images_b64=images_b64,
            manifest=manifest,
        )
