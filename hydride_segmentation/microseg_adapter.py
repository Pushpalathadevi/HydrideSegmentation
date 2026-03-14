"""Compatibility adapter bridging existing hydride paths to microseg pipeline."""

from __future__ import annotations

from src.microseg.domain import SegmentationRequest
from src.microseg.inference import build_hydride_registry
from src.microseg.plugins import frozen_checkpoint_map
from src.microseg.pipelines import SegmentationPipeline


LEGACY_GUI_MODEL_TO_ID = {
    "Conventional Model": "hydride_conventional",
    "ML Model": "hydride_ml",
}


def get_gui_model_options() -> list[str]:
    """Return model display names for desktop GUI selectors."""

    specs = build_hydride_registry().specs()
    return [spec.display_name for spec in specs]


def get_gui_model_specs() -> list[dict[str, str]]:
    """Return model metadata for GUI descriptions and selection help."""

    specs = build_hydride_registry().specs()
    try:
        frozen = frozen_checkpoint_map()
    except Exception:
        frozen = {}
    return [
        {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "feature_family": spec.feature_family,
            "description": spec.description,
            "details": spec.details,
            "model_nickname": frozen[spec.model_id].model_nickname if spec.model_id in frozen else "",
            "model_type": frozen[spec.model_id].model_type if spec.model_id in frozen else "",
            "framework": frozen[spec.model_id].framework if spec.model_id in frozen else "",
            "input_size": frozen[spec.model_id].input_size if spec.model_id in frozen else "",
            "input_dimensions": frozen[spec.model_id].input_dimensions if spec.model_id in frozen else "",
            "checkpoint_path_hint": frozen[spec.model_id].checkpoint_path_hint if spec.model_id in frozen else "",
            "application_remarks": frozen[spec.model_id].application_remarks if spec.model_id in frozen else "",
            "short_description": frozen[spec.model_id].short_description if spec.model_id in frozen else "",
            "detailed_description": frozen[spec.model_id].detailed_description if spec.model_id in frozen else "",
            "artifact_stage": frozen[spec.model_id].artifact_stage if spec.model_id in frozen else "",
            "source_run_manifest": frozen[spec.model_id].source_run_manifest if spec.model_id in frozen else "",
            "quality_report_path": frozen[spec.model_id].quality_report_path if spec.model_id in frozen else "",
        }
        for spec in specs
    ]


def resolve_gui_model_id(model_name: str) -> str:
    """Resolve a GUI model label (new or legacy) to a model identifier."""

    specs = build_hydride_registry().specs()
    display_map = {spec.display_name: spec.model_id for spec in specs}
    if model_name in display_map:
        return display_map[model_name]
    return LEGACY_GUI_MODEL_TO_ID.get(model_name, "hydride_ml")


def is_conventional_model(model_name: str) -> bool:
    """Return whether selected model uses conventional parameters."""

    return resolve_gui_model_id(model_name) == "hydride_conventional"


def run_pipeline(
    image_path: str,
    *,
    model_id: str = "hydride_conventional",
    params: dict | None = None,
    include_analysis: bool = True,
):
    """Run segmentation through the microseg orchestration layer."""

    pipeline = SegmentationPipeline.with_hydride_defaults()
    request = SegmentationRequest(
        image_path=image_path,
        model_id=model_id,
        params=params or {},
        include_analysis=include_analysis,
    )
    return pipeline.run(request)


def run_pipeline_from_gui(
    image_path: str,
    model_name: str,
    params: dict | None = None,
    *,
    include_analysis: bool = False,
):
    """Map GUI model names to model identifiers."""

    model_id = resolve_gui_model_id(model_name)
    return run_pipeline(
        image_path,
        model_id=model_id,
        params=params,
        include_analysis=include_analysis,
    )
