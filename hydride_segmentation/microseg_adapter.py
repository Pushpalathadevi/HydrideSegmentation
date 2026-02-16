"""Compatibility adapter bridging existing hydride paths to microseg pipeline."""

from __future__ import annotations

from src.microseg.domain import SegmentationRequest
from src.microseg.inference import build_hydride_registry
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
    return [
        {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "feature_family": spec.feature_family,
            "description": spec.description,
            "details": spec.details,
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
    """Run segmentation through the phase-1 microseg orchestration layer."""

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
    """Map existing GUI model names to microseg model identifiers."""

    model_id = resolve_gui_model_id(model_name)
    return run_pipeline(
        image_path,
        model_id=model_id,
        params=params,
        include_analysis=include_analysis,
    )
