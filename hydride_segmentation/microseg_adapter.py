"""Compatibility adapter bridging existing hydride paths to microseg pipeline."""

from __future__ import annotations

from src.microseg.domain import SegmentationRequest
from src.microseg.inference import build_hydride_registry
from src.microseg.inference.trained_model_loader import (
    InferenceModelReference,
    discover_inference_references,
    load_reference_from_registry,
    load_reference_from_run_dir,
)
from src.microseg.plugins import frozen_checkpoint_map
from src.microseg.pipelines import SegmentationPipeline


LEGACY_GUI_MODEL_TO_ID = {
    "Conventional Model": "hydride_conventional",
    "ML Model": "hydride_ml",
    "Hydride ML (legacy adapter)": "hydride_ml",
    "Registry: legacy hydride_ml_Unet": "hydride_ml",
    "hydride_ml_Unet": "hydride_ml",
}

_GUI_MODEL_ORDER = ("hydride_ml", "hydride_conventional")


def _build_gui_model_specs() -> list[dict[str, str]]:
    """Return the small public model list exposed in GUI/Tk selectors."""

    specs_by_id = {spec.model_id: spec for spec in build_hydride_registry().specs()}
    try:
        frozen = frozen_checkpoint_map()
    except Exception:
        frozen = {}

    payload: list[dict[str, str]] = []
    for model_id in _GUI_MODEL_ORDER:
        spec = specs_by_id.get(model_id)
        if spec is None:
            continue
        frozen_record = frozen.get(model_id)
        display_name = "Hydride ML (UNet)" if model_id == "hydride_ml" else spec.display_name
        payload.append(
            {
                "model_id": spec.model_id,
                "display_name": display_name,
                "feature_family": spec.feature_family,
                "description": spec.description,
                "details": spec.details,
                "model_nickname": frozen_record.model_nickname if frozen_record is not None else "",
                "model_type": frozen_record.model_type if frozen_record is not None else "",
                "framework": frozen_record.framework if frozen_record is not None else "",
                "input_size": frozen_record.input_size if frozen_record is not None else "",
                "input_dimensions": frozen_record.input_dimensions if frozen_record is not None else "",
                "checkpoint_path_hint": frozen_record.checkpoint_path_hint if frozen_record is not None else "",
                "application_remarks": frozen_record.application_remarks if frozen_record is not None else "",
                "short_description": frozen_record.short_description if frozen_record is not None else "",
                "detailed_description": frozen_record.detailed_description if frozen_record is not None else "",
                "artifact_stage": frozen_record.artifact_stage if frozen_record is not None else "",
                "source_run_manifest": frozen_record.source_run_manifest if frozen_record is not None else "",
                "quality_report_path": frozen_record.quality_report_path if frozen_record is not None else "",
                "file_sha256": frozen_record.file_sha256 if frozen_record is not None else "",
                "file_size_bytes": str(frozen_record.file_size_bytes) if frozen_record is not None else "",
            }
        )
    return payload


def get_gui_model_options() -> list[str]:
    """Return model display names for desktop GUI selectors."""

    return [spec["display_name"] for spec in _build_gui_model_specs()]


def get_gui_model_specs() -> list[dict[str, str]]:
    """Return model metadata for GUI descriptions and selection help."""

    return _build_gui_model_specs()

def resolve_gui_model_id(model_name: str) -> str:
    """Resolve a GUI model label (new or legacy) to a model identifier."""

    text = str(model_name).strip()
    specs = build_hydride_registry().specs()
    display_map = {spec.display_name: spec.model_id for spec in specs}
    model_id_map = {spec.model_id: spec.model_id for spec in specs}
    if text in display_map:
        return display_map[text]
    if text in model_id_map:
        return model_id_map[text]
    if text.startswith("hydride_trained::"):
        tail = text.removeprefix("hydride_trained::")
        if tail in model_id_map:
            return tail
        if tail.startswith("registry::"):
            nested = tail.removeprefix("registry::")
            if nested in model_id_map:
                return nested
    if text.startswith("registry::"):
        tail = text.removeprefix("registry::")
        if tail in model_id_map:
            return tail
    return LEGACY_GUI_MODEL_TO_ID.get(text, "hydride_ml")


def is_conventional_model(model_name: str) -> bool:
    """Return whether selected model uses conventional parameters."""

    return resolve_gui_model_id(model_name) == "hydride_conventional"


def resolve_gui_model_reference(model_name: str, params: dict | None = None) -> InferenceModelReference | None:
    """Resolve a GUI model selection to a warm-loadable ML reference when possible."""

    model_id = resolve_gui_model_id(model_name)
    cfg = dict(params or {})
    if model_id == "hydride_conventional":
        return None
    if model_id == "hydride_ml":
        run_dir = str(cfg.get("run_dir", "")).strip()
        registry_model_id = str(cfg.get("registry_model_id", "")).strip()
        checkpoint_path = str(cfg.get("checkpoint_path", cfg.get("weights_path", ""))).strip()
        if run_dir:
            return load_reference_from_run_dir(run_dir)
        if registry_model_id:
            return load_reference_from_registry(registry_model_id)
        if checkpoint_path:
            return InferenceModelReference(
                reference_id=f"checkpoint::{checkpoint_path}",
                display_name="Direct checkpoint",
                source="checkpoint_path",
                checkpoint_path=checkpoint_path,
                architecture=str(cfg.get("model_architecture", "")).strip().lower() or "unknown",
                backend_label=str(cfg.get("backend", "")).strip().lower() or "custom",
            )
        try:
            return load_reference_from_registry("hydride_ml")
        except Exception:
            return None
    try:
        return load_reference_from_registry(model_id)
    except Exception:
        refs, _warnings = discover_inference_references(include_registry=True)
        for ref in refs:
            if ref.reference_id == model_id or ref.display_name == model_name:
                return ref
    return None


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
