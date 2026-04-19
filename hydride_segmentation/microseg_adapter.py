"""Compatibility adapter bridging existing hydride paths to microseg pipeline."""

from __future__ import annotations

from dataclasses import asdict

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
    gui_specs = [
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
            "file_sha256": frozen[spec.model_id].file_sha256 if spec.model_id in frozen else "",
            "file_size_bytes": str(frozen[spec.model_id].file_size_bytes) if spec.model_id in frozen else "",
        }
        for spec in specs
    ]
    if "hydride_ml" in frozen and not any(spec["model_id"] == "hydride_ml_Unet" for spec in gui_specs):
        legacy = dict(asdict(frozen["hydride_ml"]))
        gui_specs.append(
            {
                "model_id": "hydride_ml_Unet",
                "display_name": "Registry: legacy hydride_ml_Unet",
                "feature_family": "hydride",
                "description": "Legacy UNet-compatible frozen-checkpoint alias",
                "details": "Compatibility alias for historical GUI/export records that referenced hydride_ml_Unet.",
                "model_nickname": str(legacy.get("model_nickname", "")),
                "model_type": str(legacy.get("model_type", "")),
                "framework": str(legacy.get("framework", "")),
                "input_size": str(legacy.get("input_size", "")),
                "input_dimensions": str(legacy.get("input_dimensions", "")),
                "checkpoint_path_hint": str(legacy.get("checkpoint_path_hint", "")),
                "application_remarks": str(legacy.get("application_remarks", "")),
                "short_description": str(legacy.get("short_description", "")),
                "detailed_description": str(legacy.get("detailed_description", "")),
                "artifact_stage": str(legacy.get("artifact_stage", "")),
                "source_run_manifest": str(legacy.get("source_run_manifest", "")),
                "quality_report_path": str(legacy.get("quality_report_path", "")),
                "file_sha256": str(legacy.get("file_sha256") or "metadata-unavailable"),
                "file_size_bytes": str(legacy.get("file_size_bytes") or "unknown"),
            }
        )
    return gui_specs

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
