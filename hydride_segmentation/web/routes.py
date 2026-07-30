"""HTTP routes for the intranet web application.

Two page routes render the browser interface, and a small JSON API backs it.
Every response is generated locally; no request ever leaves the host.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)

from .config import WebServerConfig
from .jobs import WebJobManager
from .models import CONVENTIONAL_MODEL_ID, ModelCatalog
from .segmentation import (
    ALLOWED_EXTENSIONS,
    CONVENTIONAL_CONTROLS,
    QUANTIFICATION_CONTROLS,
    SegmentationRequestError,
    build_conventional_params,
    build_quantification_config,
    prepare_image,
    run_web_segmentation,
    validate_upload_name,
)

_LOGGER = logging.getLogger(__name__)


def _config() -> WebServerConfig:
    return current_app.extensions["microseg_web"]["config"]


def _catalog() -> ModelCatalog:
    return current_app.extensions["microseg_web"]["catalog"]


def _limiter():
    return current_app.extensions["microseg_web"]["limiter"]

def _jobs() -> WebJobManager:
    return current_app.extensions["microseg_web"]["jobs"]


def _error(code: str, detail: str, status: int, **extra: Any):
    payload = {"ok": False, "error": {"code": code, "detail": detail}}
    payload.update(extra)
    return jsonify(payload), status


def _resolve_sample(sample_id: str) -> Path | None:
    config = _config()
    root = Path(config.repo_root)
    for sample in config.sample_images:
        if sample.sample_id == str(sample_id).strip():
            candidate = Path(sample.path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                return candidate
    return None


def create_web_blueprint() -> Blueprint:
    """Create the blueprint serving the browser UI and its JSON API."""

    bp = Blueprint(
        "microseg_web",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    # -- pages -----------------------------------------------------------

    @bp.get("/")
    def index():
        config = _config()
        catalog = _catalog()
        options = catalog.options()
        return render_template(
            "index.html",
            config=config,
            models=[option.to_dict() for option in options],
            default_model_id=catalog.default_model_id(config.default_model_id),
            controls=[control.to_dict() for control in CONVENTIONAL_CONTROLS],
            quantification_controls=[control.to_dict() for control in QUANTIFICATION_CONTROLS],
            samples=[{"id": item.sample_id, "label": item.label} for item in config.sample_images],
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            active_page="workspace",
        )

    @bp.get("/help")
    def help_page():
        config = _config()
        return render_template(
            "help.html",
            config=config,
            controls=[control.to_dict() for control in CONVENTIONAL_CONTROLS],
            quantification_controls=[control.to_dict() for control in QUANTIFICATION_CONTROLS],
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            active_page="help",
        )

    # -- api -------------------------------------------------------------

    @bp.get("/health")
    def health():
        """Liveness probe that never blocks on model loading."""

        return jsonify({"ok": True, "status": "ok"})

    @bp.get("/api/status")
    def status():
        """Report model readiness and current server load."""

        config = _config()
        catalog = _catalog()
        jobs = _jobs()
        payload = catalog.status()
        payload.update(
            {
                "ok": True,
                "site_name": config.site_name,
                "active_jobs": jobs.active_jobs,
                "queued_jobs": jobs.queued_jobs,
                "max_concurrent_jobs": config.max_concurrent_jobs,
                "max_upload_mb": config.max_upload_mb,
                "max_long_side_px": config.max_long_side_px,
                "include_analysis": config.include_analysis,
            }
        )
        return jsonify(payload)

    @bp.get("/api/models")
    def models():
        """List selectable models with availability and warm state."""

        config = _config()
        catalog = _catalog()
        return jsonify(
            {
                "ok": True,
                "models": [option.to_dict() for option in catalog.options()],
                "default_model_id": catalog.default_model_id(config.default_model_id),
            }
        )

    @bp.get("/api/samples")
    def samples():
        """List the example images offered for testing."""

        config = _config()
        return jsonify(
            {
                "ok": True,
                "samples": [{"id": item.sample_id, "label": item.label} for item in config.sample_images],
            }
        )

    @bp.get("/api/samples/<sample_id>")
    def sample_image(sample_id: str):
        """Serve one example image so users can test without their own data."""

        path = _resolve_sample(sample_id)
        if path is None:
            return _error("NOT_FOUND", f"No example image named {sample_id!r} is configured.", 404)
        return send_file(path, mimetype="image/png", download_name=path.name)

    @bp.post("/api/segment")
    def segment():
        """Run one segmentation job for an uploaded or example image."""

        config = _config()
        catalog = _catalog()
        limiter = _limiter()

        model_id = str(request.form.get("model_id", "")).strip() or catalog.default_model_id(
            config.default_model_id
        )
        option = catalog.get(model_id)
        if option is None:
            return _error("UNKNOWN_MODEL", f"Model {model_id!r} is not registered on this server.", 400)
        if not option.available:
            return _error(
                "MODEL_UNAVAILABLE",
                option.availability_message
                or "This model is not usable on this server because its checkpoint is missing.",
                409,
            )

        sample_id = str(request.form.get("sample_id", "")).strip()
        upload = request.files.get("image")

        try:
            if sample_id:
                sample_path = _resolve_sample(sample_id)
                if sample_path is None:
                    return _error("NOT_FOUND", f"No example image named {sample_id!r} is configured.", 404)
                data = sample_path.read_bytes()
                source_name = sample_path.name
            else:
                if upload is None or not str(upload.filename or "").strip():
                    return _error(
                        "NO_IMAGE",
                        "Choose an image to upload, or use one of the example images.",
                        400,
                    )
                validate_upload_name(upload.filename)
                data = upload.read()
                if len(data) > config.max_upload_bytes:
                    return _error(
                        "FILE_TOO_LARGE",
                        f"The image is larger than the {config.max_upload_mb} MB limit for this server.",
                        413,
                    )
                source_name = str(upload.filename)

            prepared = prepare_image(
                data,
                max_long_side_px=config.max_long_side_px,
                max_image_pixels=config.max_image_pixels,
                expected_extension=validate_upload_name(source_name),
            )

            form = request.form.to_dict()
            if option.is_conventional:
                params: dict[str, Any] = build_conventional_params(form)
            else:
                params = {
                    "enable_gpu": config.enable_gpu,
                    "device_policy": config.device_policy,
                }
            quantification = build_quantification_config(form)
        except SegmentationRequestError as exc:
            return _error("VALIDATION", str(exc), 400)

        def _flag(name: str, default: bool) -> bool:
            if name not in request.form:
                return default
            return str(request.form.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}

        include_analysis = _flag("include_analysis", config.include_analysis)
        include_fn_classification = include_analysis and _flag("include_fn_classification", False)

        if not limiter.acquire():
            return _error(
                "SERVER_BUSY",
                "The server is handling its maximum number of segmentation jobs. Please try again shortly.",
                503,
            )
        try:
            payload = run_web_segmentation(
                prepared,
                model_id=option.model_id,
                params=params,
                include_analysis=include_analysis,
                quantification=quantification,
                include_fn_classification=include_fn_classification,
                source_name=source_name,
            )
        except MemoryError:
            return _error(
                "OUT_OF_MEMORY",
                "The server ran out of memory for this image. Try a smaller image.",
                507,
            )
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            _LOGGER.exception("Segmentation failed for model %s", option.model_id)
            return _error("SEGMENTATION_FAILED", f"Segmentation failed: {exc}", 500)
        finally:
            limiter.release()

        payload.update(
            {
                "ok": True,
                "model_id": option.model_id,
                "model_display_name": option.display_name,
                "source_name": source_name,
                "used_example_image": bool(sample_id),
            }
        )
        fn_summary = payload.get("fn", {})
        _LOGGER.info(
            "Segmentation served: model=%s source=%s size=%dx%d downscaled=%s "
            "fn_threshold=%.1f fn_count=%s seconds=%.2f",
            option.model_id,
            source_name,
            prepared.width,
            prepared.height,
            prepared.downscaled,
            float(quantification.fn_angle_threshold_deg),
            f"{fn_summary.get('fn_count'):.4f}" if fn_summary.get("available") else "n/a",
            float(payload.get("timing", {}).get("total_seconds", 0.0)),
        )
        return jsonify(payload)

    @bp.post("/api/jobs")
    def create_job():
        """Validate an image synchronously, then process it in memory in the background."""

        config = _config()
        catalog = _catalog()
        model_id = str(request.form.get("model_id", "")).strip() or catalog.default_model_id(
            config.default_model_id
        )
        option = catalog.get(model_id)
        if option is None:
            return _error("UNKNOWN_MODEL", f"Model {model_id!r} is not registered on this server.", 400)
        if not option.available:
            return _error(
                "MODEL_UNAVAILABLE",
                option.availability_message or "This model is not usable on this server.",
                409,
            )

        sample_id = str(request.form.get("sample_id", "")).strip()
        upload = request.files.get("image")
        try:
            if sample_id:
                sample_path = _resolve_sample(sample_id)
                if sample_path is None:
                    return _error("NOT_FOUND", f"No example image named {sample_id!r} is configured.", 404)
                data = sample_path.read_bytes()
                source_name = sample_path.name
            else:
                if upload is None or not str(upload.filename or "").strip():
                    return _error("NO_IMAGE", "Choose an image or use an example image.", 400)
                source_name = str(upload.filename)
                validate_upload_name(source_name)
                data = upload.read(config.max_upload_bytes + 1)
                if len(data) > config.max_upload_bytes:
                    return _error(
                        "FILE_TOO_LARGE",
                        f"The image is larger than the {config.max_upload_mb} MB limit for this server.",
                        413,
                    )
            prepared = prepare_image(
                data,
                max_long_side_px=config.max_long_side_px,
                max_image_pixels=config.max_image_pixels,
                expected_extension=validate_upload_name(source_name),
            )
            form = request.form.to_dict()
            params = (
                build_conventional_params(form)
                if option.is_conventional
                else {"enable_gpu": config.enable_gpu, "device_policy": config.device_policy}
            )
            quantification = build_quantification_config(form)
        except SegmentationRequestError as exc:
            return _error("VALIDATION", str(exc), 400)

        def flag(name: str, default: bool) -> bool:
            if name not in request.form:
                return default
            return str(request.form.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}

        include_analysis = flag("include_analysis", config.include_analysis)
        include_fn_classification = include_analysis and flag("include_fn_classification", False)
        used_example = bool(sample_id)

        def runner(progress_hook):
            payload = run_web_segmentation(
                prepared,
                model_id=option.model_id,
                params=params,
                include_analysis=include_analysis,
                quantification=quantification,
                include_fn_classification=include_fn_classification,
                source_name=source_name,
                progress_hook=progress_hook,
            )
            payload.update(
                {
                    "ok": True,
                    "model_id": option.model_id,
                    "model_display_name": option.display_name,
                    "source_name": source_name,
                    "used_example_image": used_example,
                }
            )
            return payload

        job = _jobs().submit(runner)
        if job is None:
            return _error("SERVER_BUSY", "The in-memory work queue is full. Try again shortly.", 503)
        return (
            jsonify(
                {
                    "ok": True,
                    "job_id": job.job_id,
                    "state": job.state,
                    "status_url": f"/api/jobs/{job.job_id}",
                    "privacy": {
                        "input_transport": "memory",
                        "source_persisted": False,
                        "result_persisted": False,
                    },
                }
            ),
            202,
        )

    @bp.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        """Return new progress events and the terminal result, when ready."""

        job = _jobs().get(job_id)
        if job is None:
            return _error("NOT_FOUND", "This job does not exist or its in-memory report expired.", 404)
        try:
            after = max(0, int(request.args.get("after", "0")))
        except ValueError:
            return _error("VALIDATION", "The 'after' sequence must be an integer.", 400)
        return jsonify(job.to_dict(after_sequence=after, include_result=True))

    @bp.post("/api/warm")
    def warm():
        """Load a model into memory on demand, for operators testing a deployment."""

        catalog = _catalog()
        model_id = str(request.form.get("model_id", "")).strip()
        if not model_id:
            return _error("VALIDATION", "model_id is required.", 400)
        result = catalog.warm_model(model_id)
        status_code = 200 if result.get("state") == "ready" else 409
        return jsonify({"ok": result.get("state") == "ready", "model_id": model_id, **result}), status_code

    @bp.errorhandler(413)
    def too_large(_exc):
        config = _config()
        return _error(
            "FILE_TOO_LARGE",
            f"The image is larger than the {config.max_upload_mb} MB limit for this server.",
            413,
        )

    return bp
