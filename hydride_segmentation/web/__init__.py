"""Browser-based intranet application for microstructural segmentation.

The application factory builds a self-contained Flask app: templates and static
assets are served from this package, model metadata comes from the same registry
the desktop GUI and CLI use, and no request depends on an external network.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from hydride_segmentation.version import __version__

from .config import WebServerConfig, load_web_config
from .downloads import DownloadCatalog
from .library import ImageLibrary
from .models import ModelCatalog
from .routes import create_web_blueprint
from .jobs import WebJobManager
from .segmentation import JobLimiter

__all__ = [
    "ImageLibrary",
    "JobLimiter",
    "ModelCatalog",
    "WebServerConfig",
    "create_app",
    "load_web_config",
]

_LOGGER = logging.getLogger(__name__)


def create_app(
    config_path: str | Path | None = None,
    *,
    config: WebServerConfig | None = None,
    preload: bool | None = None,
) -> Flask:
    """Create the intranet segmentation web application.

    Parameters
    ----------
    config_path:
        Optional YAML configuration path. Defaults to the packaged configuration.
    config:
        Optional pre-resolved configuration, primarily for tests.
    preload:
        Overrides the configured startup warm-load behaviour. Tests pass
        ``False`` to keep application creation instant.

    Returns
    -------
    flask.Flask
        Configured application, ready for a WSGI server.
    """

    resolved = config if config is not None else load_web_config(config_path)

    app = Flask(__name__)
    # Allow modest multipart framing overhead; the route enforces the exact
    # image-byte ceiling before decoding.
    app.config["MAX_CONTENT_LENGTH"] = resolved.max_upload_bytes + (1024 * 1024)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MICROSEG_WEB_CONFIG"] = resolved

    for warning in resolved.warnings:
        _LOGGER.warning("Web configuration: %s", warning)

    should_preload = resolved.preload_on_startup if preload is None else bool(preload)
    catalog = ModelCatalog(
        enable_gpu=resolved.enable_gpu,
        device_policy=resolved.device_policy,
        preload_model_ids=resolved.preload_model_ids,
        preload_enabled=should_preload,
    )
    limiter = JobLimiter(
        max_concurrent_jobs=resolved.max_concurrent_jobs,
        timeout_seconds=float(resolved.request_timeout_seconds),
    )
    jobs = WebJobManager(
        max_concurrent_jobs=resolved.max_concurrent_jobs,
        max_retained_jobs=resolved.max_retained_jobs,
        retention_seconds=resolved.job_retention_seconds,
    )
    library = ImageLibrary(resolved.library_dir, max_images=resolved.library_max_images)
    downloads = DownloadCatalog(resolved.repo_root)

    app.extensions["microseg_web"] = {
        "config": resolved,
        "catalog": catalog,
        "limiter": limiter,
        "jobs": jobs,
        "library": library,
        "downloads": downloads,
    }

    @app.context_processor
    def inject_product_metadata() -> dict[str, str]:
        return {"app_version": __version__}

    app.register_blueprint(create_web_blueprint())

    if should_preload:
        if resolved.preload_in_background:
            _LOGGER.info("Warming trained models in the background")
            catalog.preload_async()
        else:
            _LOGGER.info("Warming trained models before accepting requests")
            catalog.preload()
    else:
        _LOGGER.info("Model preloading is disabled; the first request will load its model")

    # Report the library state once at startup so an operator who forgot to copy
    # the folder onto a new host sees why only the built-in examples are offered.
    library_count = len(library.list_images())
    if library_count:
        _LOGGER.info("Image library: %d image(s) from %s", library_count, resolved.library_dir)
    else:
        _LOGGER.info(
            "Image library is unavailable at %s; falling back to the %d configured example image(s)",
            resolved.library_dir or "(not configured)",
            len(resolved.sample_images),
        )

    _LOGGER.info(
        "Web application ready | config=%s | upload_limit=%dMB | jobs=%d",
        resolved.source_path,
        resolved.max_upload_mb,
        resolved.max_concurrent_jobs,
    )
    return app
