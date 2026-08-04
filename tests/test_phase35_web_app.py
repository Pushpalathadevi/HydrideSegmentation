"""Phase 35 tests for the intranet browser application."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from hydride_segmentation.web import create_app
from hydride_segmentation.web.config import WebServerConfig, load_web_config
from hydride_segmentation.web.models import ModelCatalog
from hydride_segmentation.web.segmentation import (
    CONVENTIONAL_CONTROLS,
    QUANTIFICATION_CONTROLS,
    JobLimiter,
    SegmentationRequestError,
    build_conventional_params,
    build_quantification_config,
    group_metrics,
    prepare_image,
    summarize_fn,
    validate_upload_name,
)

WEB_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hydride_segmentation" / "web"


@pytest.fixture(scope="module")
def client():
    """Return a test client with startup preloading disabled for speed."""

    app = create_app(preload=False)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _png_bytes(width: int = 64, height: int = 48, *, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    array[10:30, 10:50] = 240
    buffer = io.BytesIO()
    Image.fromarray(array).convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


# -- configuration -------------------------------------------------------


def test_config_loads_packaged_defaults_without_warnings() -> None:
    config = load_web_config()

    assert config.warnings == ()
    assert config.port > 0
    assert config.max_upload_mb > 0
    assert config.sample_images, "at least one example image must be configured"
    for sample in config.sample_images:
        assert (Path(config.repo_root) / sample.path).exists()


def test_config_falls_back_when_file_is_missing(tmp_path: Path) -> None:
    config = load_web_config(tmp_path / "absent.yml")

    assert config.port == 5005
    assert any("not found" in warning for warning in config.warnings)


def test_config_reads_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MICROSEG_WEB_PORT", "8123")
    monkeypatch.setenv("MICROSEG_WEB_LIMITS__MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("MICROSEG_WEB_SERVER__HOST", "127.0.0.1")

    config = load_web_config()

    assert config.port == 8123
    assert config.max_upload_mb == 7
    assert config.host == "127.0.0.1"
    assert config.max_upload_bytes == 7 * 1024 * 1024


def test_config_drops_example_images_that_do_not_exist(tmp_path: Path) -> None:
    config_file = tmp_path / "web.yml"
    config_file.write_text(
        "demo:\n"
        "  sample_images:\n"
        "    - path: 'data/sample_images/hydride_optical_sample.png'\n"
        "      label: 'Present'\n"
        "    - path: 'data/sample_images/does_not_exist.png'\n"
        "      label: 'Absent'\n",
        encoding="utf-8",
    )

    config = load_web_config(config_file)

    labels = [sample.label for sample in config.sample_images]
    assert "Present" in labels
    assert "Absent" not in labels
    assert any("missing" in warning for warning in config.warnings)


# -- offline guarantees --------------------------------------------------


def test_pages_reference_no_external_resources(client) -> None:
    for url in ("/", "/help"):
        body = client.get(url).get_data(as_text=True)
        external = re.findall(r'(?:src|href)="((?:https?:)?//[^"]+)"', body)
        assert external == [], f"{url} must not load anything from outside this host: {external}"


def test_static_assets_are_bundled_with_the_package() -> None:
    for relative in (
        "static/css/app.css",
        "static/js/app.js",
        "static/img/favicon.svg",
        "templates/base.html",
        "templates/index.html",
        "templates/help.html",
    ):
        assert (WEB_PACKAGE_ROOT / relative).exists(), f"missing bundled asset: {relative}"


def test_stylesheet_and_script_have_no_remote_imports() -> None:
    css = (WEB_PACKAGE_ROOT / "static" / "css" / "app.css").read_text(encoding="utf-8")
    js = (WEB_PACKAGE_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert "http://" not in css and "https://" not in css
    assert "http://" not in js and "https://" not in js
    assert "@import" not in css


def test_static_assets_are_served(client) -> None:
    for url, content_type in (
        ("/static/css/app.css", "text/css"),
        ("/static/js/app.js", "javascript"),
        ("/static/img/favicon.svg", "svg"),
        ("/static/img/conventional_pipeline.svg", "svg"),
        ("/static/img/ml_pipeline.svg", "svg"),
        ("/static/vendor/katex/katex.min.css", "text/css"),
        ("/static/vendor/katex/katex.min.js", "javascript"),
        ("/static/vendor/katex/auto-render.min.js", "javascript"),
    ):
        response = client.get(url)
        assert response.status_code == 200, f"{url} is not served"
        assert content_type in response.headers["Content-Type"]


# -- vendored KaTeX ------------------------------------------------------


def test_vendored_katex_asks_only_for_fonts_that_were_copied() -> None:
    """A partial vendor copy must fail here, not on an air-gapped host.

    The upstream stylesheet references woff, ttf, and woff2 copies of every
    face. Only woff2 is shipped, so the rules were edited; if that edit is ever
    lost or a font file is missed, the help page silently falls back to a
    non-mathematical face on a machine with no route to a CDN.
    """

    katex_dir = WEB_PACKAGE_ROOT / "static" / "vendor" / "katex"
    css = (katex_dir / "katex.min.css").read_text(encoding="utf-8")

    referenced = set(re.findall(r"url\(fonts/([^)]+)\)", css))
    assert referenced, "the vendored stylesheet declares no font faces"

    missing = sorted(name for name in referenced if not (katex_dir / "fonts" / name).exists())
    assert not missing, f"stylesheet references fonts that were not vendored: {missing}"

    other_formats = sorted(name for name in referenced if not name.endswith(".woff2"))
    assert not other_formats, f"only woff2 is vendored, but the CSS still asks for: {other_formats}"


#: URLs that appear in vendored code as identifiers rather than as fetches.
#: These are XML namespaces handed to createElementNS; nothing requests them.
_NAMESPACE_URIS = frozenset(
    {
        "http://www.w3.org/1998/Math/MathML",
        "http://www.w3.org/2000/svg",
        "http://www.w3.org/1999/xhtml",
    }
)


def test_vendored_katex_makes_no_remote_requests() -> None:
    katex_dir = WEB_PACKAGE_ROOT / "static" / "vendor" / "katex"
    for name in ("katex.min.css", "katex.min.js", "auto-render.min.js"):
        text = (katex_dir / name).read_text(encoding="utf-8", errors="ignore")
        urls = set(re.findall(r"https?://[^\"'\s)]+", text))
        fetched = sorted(urls - _NAMESPACE_URIS)
        assert not fetched, f"{name} would reach outside this host for: {fetched}"


def test_vendored_fonts_survived_checkout_intact() -> None:
    """Every vendored font must still be a parseable woff2 file.

    Contributors run with ``core.autocrlf=true``. Without the ``binary`` rules in
    ``.gitattributes`` git decides per file, by heuristic, whether to rewrite
    line endings on checkout, and a font rewritten as text stops parsing. The
    browser then substitutes a face and reports nothing, so the help page renders
    its mathematics in the wrong font with no error to notice. Checking the
    format signature catches that on whatever machine ran the checkout.
    """

    fonts = sorted((WEB_PACKAGE_ROOT / "static" / "vendor" / "katex" / "fonts").glob("*.woff2"))
    assert len(fonts) >= 20, f"expected the full KaTeX font set, found {len(fonts)}"

    for font in fonts:
        payload = font.read_bytes()
        # woff2 files begin with the signature 'wOF2'.
        assert payload[:4] == b"wOF2", (
            f"{font.name} is not a valid woff2 file; a checkout probably rewrote its bytes"
        )
        assert len(payload) > 1024, f"{font.name} is implausibly small at {len(payload)} bytes"


def test_repository_declares_binary_formats_for_checkout() -> None:
    """The rules that keep the fonts intact must themselves be committed."""

    attributes = WEB_PACKAGE_ROOT.parents[1] / ".gitattributes"
    assert attributes.exists(), ".gitattributes is required so fonts are not line-ending converted"

    text = attributes.read_text(encoding="utf-8")
    for pattern in ("*.woff2", "*.png", "*.pt"):
        assert re.search(rf"{re.escape(pattern)}\s+binary", text), (
            f".gitattributes does not mark {pattern} as binary"
        )


def test_vendored_katex_ships_its_licence() -> None:
    licence = WEB_PACKAGE_ROOT / "static" / "vendor" / "katex" / "LICENSE"
    assert licence.exists(), "vendored third-party code must ship its licence"
    assert "MIT" in licence.read_text(encoding="utf-8")


def test_packaging_includes_the_vendored_assets() -> None:
    """Both packaging configs must ship the vendor tree.

    ``pyproject.toml`` is the one the build backend reads, so a pattern present
    only in ``setup.py`` silently produces a wheel with no KaTeX fonts, and the
    help page then falls back to a non-mathematical face on exactly the
    air-gapped host this was vendored for. Both files are checked because both
    exist and they are easy to let drift apart.
    """

    repo_root = WEB_PACKAGE_ROOT.parents[1]
    configs = {
        "pyproject.toml": (repo_root / "pyproject.toml").read_text(encoding="utf-8"),
        "setup.py": (repo_root / "setup.py").read_text(encoding="utf-8"),
    }
    for name, text in configs.items():
        for pattern in (
            "static/vendor/katex/*.css",
            "static/vendor/katex/*.js",
            "static/vendor/katex/fonts/*.woff2",
        ):
            assert pattern in text, f"{name} package data is missing {pattern}"


def test_help_page_loads_katex_locally_and_renders_math() -> None:
    body = (WEB_PACKAGE_ROOT / "templates" / "help.html").read_text(encoding="utf-8")

    assert "vendor/katex/katex.min.css" in body
    assert "vendor/katex/katex.min.js" in body
    assert "vendor/katex/auto-render.min.js" in body
    assert "renderMathInElement" in body
    # Display equations are what the formulae section is for.
    assert body.count("\\[") >= 8, "the formulae section should carry display equations"


# -- pages ---------------------------------------------------------------


def test_index_page_renders_models_samples_and_help(client) -> None:
    body = client.get("/").get_data(as_text=True)

    assert "Segment a micrograph" in body
    assert "Hydride Conventional" in body
    assert "Optical micrograph" in body
    for control in CONVENTIONAL_CONTROLS:
        assert control.label in body
        assert control.help_text[:40] in body


def test_index_page_embeds_valid_bootstrap_json(client) -> None:
    body = client.get("/").get_data(as_text=True)
    match = re.search(r'<script id="bootstrap-data" type="application/json">(.*?)</script>', body, re.S)

    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["models"]
    assert payload["defaultModelId"]
    assert len(payload["controls"]) == len(CONVENTIONAL_CONTROLS)


def test_help_page_documents_every_conventional_control(client) -> None:
    body = client.get("/help").get_data(as_text=True)

    assert "How to use this tool" in body
    for control in CONVENTIONAL_CONTROLS:
        assert control.label in body
    assert "Troubleshooting" in body


# -- api -----------------------------------------------------------------


def test_health_endpoint_is_immediate(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_status_endpoint_reports_models_and_limits(client) -> None:
    payload = client.get("/api/status").get_json()

    assert payload["ok"] is True
    assert payload["conventional_available"] is True
    assert payload["max_upload_mb"] > 0
    assert isinstance(payload["models"], list) and payload["models"]


def test_models_endpoint_offers_a_runnable_default(client) -> None:
    payload = client.get("/api/models").get_json()

    default_id = payload["default_model_id"]
    match = next(model for model in payload["models"] if model["model_id"] == default_id)
    assert match["available"] is True


def test_sample_endpoints_serve_a_usable_image(client) -> None:
    listed = client.get("/api/samples").get_json()
    assert listed["samples"]

    sample_id = listed["samples"][0]["id"]
    response = client.get(f"/api/samples/{sample_id}")
    assert response.status_code == 200
    with Image.open(io.BytesIO(response.data)) as handle:
        assert handle.width > 0 and handle.height > 0

    assert client.get("/api/samples/not_a_real_sample").status_code == 404


# -- segmentation --------------------------------------------------------


def test_conventional_segmentation_returns_images_and_metrics(client) -> None:
    response = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "image": (io.BytesIO(_png_bytes()), "sample.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["model_id"] == "hydride_conventional"
    assert payload["used_example_image"] is False
    for key in ("input_png_b64", "mask_png_b64", "overlay_png_b64"):
        assert payload["images"][key]
        assert base64.b64decode(payload["images"][key])[:4] == b"\x89PNG"
    assert "area_fraction" in payload["metrics"]
    assert payload["timing"]["total_seconds"] >= 0.0


def test_segmentation_runs_on_the_example_image_without_an_upload(client) -> None:
    sample_id = client.get("/api/samples").get_json()["samples"][0]["id"]

    payload = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "sample_id": sample_id},
    ).get_json()

    assert payload["ok"] is True
    assert payload["used_example_image"] is True
    assert payload["images"]["overlay_png_b64"]


def test_conventional_parameters_change_the_result(client) -> None:
    image = _png_bytes(128, 128)

    def run(offset: int) -> float:
        payload = client.post(
            "/api/segment",
            data={
                "model_id": "hydride_conventional",
                "adaptive_offset": str(offset),
                "image": (io.BytesIO(image), "sample.png"),
            },
            content_type="multipart/form-data",
        ).get_json()
        assert payload["ok"] is True
        return float(payload["metrics"]["area_fraction"])

    assert run(2) != run(40)


def test_trained_model_route_is_served_when_a_checkpoint_is_installed(client) -> None:
    models = client.get("/api/models").get_json()["models"]
    trained = [m for m in models if m["available"] and not m["is_conventional"]]
    if not trained:
        pytest.skip("no trained checkpoint is installed in this environment")

    payload = client.post(
        "/api/segment",
        data={"model_id": trained[0]["model_id"], "image": (io.BytesIO(_png_bytes()), "sample.png")},
        content_type="multipart/form-data",
    ).get_json()

    assert payload["ok"] is True
    assert payload["images"]["overlay_png_b64"]
    assert payload["manifest"]["image"]["width"] > 0


def test_large_images_are_downscaled_and_reported(client) -> None:
    app = create_app(preload=False)
    app.extensions["microseg_web"]["config"].max_long_side_px = 64
    with app.test_client() as scoped:
        payload = scoped.post(
            "/api/segment",
            data={
                "model_id": "hydride_conventional",
                "image": (io.BytesIO(_png_bytes(200, 150)), "big.png"),
            },
            content_type="multipart/form-data",
        ).get_json()

    assert payload["ok"] is True
    meta = payload["manifest"]["image"]
    assert meta["downscaled"] is True
    assert max(meta["width"], meta["height"]) == 64
    assert meta["original_width"] == 200


# -- request validation --------------------------------------------------


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        ({"model_id": "hydride_conventional"}, 400, "NO_IMAGE"),
        ({"model_id": "no_such_model", "sample_id": "hydride_optical_sample"}, 400, "UNKNOWN_MODEL"),
    ],
)
def test_invalid_requests_return_actionable_errors(client, data, status, code) -> None:
    response = client.post("/api/segment", data=data)

    assert response.status_code == status
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["code"] == code
    assert payload["error"]["detail"].strip()


def test_unsupported_file_type_is_rejected_by_name(client) -> None:
    response = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "image": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "not supported" in response.get_json()["error"]["detail"]


def test_corrupt_image_with_valid_extension_is_rejected(client) -> None:
    response = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "image": (io.BytesIO(b"not a png"), "fake.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "could not be read" in response.get_json()["error"]["detail"]


def test_oversized_upload_is_rejected_before_processing() -> None:
    app = create_app(config=WebServerConfig(max_upload_mb=1, repo_root=str(Path.cwd())), preload=False)
    with app.test_client() as scoped:
        response = scoped.post(
            "/api/segment",
            data={
                "model_id": "hydride_conventional",
                "image": (io.BytesIO(b"0" * (2 * 1024 * 1024)), "big.png"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "FILE_TOO_LARGE"


def test_validate_upload_name_accepts_and_rejects_extensions() -> None:
    assert validate_upload_name("scan.TIF") == "tif"
    with pytest.raises(SegmentationRequestError):
        validate_upload_name("scan.exe")
    with pytest.raises(SegmentationRequestError):
        validate_upload_name("noextension")


def test_build_conventional_params_enforces_documented_constraints() -> None:
    params = build_conventional_params({})
    assert params["adaptive"]["block_size"] % 2 == 1

    with pytest.raises(SegmentationRequestError, match="odd"):
        build_conventional_params({"adaptive_block_size": "12"})
    with pytest.raises(SegmentationRequestError, match="greater than zero"):
        build_conventional_params({"clahe_clip_limit": "0"})
    with pytest.raises(SegmentationRequestError, match="must be a number"):
        build_conventional_params({"morph_kernel": "wide"})


def test_prepare_image_converts_grayscale_and_preserves_size() -> None:
    prepared = prepare_image(_png_bytes(80, 60, mode="L"))

    assert prepared.array.ndim == 3 and prepared.array.shape[2] == 3
    assert (prepared.width, prepared.height) == (80, 60)
    assert prepared.downscaled is False


def test_prepare_image_rejects_empty_payload() -> None:
    with pytest.raises(SegmentationRequestError):
        prepare_image(b"")


# -- catalog and concurrency --------------------------------------------


# -- Fn quantification ---------------------------------------------------


def _run(client, **extra):
    data = {"model_id": "hydride_conventional", "sample_id": "hydride_optical_sample"}
    data.update(extra)
    payload = client.post("/api/segment", data=data).get_json()
    assert payload["ok"] is True, payload
    return payload


def test_segmentation_reports_fn_for_the_conventional_route(client) -> None:
    payload = _run(client)

    fn = payload["fn"]
    assert fn["available"] is True
    assert 0.0 <= fn["fn_count"] <= 1.0
    assert 0.0 <= fn["fn_length_weighted"] <= 1.0
    assert fn["count_numerator"] <= fn["count_denominator"]
    assert fn["length_numerator_px"] <= fn["length_denominator_px"] + 1e-6
    assert fn["angle_threshold_deg"] == 45.0

    for key in ("fn_count", "fn_length_weighted", "fn_angle_threshold_deg", "fn_count_numerator"):
        assert key in payload["metrics"]


def test_segmentation_reports_fn_for_the_trained_route(client) -> None:
    models = client.get("/api/models").get_json()["models"]
    trained = [m for m in models if m["available"] and not m["is_conventional"]]
    if not trained:
        pytest.skip("no trained checkpoint is installed in this environment")

    payload = _run(client, model_id=trained[0]["model_id"])

    assert payload["fn"]["available"] is True
    assert "fn_length_weighted" in payload["metrics"]


def test_fn_threshold_changes_the_reported_fraction(client) -> None:
    strict = _run(client, fn_angle_threshold_deg="80")["fn"]
    lenient = _run(client, fn_angle_threshold_deg="10")["fn"]

    assert strict["angle_threshold_deg"] == 80.0
    assert lenient["angle_threshold_deg"] == 10.0
    assert lenient["count_numerator"] >= strict["count_numerator"]
    assert lenient["fn_count"] >= strict["fn_count"]


def test_minimum_feature_size_excludes_small_features_from_fn(client) -> None:
    unfiltered = _run(client, min_feature_pixels="1")["fn"]
    filtered = _run(client, min_feature_pixels="200")["fn"]

    assert filtered["count_denominator"] <= unfiltered["count_denominator"]
    assert filtered["excluded_small_features"] >= unfiltered["excluded_small_features"]


def test_quantification_settings_are_recorded_in_the_manifest(client) -> None:
    payload = _run(client, fn_angle_threshold_deg="37.5", min_feature_pixels="12")

    quantification = payload["manifest"]["quantification"]
    assert quantification["fn_angle_threshold_deg"] == 37.5
    assert quantification["min_feature_pixels"] == 12
    assert quantification["include_analysis"] is True


def test_fn_classification_views_are_opt_in(client) -> None:
    without = _run(client)
    assert "fn_classification_png_b64" not in without["images"]

    with_views = _run(client, include_fn_classification="true")
    assert base64.b64decode(with_views["images"]["fn_classification_png_b64"])[:4] == b"\x89PNG"
    assert base64.b64decode(with_views["images"]["fn_angle_threshold_png_b64"])[:4] == b"\x89PNG"
    assert with_views["manifest"]["quantification"]["include_fn_classification"] is True


def test_fn_classification_survives_masks_with_many_features(client) -> None:
    """A dense mask previously exhausted memory in the Fn classification renderer."""

    payload = _run(client, min_feature_pixels="1", include_fn_classification="true")

    assert payload["fn"]["count_denominator"] > 100
    assert payload["images"]["fn_classification_png_b64"]


def test_metrics_are_grouped_with_fn_first(client) -> None:
    groups = _run(client)["metric_groups"]

    assert groups[0]["key"] == "fn"
    assert groups[0]["title"].startswith("Radial hydride fraction")
    titles = [group["key"] for group in groups]
    for expected in ("coverage", "orientation", "size"):
        assert expected in titles


def test_grouping_keeps_every_metric() -> None:
    metrics = {"fn_count": 0.1, "hydride_count": 5, "orientation_mean_deg": 12.0, "unexpected_metric": 1}

    grouped = group_metrics(metrics)

    flattened = {entry["key"] for group in grouped for entry in group["metrics"]}
    assert flattened == set(metrics)
    assert any(group["key"] == "other" for group in grouped)


def test_summarize_fn_reports_unavailable_without_fn_metrics() -> None:
    assert summarize_fn({"hydride_count": 3}) == {"available": False}


def test_quantification_config_defaults_and_validation() -> None:
    config = build_quantification_config({})
    assert config.fn_angle_threshold_deg == 45.0
    assert config.min_feature_pixels == 1
    assert config.include_fn_metrics is True

    tuned = build_quantification_config({"fn_angle_threshold_deg": "30", "min_feature_pixels": "25"})
    assert tuned.fn_angle_threshold_deg == 30.0
    assert tuned.min_feature_pixels == 25

    with pytest.raises(SegmentationRequestError, match="between 0 and 90"):
        build_quantification_config({"fn_angle_threshold_deg": "120"})
    with pytest.raises(SegmentationRequestError, match="at least 1 pixel"):
        build_quantification_config({"min_feature_pixels": "0"})
    with pytest.raises(SegmentationRequestError, match="must be a number"):
        build_quantification_config({"fn_angle_threshold_deg": "steep"})


def test_invalid_fn_threshold_is_rejected_by_the_endpoint(client) -> None:
    response = client.post(
        "/api/segment",
        data={
            "model_id": "hydride_conventional",
            "sample_id": "hydride_optical_sample",
            "fn_angle_threshold_deg": "200",
        },
    )

    assert response.status_code == 400
    assert "between 0 and 90" in response.get_json()["error"]["detail"]


def test_workspace_exposes_quantification_controls_and_fn_panel(client) -> None:
    body = client.get("/").get_data(as_text=True)

    for control in QUANTIFICATION_CONTROLS:
        assert control.label in body
        assert f'id="ctl-{control.key}"' in body
    assert 'id="fn-panel"' in body
    assert "Radial hydride fraction (Fn)" in body

    match = re.search(r'<script id="bootstrap-data" type="application/json">(.*?)</script>', body, re.S)
    payload = json.loads(match.group(1))
    assert len(payload["quantificationControls"]) == len(QUANTIFICATION_CONTROLS)


def test_help_page_explains_fn_and_its_settings(client) -> None:
    body = client.get("/help").get_data(as_text=True)

    assert 'id="fn"' in body
    assert "Length-weighted Fn" in body
    assert "Count-based Fn" in body
    for control in QUANTIFICATION_CONTROLS:
        assert control.label in body


def test_help_page_documents_both_processing_pipelines(client) -> None:
    body = client.get("/help").get_data(as_text=True)

    assert 'id="pipeline"' in body
    for diagram in ("img/conventional_pipeline.svg", "img/ml_pipeline.svg"):
        assert diagram in body, f"the help page does not show {diagram}"


@pytest.mark.parametrize(
    ("diagram", "stages"),
    [
        (
            "conventional_pipeline.svg",
            ("Normalise", "CLAHE", "Adaptive threshold", "Morphological closing", "Binary mask"),
        ),
        (
            "ml_pipeline.svg",
            ("RGB tensor", "Encoder", "Decoder", "convolution", "Binary mask"),
        ),
    ],
)
def test_pipeline_flow_charts_name_the_stages_the_code_runs(diagram, stages) -> None:
    """Each flow chart must describe the pipeline it claims to.

    The diagrams are hand-drawn SVG, so nothing but a test keeps them honest
    when the implementation moves on.
    """

    svg = (WEB_PACKAGE_ROOT / "static" / "img" / diagram).read_text(encoding="utf-8")

    for stage in stages:
        assert stage in svg, f"{diagram} does not mention {stage!r}"


@pytest.mark.parametrize("diagram", ["conventional_pipeline.svg", "ml_pipeline.svg"])
def test_pipeline_flow_charts_are_valid_and_self_contained(diagram) -> None:
    svg = (WEB_PACKAGE_ROOT / "static" / "img" / diagram).read_text(encoding="utf-8")

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")

    # A screen reader needs these; the chart carries real information.
    assert "<title" in svg and "<desc" in svg, f"{diagram} lacks a title and description"

    # Gradients, filters, and markers must resolve, or shapes render unfilled.
    declared = set(re.findall(r'\sid="([^"]+)"', svg))
    referenced = set(re.findall(r"url\(#([^)]+)\)", svg))
    assert not (referenced - declared), f"{diagram} references undefined ids: {referenced - declared}"

    remote = svg.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "http://" not in remote and "https://" not in remote, (
        f"{diagram} must not reference anything remote"
    )


def test_help_page_only_documents_metrics_the_server_actually_reports(client) -> None:
    """Guard the formulae section against drifting away from the real payload.

    Every metric key cited in the help page is checked against the keys a real
    run emits, so renaming a metric without updating the documentation fails
    here rather than silently misleading a reader.
    """

    body = client.get("/help").get_data(as_text=True)
    documented = set(re.findall(r'<span class="metric-key">([a-z0-9_]+)</span>', body))
    assert documented, "the formulae section must cite the metric keys it defines"

    produced = set(_run(client)["metrics"])

    unknown = documented - produced
    assert not unknown, f"help page documents metrics the server never reports: {sorted(unknown)}"


def test_help_page_defines_the_quantification_formulae(client) -> None:
    body = client.get("/help").get_data(as_text=True)

    assert 'id="formulae"' in body
    for heading in (
        "Feature orientation",
        "Feature length",
        "Count-based Fn",
        "Length-weighted Fn",
        "Area fraction",
        "Equivalent circular diameter",
        "Alignment index",
        "Orientation entropy",
    ):
        assert heading.upper() in body.upper(), f"formulae section is missing {heading!r}"


def test_timing_separates_inference_from_analysis(client) -> None:
    timing = _run(client)["timing"]

    assert timing["inference_seconds"] >= 0.0
    assert timing["analysis_seconds"] >= 0.0
    assert timing["total_seconds"] >= timing["inference_seconds"]


def test_catalog_default_falls_back_when_configured_model_is_unavailable() -> None:
    catalog = ModelCatalog()

    assert catalog.default_model_id("definitely_not_installed") in {
        option.model_id for option in catalog.options() if option.available
    }


def test_catalog_reports_conventional_as_needing_no_warm_load() -> None:
    catalog = ModelCatalog()

    result = catalog.warm_model("hydride_conventional")
    assert result["state"] == "ready"
    assert catalog.status()["conventional_available"] is True


def test_catalog_warm_load_reports_unknown_models() -> None:
    catalog = ModelCatalog()

    assert catalog.warm_model("not_a_model")["state"] == "error"


def test_job_limiter_bounds_concurrency() -> None:
    limiter = JobLimiter(max_concurrent_jobs=1, timeout_seconds=0.1)

    assert limiter.acquire() is True
    assert limiter.active_jobs == 1
    assert limiter.acquire() is False, "a second job must not start while the slot is taken"
    limiter.release()
    assert limiter.active_jobs == 0
    assert limiter.acquire() is True
    limiter.release()


def test_server_reports_busy_when_all_job_slots_are_taken() -> None:
    app = create_app(preload=False)
    limiter = app.extensions["microseg_web"]["limiter"]
    app.extensions["microseg_web"]["limiter"] = JobLimiter(max_concurrent_jobs=1, timeout_seconds=0.1)
    app.extensions["microseg_web"]["limiter"].acquire()
    try:
        with app.test_client() as scoped:
            response = scoped.post(
                "/api/segment",
                data={"model_id": "hydride_conventional", "sample_id": "hydride_optical_sample"},
            )
        assert response.status_code == 503
        assert response.get_json()["error"]["code"] == "SERVER_BUSY"
    finally:
        app.extensions["microseg_web"]["limiter"] = limiter
