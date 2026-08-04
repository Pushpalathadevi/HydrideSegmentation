"""Tests for the browsable server-side image library.

The library folder is supplied per deployment rather than committed, so these
tests cover both the populated case and every way it can be absent, plus the
path handling on the one endpoint that accepts a client-supplied name.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydride_segmentation.web import create_app
from hydride_segmentation.web.config import load_web_config
from hydride_segmentation.web.library import ImageLibrary, resolve_library_image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, width: int = 64, height: int = 48, *, mode: str = "RGB") -> Path:
    rng = np.random.default_rng(11)
    array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    array[8:24, 8:40] = 235
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert(mode).save(path)
    return path


@pytest.fixture
def library_dir(tmp_path: Path) -> Path:
    """Return a populated library folder holding a mix of files."""

    root = tmp_path / "test_library"
    _write_image(root / "optical_scan.png")
    _write_image(root / "sem_view.tif")
    _write_image(root / "Second_Sample.PNG")
    # Files the library must skip.
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / "README.md").write_text("# not an image", encoding="utf-8")
    # Subfolders are out of scope by design.
    _write_image(root / "nested" / "ignored.png")
    return root


def _client(library_dir: Path | str | None):
    """Return a test client whose library points at ``library_dir``."""

    resolved = replace(load_web_config(), library_dir=str(library_dir or ""))
    app = create_app(config=resolved, preload=False)
    app.config.update(TESTING=True)
    return app.test_client()


# -- configuration -------------------------------------------------------


def test_default_config_points_at_the_repository_library_folder() -> None:
    config = load_web_config()

    assert config.library_dir, "a library folder must be configured by default"
    assert Path(config.library_dir).name == "test_library"
    assert Path(config.library_dir).is_absolute()
    assert config.library_max_images > 0


def test_a_missing_library_folder_is_not_a_configuration_warning(tmp_path: Path) -> None:
    """Falling back to the example images is expected, not a misconfiguration."""

    config_file = tmp_path / "web.yml"
    config_file.write_text("demo:\n  library_dir: 'nowhere_at_all'\n", encoding="utf-8")

    config = load_web_config(config_file)

    assert not any("nowhere_at_all" in warning for warning in config.warnings)


# -- scanning ------------------------------------------------------------


def test_listing_includes_images_and_skips_everything_else(library_dir: Path) -> None:
    library = ImageLibrary(library_dir)

    names = [item.image_id for item in library.list_images()]

    assert "optical_scan.png" in names
    assert "sem_view.tif" in names
    assert "Second_Sample.PNG" in names, "extension matching must be case insensitive"
    assert "notes.txt" not in names
    assert "README.md" not in names
    assert "ignored.png" not in names, "subfolders are ignored by design"


def test_labels_are_readable_versions_of_the_filename(library_dir: Path) -> None:
    library = ImageLibrary(library_dir)

    labels = {item.image_id: item.label for item in library.list_images()}

    assert labels["optical_scan.png"] == "optical scan"


def test_listing_is_capped_at_the_configured_maximum(tmp_path: Path) -> None:
    root = tmp_path / "big_library"
    for index in range(8):
        _write_image(root / f"image_{index:02d}.png")

    library = ImageLibrary(root, max_images=3)

    assert len(library.list_images()) == 3


def test_images_added_after_the_first_scan_appear(library_dir: Path) -> None:
    """New files must show up without restarting, since operators drop them in live."""

    library = ImageLibrary(library_dir)
    before = len(library.list_images())

    _write_image(library_dir / "added_later.png")
    # The listing cache is deliberately short-lived; expire it explicitly rather
    # than sleeping through it.
    library._listing_expires_at = 0.0

    names = [item.image_id for item in library.list_images()]
    assert len(names) == before + 1
    assert "added_later.png" in names


@pytest.mark.parametrize("missing", ["", "does_not_exist_anywhere"])
def test_absent_library_reports_unavailable(tmp_path: Path, missing: str) -> None:
    target = str(tmp_path / missing) if missing else ""

    library = ImageLibrary(target)

    assert library.list_images() == []
    assert library.available() is False


def test_empty_library_folder_reports_unavailable(tmp_path: Path) -> None:
    empty = tmp_path / "empty_library"
    empty.mkdir()

    assert ImageLibrary(empty).available() is False


def test_a_file_where_the_folder_should_be_is_handled(tmp_path: Path) -> None:
    not_a_folder = tmp_path / "test_library"
    not_a_folder.write_text("oops", encoding="utf-8")

    assert ImageLibrary(not_a_folder).available() is False


# -- identifier handling -------------------------------------------------


@pytest.mark.parametrize(
    "image_id",
    [
        "",
        ".",
        "..",
        "../secret.png",
        "..\\secret.png",
        "nested/ignored.png",
        "nested\\ignored.png",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "notes.txt",
        "optical_scan.png\x00.txt",
        "missing_file.png",
    ],
)
def test_hostile_or_unknown_ids_resolve_to_nothing(library_dir: Path, image_id: str) -> None:
    assert resolve_library_image(library_dir, image_id) is None


def test_a_valid_id_resolves_inside_the_library(library_dir: Path) -> None:
    resolved = resolve_library_image(library_dir, "optical_scan.png")

    assert resolved is not None
    assert resolved.parent == library_dir.resolve()


def test_a_symlink_pointing_outside_the_library_is_refused(
    library_dir: Path, tmp_path: Path
) -> None:
    outside = _write_image(tmp_path / "outside" / "private.png")
    link = library_dir / "innocent.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks unprivileged")

    assert resolve_library_image(library_dir, "innocent.png") is None


# -- http api ------------------------------------------------------------


def test_api_lists_the_library_when_it_is_populated(library_dir: Path) -> None:
    payload = _client(library_dir).get("/api/library").get_json()

    assert payload["ok"] is True
    assert payload["available"] is True
    assert payload["source"] == "library"
    assert payload["count"] == 3
    entry = payload["images"][0]
    assert entry["url"].startswith("/api/library/")
    assert entry["thumb_url"].endswith("/thumb")


def test_api_falls_back_to_the_configured_examples(tmp_path: Path) -> None:
    payload = _client(tmp_path / "not_there").get("/api/library").get_json()

    assert payload["available"] is False
    assert payload["source"] == "fallback"
    assert payload["images"] == []
    assert payload["samples"], "the built-in examples must still be offered"


def test_api_serves_full_images_and_smaller_thumbnails(library_dir: Path) -> None:
    client = _client(library_dir)

    full = client.get("/api/library/optical_scan.png")
    thumb = client.get("/api/library/optical_scan.png/thumb")

    assert full.status_code == 200
    assert thumb.status_code == 200
    assert thumb.mimetype == "image/jpeg"
    with Image.open(io.BytesIO(thumb.data)) as handle:
        assert max(handle.size) <= 320


def test_api_thumbnails_downscale_a_large_image(tmp_path: Path) -> None:
    root = tmp_path / "test_library"
    _write_image(root / "large.png", width=1400, height=1000)

    thumb = _client(root).get("/api/library/large.png/thumb")

    with Image.open(io.BytesIO(thumb.data)) as handle:
        assert max(handle.size) == 320


def test_api_rejects_traversal_and_unknown_ids(library_dir: Path) -> None:
    client = _client(library_dir)

    for image_id in ["..%2F..%2F.gitignore", "notes.txt", "missing.png", "nested%2Fignored.png"]:
        assert client.get(f"/api/library/{image_id}").status_code == 404
        assert client.get(f"/api/library/{image_id}/thumb").status_code == 404


def test_index_page_offers_the_library_when_available(library_dir: Path) -> None:
    body = _client(library_dir).get("/").get_data(as_text=True)

    assert "Browse the server library" in body
    assert '"libraryAvailable": true' in body.replace("'", '"')


# -- segmentation --------------------------------------------------------


def test_segmenting_a_library_image_succeeds(library_dir: Path) -> None:
    response = _client(library_dir).post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "library_id": "optical_scan.png"},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200, payload
    assert payload["ok"] is True
    assert payload["image_origin"] == "library"
    assert payload["used_example_image"] is True
    assert payload["source_name"] == "optical_scan.png"
    assert payload["images"]["mask_png_b64"]
    assert "area_fraction" in payload["metrics"]


def test_segmenting_an_unknown_library_image_is_a_clean_404(library_dir: Path) -> None:
    response = _client(library_dir).post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "library_id": "../../setup.py"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "NOT_FOUND"


def test_a_library_image_larger_than_the_upload_limit_is_still_accepted(tmp_path: Path) -> None:
    """The upload ceiling guards the network path, which library images bypass."""

    root = tmp_path / "test_library"
    # Random pixels compress poorly, so this comfortably exceeds the 5 MB ceiling.
    _write_image(root / "huge.png", width=1800, height=1400)
    assert (root / "huge.png").stat().st_size > 5 * 1024 * 1024

    response = _client(root).post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "library_id": "huge.png"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200, response.get_json()


def test_uploads_and_examples_still_work_alongside_the_library(library_dir: Path) -> None:
    """The library must not displace the two paths that already existed."""

    client = _client(library_dir)
    buffer = io.BytesIO()
    Image.fromarray(
        np.random.default_rng(3).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    ).save(buffer, format="PNG")

    uploaded = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "image": (io.BytesIO(buffer.getvalue()), "u.png")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    assert uploaded.get_json()["image_origin"] == "upload"

    listed = client.get("/api/samples").get_json()["samples"]
    sampled = client.post(
        "/api/segment",
        data={"model_id": "hydride_conventional", "sample_id": listed[0]["id"]},
        content_type="multipart/form-data",
    )
    assert sampled.status_code == 200
    assert sampled.get_json()["image_origin"] == "sample"


def test_a_request_naming_no_image_is_rejected(library_dir: Path) -> None:
    response = _client(library_dir).post(
        "/api/segment",
        data={"model_id": "hydride_conventional"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "NO_IMAGE"
