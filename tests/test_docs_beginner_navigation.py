from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_beginner_docs_pages_exist() -> None:
    expected = [
        "learning_path.md",
        "why_tradeoffs.md",
        "glossary.md",
        "archive_index.md",
        "student_notebooks.md",
        "tutorials/index.md",
        "tutorials/01_data_preparation_and_dataset_planning.md",
        "tutorials/02_ml_training_with_pixel_baselines.md",
        "tutorials/03_post_processing_and_human_correction.md",
        "tutorials/04_evaluation_testing_and_report_reading.md",
    ]
    for name in expected:
        assert (DOCS / name).exists(), name


def test_docs_conf_enables_notebook_rendering() -> None:
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert '"myst_parser"' in conf
    assert '"nbsphinx"' not in conf
    assert '".ipynb"' not in conf


def test_index_routes_beginner_path_and_archive() -> None:
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for marker in ("learning_path", "student_notebooks", "why_tradeoffs", "archive_index", "tutorials/index"):
        assert marker in index, marker


def test_student_notebooks_list_rendered_pages() -> None:
    text = (DOCS / "student_notebooks.md").read_text(encoding="utf-8")
    for marker in (
        "tutorials/01_data_preparation_and_dataset_planning.html",
        "tutorials/02_ml_training_with_pixel_baselines.html",
        "tutorials/03_post_processing_and_human_correction.html",
        "tutorials/04_evaluation_testing_and_report_reading.html",
    ):
        assert marker in text, marker
