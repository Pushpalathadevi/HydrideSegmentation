"""Sphinx configuration for the MicroSeg / HydrideSegmentation documentation site."""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

project = "MicroSeg Documentation"
author = "HydrideSegmentation contributors"
copyright = f"{_dt.datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "attrs_block",
    "attrs_inline",
    "substitution",
]
myst_fence_as_directive = ["mermaid"]

myst_heading_anchors = 3
autosectionlabel_prefix_document = True
templates_path = ["_templates"]
exclude_patterns = ["_build", ".DS_Store", "**/__pycache__/**"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
language = "en"
pygments_style = "sphinx"
html_theme = "furo"
html_title = "MicroSeg Documentation"
html_baseurl = os.environ.get("MICROSEG_DOCS_BASEURL", "")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = ""
html_show_sourcelink = True
html_copy_source = False
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-sidebar-background": "#f3f7fd",
        "color-sidebar-background-border": "#c7d4e8",
        "color-sidebar-item-background": "#f3f7fd",
        "color-sidebar-item-background--current": "#dce8ff",
        "color-sidebar-item-background--hover": "#e7efff",
        "color-sidebar-item-expander-background": "#d6e2f6",
        "color-sidebar-item-expander-background--hover": "#c7d9f5",
        "color-sidebar-link-text": "#30435a",
        "color-sidebar-link-text--top-level": "#0b5fff",
        "color-sidebar-caption-text": "#53657c",
        "color-sidebar-brand-text": "#0a2b52",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-background--focus": "#ffffff",
        "color-sidebar-search-border": "#c7d4e8",
        "color-sidebar-search-foreground": "#17212b",
        "color-sidebar-search-icon": "#5d7594",
    },
}
html_context = {
    "display_github": False,
    "current_year": _dt.datetime.now().year,
}

mathjax3_config = {
    "tex": {
        "packages": {"[+]": ["ams"]},
    }
}

nitpicky = False
numfig = True
