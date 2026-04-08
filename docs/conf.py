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
myst_fence_as_directive = []

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
html_extra_path = ["notebooks"]
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
        "color-brand-primary": "#b07d2a",
        "color-brand-content": "#8f651f",
        "color-api-name": "#19324a",
        "color-api-pre-name": "#19324a",
        "color-sidebar-background": "#12324c",
        "color-sidebar-background-border": "#0b2031",
        "color-sidebar-item-background": "#12324c",
        "color-sidebar-item-background--current": "#1d466a",
        "color-sidebar-item-background--hover": "#1a3e5c",
        "color-sidebar-item-expander-background": "#1f4d72",
        "color-sidebar-item-expander-background--hover": "#275c86",
        "color-sidebar-link-text": "#edf4fb",
        "color-sidebar-link-text--top-level": "#ffe8b3",
        "color-sidebar-caption-text": "#c8d7e6",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-search-background": "#0f2740",
        "color-sidebar-search-background--focus": "#143556",
        "color-sidebar-search-border": "#355c7f",
        "color-sidebar-search-foreground": "#f5f8fb",
        "color-sidebar-search-icon": "#c5d6e7",
    },
}
html_context = {
    "display_github": False,
    "current_year": _dt.datetime.now().year,
}

mathjax_local_path = DOCS / "_static" / "mathjax" / "es5" / "tex-mml-chtml.js"

if mathjax_local_path.exists():
    mathjax_path = "mathjax/es5/tex-mml-chtml.js"
else:
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/es5/tex-mml-chtml.js"

mathjax_common_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "packages": {"[+]": ["ams"]},
    }
}

# Sphinx 9 defaults to MathJax v4. Keep the v3 alias for compatibility with older local builds.
mathjax4_config = mathjax_common_config
mathjax3_config = mathjax_common_config

nitpicky = False
numfig = True
