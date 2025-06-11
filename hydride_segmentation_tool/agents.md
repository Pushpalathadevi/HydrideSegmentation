* **Modularity first** – All GUI elements live in `hydride_app/gui/`.
* **Headless core** – Every function in `hydride_app/core/` must run without a display.
* **Type-hinted & documented** – Use PEP 484 type hints plus Google-style docstrings.
* **Unit-tested** – Any new public function requires a matching pytest test.
* **Offline ready** – Favor pure-Python & include fallback stubs for heavy ML libs.
* **No hard-coded paths** – Use `hydride_app.utils.paths` helpers.
* **Public APIs** are re-exported in `__init__.py` for clean import paths.
* **Respect the end-user** – GUI stays simple: load → segment → visualize → export.
