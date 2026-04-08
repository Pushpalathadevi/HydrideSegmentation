# MicroSeg Documentation

```{raw} html
<div class="hero-panel">
<h1>MicroSeg Documentation</h1>
<p>Scientific, local-first documentation for hydride segmentation today and broader microstructural segmentation tomorrow.</p>
<p>This site is built from the repository sources with Sphinx, MyST Markdown, and SVG-first diagrams so the documentation can ship as dynamic HTML and PDF artifacts.</p>
</div>
```

## What This Site Covers

- Project mission, non-negotiable principles, and current delivery status
- Beginner-friendly reading paths, glossary support, and a clear notebook ladder
- Exact GUI and CLI commands for day-to-day use
- Output locations, report structures, and analysis workflows
- Algorithms, mathematics, metrics, and scientific validation logic
- Classical segmentation flow sheets with parameter meaning and tuning guidance
- Model-family architecture comparison with citations and performance trade-offs
- A GUI integration guide for bringing trained models into desktop inference
- A worked comparison example for conventional vs ML outputs
- A decision tree for choosing a starting model
- Developer guidance for extending the platform without breaking traceability

## Build Commands

HTML:

```bash
python scripts/build_docs.py --html-only
```

HTML + PDF:

```bash
python scripts/build_docs.py
```

Serve the generated HTML locally:

```bash
python -m http.server 8000 -d docs/_build/html
```

## Current State Snapshot

- CPU-first local inference is available.
- Qt desktop workflows, correction export, and result packaging are implemented.
- Config-driven CLI and GUI orchestration share the same core contracts.
- Training, evaluation, deployment, and feedback loops emit structured reports.
- Student notebook tutorials cover the hydride sample workflow from dataset prep to evaluation.
- The docs system itself is a first-class build artifact with searchable notebook pages.

## Primary Entry Points

```{figure} diagrams/documentation_system.svg
:alt: Documentation build and consumption flow
:class: wide

Documentation source-to-output pipeline.
```

```{figure} diagrams/code_architecture_overview.svg
:alt: MicroSeg architecture overview
:class: wide

Repository architecture and data-flow map.
```

```{figure} diagrams/results_workspace_schematic.svg
:alt: GUI workspace schematic
:class: wide

Schematic view of the desktop results workspace and export surfaces.
```

## Start Here

```{toctree}
:maxdepth: 1
:caption: Start Here

learning_path
student_onramp
documentation_principles
mission_statement
glossary
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

student_notebooks
tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Why / Tradeoffs

why_tradeoffs
algorithms
conventional_segmentation_pipeline
model_selection_decision_tree
worked_example_conventional_vs_ml
```

```{toctree}
:maxdepth: 1
:caption: Reference

usage_commands
results_analysis
gui_user_guide
configuration_workflow
scientific_validation
model_architecture_manuscript_foundation
gui_model_integration_guide
developer_guide
README
```

```{toctree}
:maxdepth: 1
:caption: Archive

archive_index
```
