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
- Exact GUI and CLI commands for day-to-day use
- Output locations, report structures, and analysis workflows
- Algorithms, mathematics, metrics, and scientific validation logic
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
- The docs system itself is now a first-class build artifact.

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
:caption: Core

documentation_principles
mission_statement
current_state_gap_analysis
code_architecture_map
usage_commands
results_analysis
algorithms
developer_guide
```

```{toctree}
:maxdepth: 1
:caption: Existing Reference Docs

gui_user_guide
configuration_workflow
scientific_validation
model_architecture_manuscript_foundation
deployment_ops_workflow
hpc_ga_user_guide
hydride_research_workflow
development_workflow
README
```
