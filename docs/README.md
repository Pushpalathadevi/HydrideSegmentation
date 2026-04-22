# Documentation Index

This directory is both the human-readable documentation library and the source tree for the Sphinx site.

Start here:

- [`index.md`](index.md) for the rendered landing page
- [`cli_windows_linux.md`](cli_windows_linux.md) for environment setup and command-line troubleshooting
- [`learning_path.md`](learning_path.md) for the beginner route through the repo
- [`student_onramp.md`](student_onramp.md) for the study guide and reading order
- [`tutorials/05_paired_dataset_preparation_and_training_cli.md`](tutorials/05_paired_dataset_preparation_and_training_cli.md) for the canonical paired-folder dataset and training walkthrough
- [`glossary.md`](glossary.md) for beginner-friendly terminology
- [`student_notebooks.md`](student_notebooks.md) for runnable sample-data tutorials
- [`tutorials/index.md`](tutorials/index.md) for the searchable transcript versions of the notebook curriculum
- [`usage_commands.md`](usage_commands.md) for exact command recipes
- [`why_tradeoffs.md`](why_tradeoffs.md) for principles, alternatives, and design choices
- [`documentation_principles.md`](documentation_principles.md) for the normative docs contract
- [`mission_statement.md`](mission_statement.md) for project direction
- [`results_analysis.md`](results_analysis.md) for output locations and report inspection
- [`algorithms.md`](algorithms.md) for the mathematics behind the metrics and trainers
- [`conventional_segmentation_pipeline.md`](conventional_segmentation_pipeline.md) for the classical algorithm flow sheet and parameter guide
- [`model_selection_decision_tree.md`](model_selection_decision_tree.md) for a simple model choice guide
- [`worked_example_conventional_vs_ml.md`](worked_example_conventional_vs_ml.md) for a side-by-side comparison workflow
- [`gui_model_integration_guide.md`](gui_model_integration_guide.md) for air-gapped registration of a trained `.pth` checkpoint into desktop inference
- [`developer_guide.md`](developer_guide.md) for extension and contribution guidance

Core planning and governance docs:

- Mission: `mission_statement.md`
- Baseline audit: `base_zero_audit.md`
- Target architecture: `target_architecture.md`
- Code architecture and data flow map: `code_architecture_map.md`
- Product specification: `local_desktop_product_spec.md`
- Scientific validation protocol: `scientific_validation.md`
- Development workflow: `development_workflow.md`
- Repository health audit: `repo_health_audit.md`
- Deployment and productization roadmap: `deployment_productization_master_roadmap.md`
- Archive and phase history: `archive_index.md`

Build the docs with:

```bash
pip install -r requirements-docs.txt
python scripts/build_docs.py
```

Math rendering is configured for offline-first reproducibility: the repository vendors the MathJax v4 bundle at `docs/_static/mathjax/es5/tex-mml-chtml.js`. Sphinx uses that local asset when present and only falls back to the CDN if the local file is missing.

All flow sheets, schematics, and publication-style diagrams in the docs should be committed as static SVG files under `docs/diagrams/` and referenced from the markdown pages. Inline Mermaid blocks are reserved for temporary authoring only and should not ship in user-facing documentation.
