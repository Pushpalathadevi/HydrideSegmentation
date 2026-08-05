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
- [`hydride_connectivity_index.md`](hydride_connectivity_index.md) for the deferred HCI candidate formulation, prototype audit, scientific decisions, and promotion gates
- [`phase33_interactive_conventional_gui.md`](phase33_interactive_conventional_gui.md) for the side-by-side live conventional-segmentation GUI closeout
- [`model_selection_decision_tree.md`](model_selection_decision_tree.md) for a simple model choice guide
- [`worked_example_conventional_vs_ml.md`](worked_example_conventional_vs_ml.md) for a side-by-side comparison workflow
- [`gui_model_integration_guide.md`](gui_model_integration_guide.md) for installing a trained checkpoint into desktop and CLI inference, including on air-gapped machines
- [`phase34_model_installation.md`](phase34_model_installation.md) for the local model-installation closeout
- [`intranet_web_app.md`](intranet_web_app.md) for serving the browser app to colleagues on an air-gapped intranet
- [`windows_offline_installer.md`](windows_offline_installer.md) for building and validating the single-file Windows installer
- [`releases/v1.0.0.md`](releases/v1.0.0.md) for the first stable release notes, installation guidance, compatibility notes, and validation evidence
- [`releases/v1.0.0.closeout.md`](releases/v1.0.0.closeout.md) for the human-readable release stocktake, final installer checksum, validation results, and remaining gaps
- [`phase35_intranet_web_app.md`](phase35_intranet_web_app.md) for the intranet web application closeout
- [`phase36_memory_safe_interfaces_and_hci_spec.md`](phase36_memory_safe_interfaces_and_hci_spec.md) for memory-safe interfaces and the HCI candidate-spec closeout
- [`frozen_checkpoint_registry.md`](frozen_checkpoint_registry.md) for the registry metadata the installer writes when adding a new trained model
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
