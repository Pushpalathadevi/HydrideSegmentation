# Student Notebook Studio

These notebooks are runnable teaching labs built around the current hydride segmentation workflow.
They are meant to help students understand the code base and the ML workflow shape before they touch a larger dataset or a live GUI session.

The raw `.ipynb` files are copied into the built documentation site under `notebooks/`.
Open them with JupyterLab, VS Code, or any notebook viewer that understands standard Jupyter notebooks.

## Learning Path

1. Read the mission and the student on-ramp.
2. Open the data-preparation notebook and inspect the sample data layout.
3. Train the baseline model notebook on the prepared sample split.
4. Inspect correction and export behavior in the post-processing notebook.
5. Finish with evaluation and testing so you can read reports with confidence.

## Notebook Catalog

| Notebook | What It Teaches | Main Code Paths |
|---|---|---|
| [01_data_preparation_and_dataset_planning.ipynb](notebooks/01_data_preparation_and_dataset_planning.ipynb) | Source/mask layout, leakage-aware split planning, dataset prep, QA | `src.microseg.dataops` |
| [02_ml_training_with_pixel_baselines.ipynb](notebooks/02_ml_training_with_pixel_baselines.ipynb) | CPU-first ML training, manifest writing, inference on a sample image | `src.microseg.training.pixel_classifier`, optional torch pixel baseline |
| [03_post_processing_and_human_correction.ipynb](notebooks/03_post_processing_and_human_correction.ipynb) | Correction sessions, undo/redo, annotation overlays, export packages | `src.microseg.corrections`, `src.microseg.ui`, `src.microseg.app.desktop_workflow` |
| [04_evaluation_testing_and_report_reading.ipynb](notebooks/04_evaluation_testing_and_report_reading.ipynb) | Hydride statistics, visualizations, dataset QA, report inspection | `src.microseg.evaluation`, `src.microseg.dataops` |

## Suggested Way To Use Them

- Keep the notebooks on a scratch branch or a clean workspace.
- Run them in order once so you see how each stage feeds the next one.
- Compare notebook outputs with the GUI screenshots and the command-line examples in `usage_commands.md`.
- Treat the notebook outputs as study artifacts, not as canonical training data unless you replace the demo sample with a curated dataset.

## Practical Notes

- The notebooks write outputs under `outputs/notebook_tutorials/`.
- The demo dataset is derived from the bundled hydride sample image so the workflow is self-contained.
- The examples use the same config and export contracts as the application code.
- The notebooks favor CPU-first, lightweight examples so students can run them on a laptop.
