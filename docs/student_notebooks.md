# Student Notebook Studio

These notebooks are runnable teaching labs built around the current hydride segmentation workflow.
They are meant to help students understand the code base and the ML workflow shape before they touch a larger dataset or a live GUI session.

The notebook curriculum is published two ways:

- a searchable Sphinx transcript under `tutorials/`
- the raw `.ipynb` files copied into the built documentation output root for direct download

Open the raw notebooks with JupyterLab, VS Code, or any notebook viewer that understands standard Jupyter notebooks.

## Learning Path

1. Read the mission and the student on-ramp.
2. Open the data-preparation notebook and inspect the sample data layout.
3. Train the baseline model notebook and follow the inference loop examples.
4. Inspect correction and export behavior in the post-processing notebook.
5. Finish with evaluation and testing so you can read reports with confidence.

## Notebook Catalog

| Notebook | What It Teaches | Main Code Paths |
|---|---|---|
| <a href="tutorials/01_data_preparation_and_dataset_planning.html">01_data_preparation_and_dataset_planning</a> | Source/mask layout, classical preprocessing, leakage-aware split planning, dataset prep, QA | `src.microseg.dataops` |
| <a href="tutorials/02_ml_training_with_pixel_baselines.html">02_ml_training_with_pixel_baselines</a> | CPU-first ML training, inference on a sample image, batch inference loop examples, manifest writing | `src.microseg.training.pixel_classifier`, optional torch pixel baseline |
| <a href="tutorials/03_post_processing_and_human_correction.html">03_post_processing_and_human_correction</a> | Correction sessions, undo/redo, annotation overlays, export packages | `src.microseg.corrections`, `src.microseg.ui`, `src.microseg.app.desktop_workflow` |
| <a href="tutorials/04_evaluation_testing_and_report_reading.html">04_evaluation_testing_and_report_reading</a> | Hydride statistics, visualizations, dataset QA, report inspection | `src.microseg.evaluation`, `src.microseg.dataops` |

For the transcript-style notebook pages, start with [`tutorials/index.md`](tutorials/index.md).

## How To Navigate The Notebook Pages

- Use the table above if you want the rendered Sphinx transcript page.
- Use the raw `.ipynb` file from the documentation output if you want to open it directly in JupyterLab.
- Run the notebooks in order the first time so you can see how the artifacts flow forward.
- Compare notebook outputs with the GUI screenshots and the command-line examples in `usage_commands.md`.

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

## Related Pages

- [`learning_path.md`](learning_path.md)
- [`why_tradeoffs.md`](why_tradeoffs.md)
- [`glossary.md`](glossary.md)
