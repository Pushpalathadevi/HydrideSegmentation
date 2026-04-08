# Principles, Best Practices, and Tradeoffs

This page explains why the repository does things the way it does and what the main alternatives are.
It is intended to help students compare methods without losing the scientific context.

## Core Decisions

| Decision | Why this repo favors it | Common alternative | When the alternative makes sense |
|---|---|---|---|
| CPU-first runtime | Students can run the workflow locally without GPU access | GPU-only workflows | Large training runs or HPC sweeps |
| Baseline before ML | You need a known reference before judging improvement | Train a complex model immediately | When the baseline is already established and documented |
| Leakage-aware split planning | Near-duplicate hydride samples can make results look better than they are | Random splitting only | When the dataset has no grouping or augmentation families |
| Versioned correction export | Corrected masks are scientific artifacts and should be traceable | Ad hoc image export | Quick experiments with no downstream reuse |
| Fixed validation exemplars | Stable sample panels make training progress interpretable | Fully random validation sampling | Very large datasets where coverage matters more than a fixed panel |
| GUI hides secondary controls | Images and plots deserve the most screen space | Everything visible all the time | Power-user workflows on a very large screen |
| Sphinx + notebooks | Students need searchable explanations plus runnable labs | PDF-only handouts | Static review without code exploration |

## Why The Baseline Comes First

The classical path is not a side quest.
It tells you what the data look like before a model learns anything.
If a simple thresholding or morphology step already explains the structure well, the model must justify itself against that baseline.

Use the baseline to answer:

- Are the boundaries visible at all?
- Is the signal strong enough for automatic segmentation?
- Do you need preprocessing, not just more model capacity?
- Are you over-optimizing a model for a data problem?

## Why The Repo Keeps Correction Explicit

Correction is not just a UI convenience.
It is how the project turns model output into reusable training or evaluation data.

Explicit correction keeps these things visible:

- what the model predicted
- what the human changed
- when the change happened
- which class map and export schema were used
- how to reproduce the corrected artifact later

## Why Sphinx And Not Docs-In-A-Notebook Only

The repo needs both:

- notebooks for hands-on experimentation
- Sphinx pages for search, navigation, cross-links, and stable reading order

The Sphinx site is the index.
The notebooks are the labs.

## Other Choices Worth Knowing

| Topic | Current choice | Other valid choice |
|---|---|---|
| Segmentation family | Hydride-focused workflows, designed to generalize | A domain-specific app for a single specimen type only |
| Notebook style | Explanatory, runnable, repo-linked labs | Minimal code-only notebooks |
| Artifact style | JSON/HTML reports plus images and manifests | Images only |
| User flow | GUI + CLI + notebooks | GUI-only or CLI-only |
| Runtime policy | CPU-safe default, GPU opt-in | GPU-required default |

## Related Reading

- [`algorithms.md`](algorithms.md)
- [`conventional_segmentation_pipeline.md`](conventional_segmentation_pipeline.md)
- [`model_selection_decision_tree.md`](model_selection_decision_tree.md)
- [`worked_example_conventional_vs_ml.md`](worked_example_conventional_vs_ml.md)
