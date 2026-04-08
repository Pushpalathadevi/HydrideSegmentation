# Glossary

This glossary is intentionally short, practical, and beginner oriented.

| Term | Meaning |
|---|---|
| CLAHE | Contrast-limited adaptive histogram equalization; a local contrast enhancement step |
| Baseline | The simplest method used as a reference before trying a more complex model |
| Leakage | Information from validation or test data accidentally entering training |
| Connected component | One contiguous foreground object in a binary mask |
| Correction export | A versioned package containing corrected masks and provenance metadata |
| Manifest | A machine-readable summary of the files, settings, and outputs for a run |
| Overlay | An image showing the mask on top of the source image |
| Pixel classifier | A lightweight model that predicts classes from per-pixel features |
| Split plan | The rule used to divide a dataset into train, validation, and test subsets |
| Validation exemplar | A fixed sample tracked across epochs to make progress easier to read |
| Warm start | Training from existing weights or a checkpoint instead of random initialization |
| Human-in-the-loop | A workflow where a person reviews or corrects model output |
| Provenance | Metadata showing where an artifact came from and how it was created |
| QA report | A quality-check report that flags dataset or export problems |
| Report | A human-readable summary of results, metrics, and supporting plots |

## Where To Learn More

- Dataset and split planning: [`data_preparation.md`](data_preparation.md)
- Evaluation and validation: [`scientific_validation.md`](scientific_validation.md)
- Algorithm background: [`algorithms.md`](algorithms.md)
- Workflow guidance: [`student_onramp.md`](student_onramp.md)
