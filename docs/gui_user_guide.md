# GUI User Guide (Qt Desktop)

## Primary Workflows

1. Load image and select model.
2. Run segmentation.
3. Inspect prediction in split view.
4. Correct annotations.
5. Export corrected sample.
6. Package datasets for training.
7. Save session and resume later.
8. Run full train/infer/evaluate/package jobs from Workflow Hub.

## Correction Workflow

Tools:
- `brush`: paint/erase locally
- `polygon`: click polygon vertices, right-click to commit
- `lasso`: freehand region commit on mouse release
- `feature_select`: click connected component to delete (erase mode) or relabel (add mode)

Class controls:
- `Edit Classes` allows `index,name,#RRGGBB[,description]` editing.
- `Class` selector sets active class index for add-mode drawing and relabel operations.

Inspection controls:
- zoom in/out/reset
- synchronized split-view pan/zoom
- transparency sliders for predicted, corrected, and diff layers

## Exporting Corrections

`Export Corrected Sample` supports selectable formats:
- indexed PNG
- color PNG
- NumPy `.npy`

Output includes correction metadata and provenance (`correction_record.json`).

## Session Persistence

- `Save Session` writes a restartable project folder with images, masks, class map, notes, and UI state.
- `Load Session` restores run state and correction workspace.

## Pipeline Hub

The `Workflow Hub` tab includes orchestration sub-tabs:
- `Inference`: launches `microseg-cli infer`
- `Training`: launches `microseg-cli train`
- `Evaluation`: launches `microseg-cli evaluate`
- `Packaging`: launches `microseg-cli package`

Operational behavior:
- one active orchestration job at a time
- live command output log
- job completion/failure status dialogs
- config path + override support per job
- per-job GPU controls (`Enable GPU` + `device policy`) with CPU fallback behavior
- default is CPU execution unless GPU is explicitly enabled
