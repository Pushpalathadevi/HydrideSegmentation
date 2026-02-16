# GUI User Guide (Qt Desktop)

## Primary Workflows

1. Load image and select model.
2. Run segmentation.
3. Inspect prediction in split view.
4. Correct annotations.
5. Export corrected sample.
6. Package datasets for training.
7. Save session and resume later.

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

The `Workflow Hub` tab supports dataset packaging:
- input correction exports directory
- output dataset directory
- deterministic split ratios and seed
