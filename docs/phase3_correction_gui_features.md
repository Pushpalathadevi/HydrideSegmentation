# Phase 3 Correction GUI Feature Plan and Implementation

Date: 2026-02-15

## Design Goals

The correction UI must optimize precision, speed, and confidence for human-in-the-loop editing of segmentation masks.

Primary UX goals:
- fast navigation on large micrographs
- clear visual separation of prediction vs correction vs differences
- low-friction correction actions with safe undo/redo
- exportable provenance-ready correction outputs

## Implemented Feature Set

### 1. Navigation and View Control
- Zoom in/out/reset controls in toolbar
- Ctrl+mouse-wheel zoom on correction canvas
- Zoom status indicator in UI

### 2. Layered Annotation Visualization
- Predicted mask overlay toggle + transparency slider
- Corrected mask overlay toggle + transparency slider
- Difference overlay toggle + transparency slider
  - added pixels highlighted (green)
  - removed pixels highlighted (purple)

### 3. Correction Editing Tools
- Brush add mode
- Brush erase mode
- Polygon add/erase mode (left-click points, right-click finalize)
- Lasso add/erase mode (freehand path, release to apply)
- Adjustable brush radius
- Correction reset to initial prediction
- Undo / redo

### 4. Efficiency Features
- Keyboard shortcuts:
  - `B` brush tool
  - `P` polygon tool
  - `L` lasso tool
  - `A` add mode
  - `R` erase mode
  - `Ctrl+Z` undo
  - `Ctrl+Y` redo
  - `Ctrl++` zoom in
  - `Ctrl+-` zoom out
  - `Ctrl+0` zoom reset
  - `Esc` cancel polygon/lasso preview
- Live cursor coordinate indicator
- Live correction action/foreground status indicator

### 5. Workflow Integration
- Works with run history and batch outputs
- Split-view correction workspace with synchronized pan and zoom
- Exports corrected sample with versioned schema metadata
- Compatible with correction dataset packaging CLI

### 6. Operational Robustness
- Structured GUI logging for major user actions and failures
- Exception-safe run/batch/export flows with user-facing error dialogs
- About/help/shortcut dialogs embedded in application menu

## Implementation Map

- Qt UI main window: `hydride_segmentation/qt/main_window.py`
- Layer composition logic: `src/microseg/ui/annotation_view.py`
- Correction session model: `src/microseg/corrections/session.py`

## Test Coverage

- `tests/test_phase3_annotation_view.py`
- `tests/test_phase3_corrections.py`

## Planned Next UX Iterations

- optional edge-snapping assistance
- configurable color themes for color-vision accessibility
- side-by-side multi-mask comparison for reviewer QA
