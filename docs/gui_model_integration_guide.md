# How To Integrate a Trained Model Into the GUI for Inference

## Purpose

This guide explains how to take a trained model and make it available inside the desktop GUI so it can be used for inference.

The goal is to make the process beginner-friendly while still respecting the repository's scientific traceability rules.

## What "Integrated Into The GUI" Means

For this project, a model is considered integrated when:

1. the model appears in the GUI model selector,
2. the GUI can resolve it to a model ID,
3. the inference backend can load the associated checkpoint or pretrained bundle,
4. the GUI can run inference on a chosen image,
5. the output can be reviewed and exported with the normal desktop workflow.

## High-Level Flow

![GUI model integration workflow](diagrams/gui_model_integration_guide.svg)

The workflow is a static SVG so the page remains legible offline.

## Step 1 - Confirm the Model Type

First decide what kind of model you have:

- a conventional rule-based baseline,
- a learned PyTorch checkpoint,
- a Hugging Face model directory,
- an internal hybrid model,
- a partial warm-start bundle.

This matters because the loader path is different for each one.

## Step 2 - Make Sure the Model Is Scientifically Legitimate

Before touching the GUI, confirm:

- the training run finished successfully,
- the report exists,
- the checkpoint can be loaded,
- the architecture recorded in metadata matches the model you trained,
- the class map is correct,
- the output mask format is compatible with the correction/export path.

For ML models, also confirm that the model was validated on representative samples before you call it a candidate for routine use.

## Step 3 - Register The Model

The GUI does not discover models by guessing filenames. It relies on metadata.

That means you should update the frozen registry or the model discovery layer so the UI can display:

- a human-readable name,
- the backend family,
- the model ID,
- the checkpoint hint,
- the description,
- the applicability note.

For frozen models, the canonical registry is:

- [`frozen_checkpoints/model_registry.json`](../frozen_checkpoints/model_registry.json)

If the checkpoint is local-only and should stay out of git, place its binary under `frozen_checkpoints/candidates/` and write an ignored overlay registry at `frozen_checkpoints/model_registry.local.json`. The GUI and CLI will merge that overlay with the canonical registry at runtime.

## Step 4 - Make The Loader Understand The Artifact

The runtime loader must be able to resolve one of these:

- `run_dir`
- `registry_model_id`
- `checkpoint_path`

If the model uses a different storage format, add the appropriate loader path before exposing it in the GUI.

Important:

- do not expose a model in the GUI unless the loader path is already tested,
- do not rely on manual filename conventions alone.

## Step 5 - Confirm The GUI Mapping

The GUI model selector maps display names to model IDs through the compatibility layer.

The important rule is:

- the GUI label is for users,
- the model ID is for the runtime.

So when you add a new model, verify:

- the display name appears,
- the display name resolves to the correct model ID,
- the model ID matches the registry and loader metadata.

## Step 6 - Test Inference Outside The GUI First

Before using the desktop app, test the model on the command line.

Recommended checks:

1. load the model,
2. run a single image,
3. verify the output mask shape,
4. inspect the overlay,
5. confirm the result is deterministic where expected.

This isolates model issues from GUI issues.

## Step 7 - Run A GUI Smoke Test

Open the desktop app and:

1. select the model,
2. load a sample image,
3. run inference,
4. inspect the mask,
5. inspect the overlay,
6. confirm the output paths or export options.

The desktop inference action now runs in a CLI subprocess, so the window should stay responsive while a model is processing. Long-running models can still take time to finish, but the UI should not freeze while they do, and the exported result is loaded back into the GUI when the subprocess completes.

If the model is conventional:

- the conventional parameter controls should be visible.

If the model is learned:

- the GUI should hide the rule-based controls and show the inference-oriented workflow instead.

## Step 8 - Record The Integration Metadata

For traceability, write down:

- model ID,
- architecture,
- checkpoint path or bundle directory,
- source revision or run ID,
- class map,
- input size assumptions,
- device policy,
- validation status.

This is what makes the GUI integration scientifically reproducible.

## Common Failure Modes

### The model does not appear in the GUI

Possible causes:

- not registered in the model metadata,
- wrong display-name mapping,
- stale docs or stale cache,
- loader does not recognize the artifact type.

### The model appears but inference fails

Possible causes:

- checkpoint architecture mismatch,
- missing pretrained bundle files,
- unsupported class map,
- wrong device policy,
- incompatible input dimensions.

### The model runs but results look wrong

Possible causes:

- trained on a different label convention,
- preprocessing mismatch,
- improper normalization,
- model was never validated on the target domain,
- output is correct mathematically but scientifically weak.

## Beginner Checklist

- Do not skip registry validation.
- Do not skip a command-line smoke test.
- Do not assume a trained model is automatically GUI-ready.
- Do not expose a model to users before the loader path is verified.
- Do document the model family and checkpoint provenance.

## Practical Mapping Hint

If you are adding a model to the current desktop stack, update these layers together:

- model registry,
- inference loader,
- GUI metadata adapter,
- docs page,
- smoke test or unit test.

That keeps the code and the documentation aligned.

## Related Pages

- [`docs/model_selection_decision_tree.md`](model_selection_decision_tree.md)
- [`docs/model_architecture_manuscript_foundation.md`](model_architecture_manuscript_foundation.md)
- [`docs/pretrained_model_catalog.md`](pretrained_model_catalog.md)
- [`docs/usage_commands.md`](usage_commands.md)
