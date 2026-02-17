# Repository Blueprint (Target Layout)

This layout is the migration target. Existing modules remain until replaced.

```
HydrideSegmentation/
  AGENTS.md
  README.md
  docs/
    mission_statement.md
    base_zero_audit.md
    target_architecture.md
    development_roadmap.md
    scientific_validation.md
    training_data_requirements.md
    local_desktop_product_spec.md
    repository_blueprint.md
    adr/
  src/
    microseg/
      domain/
      core/
      corrections/
      dataops/
      inference/
      evaluation/
      training/
      plugins/
      pipelines/
      io/
      ui/
      app/
      utils/
      quality/
  apps/
    desktop/
      # Qt app shell and packaging assets
  configs/
    models/
    pipelines/
    app/
  data/
    raw/
    interim/
    processed/
    corrected_exports/
  models/
    registry/
    weights/
  frozen_checkpoints/
    model_registry.json
    smoke/
    candidates/
    promoted/
  scripts/
  tests/
    unit/
    integration/
    e2e/
  hydride_segmentation/   # compatibility layer during migration
    qt/                   # phase-3 Qt GUI implementation
```

## Migration Rule

Do not remove `hydride_segmentation/` until equivalent behavior exists in `src/microseg` with passing tests and updated docs.
