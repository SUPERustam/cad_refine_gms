# Migration Status

The migration from the legacy layered layout to the target architecture in `docs/design_doc.md` is complete in the current codebase.

## Completed Against The Design

- `cad_rl/` uses the flat stage-module layout from the design doc.
- `cad_rl.data` remains the only package directory and owns dataset prep, dataset loading, and dataset-prep rendering helpers.
- Runtime record dataclasses now live in `cad_rl.runtime`.
- The active config tree is the experiment-directory layout:
  - `configs/<experiment>/common.yaml`
  - `configs/<experiment>/<stage>.yaml`
  - optional `configs/systems/*.yaml`
- Flat legacy config profiles have been removed.
- Root CLIs are thin wrappers over package-stage entrypoints.
- The execution/metrics boundary is split so execution returns execution results and metrics consume them.

## Verification

Minimum verification command:

```bash
pytest tests/test_config_runtime.py tests/test_specs_and_artifacts.py tests/test_training_contracts.py tests/test_dataset_boundary.py
```

Current acceptance criteria:

- every public CLI supports `--config`, optional `--system`, and optional `--dry-run`
- root CLIs remain thin wrappers only
- filesystem artifacts remain the source of truth for stage handoffs and resume state
- training, inference, evaluation, and runtime code do not import dataset-prep-only rendering helpers directly
- no flat legacy config profiles remain under `configs/`
