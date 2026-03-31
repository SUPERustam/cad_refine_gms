# Redesign Migration Plan for `cad_refine_gms`

## Summary

This document translates the target architecture in `docs/design_doc.md` into an implementation plan for the current repo.

The migration should use a fast cutover, not a long compatibility period. The end state is:

- stage-first configs under `configs/runs/` and `configs/systems/`
- thin root CLIs that delegate only to `cad_rl.pipelines`
- `cad_rl.config` as the only runtime contract authority
- explicit CAD execution under `cad_rl.env`
- filesystem-authoritative manifests, lineage, logs, and stage artifacts

Current repo facts that drive this plan:

- `train.py` already routes through `cad_rl.pipelines`, but `infer.py`, `build_meshes.py`, and `evaluate.py` still assemble lower-level pieces directly
- `cad_rl.config` already contains useful runtime dataclasses and profile resolution, but it still resolves a single legacy `ResolvedExperimentConfig` from flat component configs
- `cad_rl.specs` duplicates runtime task/model contract logic and should be removed early
- `cad_rl.metrics.async_metrics` currently performs CAD code execution, which violates the target `cad_rl.env` boundary
- runtime record dataclasses exist, but stage handoff contracts and per-sample lineage are still incomplete and inconsistent

## Migration Phases

### Phase 1: Config cutover and schema replacement

Introduce the new config tree as the only active runtime layout:

```text
configs/
├── runs/
│   └── <topic>/
│       ├── base.yaml
│       ├── prepare_dataset.yaml
│       ├── train.yaml
│       ├── infer.yaml
│       ├── build_meshes.yaml
│       ├── eval.yaml
│       └── compare.yaml
├── systems/
│   ├── local.yaml
│   ├── slurm_gpu.yaml
│   └── cluster.yaml
└── legacy/
    └── existing flat configs
```

Implementation decisions:

- move existing flat configs under `configs/legacy/` for reference only, not active runtime loading
- replace `ResolvedExperimentConfig`-centric loading with stage-specific config resolution
- define the canonical runtime config types in `cad_rl.config`:
  - `BaseExperimentConfig`
  - `SystemConfig`
  - `PrepareDatasetConfig`
  - `TrainStageConfig`
  - `InferStageConfig`
  - `MeshStageConfig`
  - `EvalStageConfig`
  - `CompareStageConfig`
  - shared subtypes: `LoggingConfig`, `ProfilingConfig`, `CadExecutorConfig`, `MetricSpec`, `RewardTerm`, `ResumePolicy`
- make resolver behavior fixed:
  - load `base.yaml`
  - deep-merge the stage overlay
  - deep-merge the optional system config
  - return the exact stage dataclass needed by the CLI
- remove `cad_rl.specs` during this phase
- replace any `cad_rl.specs` imports with `cad_rl.config` equivalents before later pipeline work starts

Acceptance criteria:

- runtime code no longer depends on flat `experiment.*`, `task.*`, `model.*`, `algorithm.*`, `trainer.*`, or `machine.*` configs
- the active resolver returns stage-specific runtime contracts
- `cad_rl.specs` is gone from runtime imports and tests

### Phase 2: Public CLI and pipeline cutover

Keep the public root entrypoints:

- `prepare_dataset.py`
- `train.py`
- `resume_train.py`
- `infer.py`
- `build_meshes.py`
- `evaluate.py`
- `compare_runs.py`

But make every CLI thin and pipeline-only.

Standardize the CLI contract:

- every root CLI accepts `--config`
- every root CLI may accept `--system`
- every root CLI may accept `--dry-run`
- remove multi-profile and ad hoc workflow flags such as:
  - `--experiment`
  - `--task-profile`
  - `--model-profile`
  - `--var-name`
  - `--batch-size`
  - `--num-workers`
  - raw path-based stage wiring flags

Stage YAMLs become authoritative for:

- upstream artifact references
- checkpoint selection
- split selection
- worker policy
- logging
- profiling
- stage outputs

Additional decisions:

- `resume_train.py` remains a separate entrypoint but simply invokes the train pipeline with `ResumePolicy` from config
- `cad_rl.pipelines` becomes the only orchestration layer for prepare-dataset, train, resume, infer, build-meshes, evaluate, and compare
- root CLIs must not import `cad_rl.metrics`, `cad_rl.data`, `cad_rl.algorithms`, or model helpers directly except through pipeline APIs

Acceptance criteria:

- all seven root CLIs are thin wrappers around pipeline functions
- all operational behavior comes from stage config, not bespoke shell flags
- `--dry-run` prints the resolved stage contract without side effects

### Phase 3: Extract `cad_rl.env` and normalize stage handoffs

Create `cad_rl.env` as the only CAD execution boundary.

Move these responsibilities out of `cad_rl.metrics.async_metrics`:

- code execution
- timeout policy
- worker model
- process isolation
- stdout and stderr capture
- structured failure data
- execution timings

Define the `cad_rl.env` contract around execution request and result objects:

- request contains generated code, `output_var_name`, and execution policy
- result contains status, failure reason, timing, diagnostics, and output mesh or BREP paths

Fix stage boundaries:

- inference generates programs only
- build-meshes executes programs through `cad_rl.env` and writes execution or mesh records
- evaluation computes metrics from mesh-stage outputs only

Normalize per-sample lineage around a stable `sample_id`.

All stage records should carry enough information for replay and debugging:

- `sample_id`
- source artifact reference
- checkpoint reference where relevant
- stage status
- duration
- structured metadata

Acceptance criteria:

- CAD execution no longer occurs inside metric helpers
- failed execution is preserved as structured stage output, not silently converted into missing metrics
- post-training stages are resumable from explicit handoff artifacts

### Phase 4: Reward and metric refactor

Replace hardcoded reward modes with structured reward composition in `cad_rl.config`.

Reward config must resolve as:

- aggregate policy
- ordered reward terms
- metric name
- weight
- transform
- clamp policy
- default-on-failure
- enabled flag

Metrics used for online reward and offline evaluation must resolve through the same registry.

Compatibility requirement:

- current shipped behavior must still be representable through config, including:
  - constant or IoU-driven reward
  - GMS or AOC-GMS profiles
  - current failure reward defaults

Introduce explicit backend seams without changing the workflow shape:

- `ModelBackend`
- `TrainerEngine`
- `RolloutBackend`
- `CadExecutorBackend`

Acceptance criteria:

- reward behavior is chosen by config, not branchy trainer code
- existing experiment profiles can be rewritten in the new tree without adding new reward code
- metric lookup is shared between training-time reward and offline evaluation

### Phase 5: Runtime materialization, logging, and cleanup

Standardize per-stage filesystem layout under the run registry:

- resolved config
- stage manifest
- structured logs
- lineage JSONL
- stage summary
- profiler outputs

`cad_rl.runtime` owns:

- run and stage directories
- manifests
- checkpoint references
- replay metadata
- artifact naming

Logging and debugging rules:

- use Python `logging` for runtime paths
- emit JSONL event logs alongside human-readable console logs
- keep `print` statements out of runtime code except minimal CLI dry-run output
- make debug mode config-driven

Filesystem remains the source of truth. Comet stays mirror-only.

After the cutover is stable, remove dead code tied to:

- legacy flat config loading
- `cad_rl.specs`
- CAD execution inside metric helpers
- root CLIs that bypass pipelines

Acceptance criteria:

- every stage writes a standard manifest and summary
- replay and checkpoint lookup operate from runtime-owned metadata
- profiling and structured logs are attached to stage artifacts

## Public Interfaces and Type Changes

The redesign changes public interfaces in these concrete ways:

- root CLI contract becomes `--config [--system] [--dry-run]` across all public entrypoints
- `cad_rl.config` becomes the only runtime contract authority
- `cad_rl.specs` is deleted instead of preserved as a compatibility wrapper
- runtime config types become stage-specific instead of one generic resolved experiment blob
- `cad_rl.env` becomes a new public package boundary for CAD execution
- evaluation input changes from “inference records plus inline execution” to “mesh-stage records plus manifests”
- stage configs own explicit upstream artifact references and checkpoint selection so stage behavior is config-driven

## Test Plan

Keep verification focused in the existing suite names and expand them to cover the redesign.

### `tests/test_config_runtime.py`

Validate:

- stage config merge for base + stage + system
- active runtime APIs do not resolve legacy flat configs
- stage manifests and registry layout materialize correctly
- root CLI dry-run returns the expected resolved stage contract

### `tests/test_specs_and_artifacts.py`

Validate:

- prepared dataset manifest and fingerprint round-trip
- `InferenceRecord`, `MeshRecord`, and `EvalRecord` round-trip with stable `sample_id`
- stage handoff artifacts are replayable from manifest + input artifact

### `tests/test_training_contracts.py`

Validate:

- structured reward terms reproduce current shipped reward behavior
- resume policy resolves correctly from config
- train-stage contract preserves current training defaults where intended

### `tests/test_dataset_boundary.py`

Validate:

- dataset-prep-only rendering stays under `cad_rl.data`
- training, inference, evaluation, and runtime code do not import dataset-prep internals
- CAD execution lives under `cad_rl.env`, not `cad_rl.metrics`
- no root helper-module imports leak back into package code

### Acceptance scenarios across the workflow

The migration is complete when the following scenarios pass:

- prepare-dataset writes a prepared artifact plus manifest without leaking runtime dependencies
- train and resume both run from stage config only
- infer writes generation records without executing CAD code
- build-meshes executes code through `cad_rl.env` and preserves failures structurally
- evaluate reads mesh-stage outputs and summarizes valid vs invalid samples correctly
- compare reads eval summaries or manifests and produces a comparison report
- any post-training stage can be rerun from its input artifact and manifest without retraining

## Assumptions and Defaults

- the migration optimizes for fast cutover, not long backward compatibility
- `cad_rl.specs` is removed early and not maintained as a wrapper layer
- root CLI names stay unchanged, but their flags are intentionally simplified
- legacy flat configs remain in `configs/legacy/` only as reference examples during migration
- existing runtime record dataclasses can evolve, but the final handoff model is stage-first and lineage-driven
- this file is intentionally named `docs/redesing_plan.md` to match the requested path
