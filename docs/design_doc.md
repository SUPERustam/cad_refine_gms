# Architecture Principles for cad_refine_gms

## Summary

Center the repo on five clear authorities:

- `cad_rl.config`: typed contracts, schema evolution, and profile resolution
- `cad_rl.pipelines`: the only canonical workflow entrypoints
- `cad_rl.runtime`: replayable run registry, artifacts, and manifests
- `cad_rl.env`: CAD code execution as a first-class subsystem
- leaf subsystems (`data`, `models`, `algorithms`, `metrics`): isolated implementation details behind stable interfaces

The repo should optimize for fast research iteration, fast root-cause isolation, and low-friction future extensions such as async GRPO, alternative model backends, and faster CAD execution.

Crucial distinction: evaluation is the post-training measurement pipeline over generated programs, executed meshes, and summaries. Verification is smoke/regression testing of the codebase. They are separate architectural concerns.

## Target Project Tree

The repo should converge toward this top-level structure:

```text
cad_refine_gms/
├── configs/
│   ├── runs/
│   │   └── <research_topic>/
│   │       ├── base.yaml
│   │       ├── prepare_dataset.yaml
│   │       ├── train.yaml
│   │       ├── infer.yaml
│   │       ├── build_meshes.yaml
│   │       ├── eval.yaml
│   │       └── compare.yaml
│   ├── systems/
│   │   ├── local.yaml
│   │   ├── slurm_gpu.yaml
│   │   └── cluster.yaml
│   └── legacy/
│       ├── experiment.*.yaml
│       ├── task.*.yaml
│       ├── model.*.yaml
│       ├── algorithm.*.yaml
│       ├── trainer.*.yaml
│       └── machine.*.yaml
├── cad_rl/
│   ├── config/
│   ├── pipelines/
│   ├── runtime/
│   ├── env/
│   ├── data/
│   ├── models/
│   ├── algorithms/
│   ├── metrics/
│   └── utils.py
├── docs/
├── tests/
├── prepare_dataset.py
├── train.py
├── resume_train.py
├── infer.py
├── build_meshes.py
├── evaluate.py
└── compare_runs.py
```

Meaning of the main boundaries:

- `cad_rl.config`: canonical schemas, compatibility loading, and config resolution
- `cad_rl.pipelines`: one orchestrator per public workflow stage
- `cad_rl.runtime`: registry, manifests, run directories, checkpoint tracking
- `cad_rl.env`: CAD execution, timeouts, worker policy, execution records
- `cad_rl.data`: dataset prep and prepared dataset access only
- `cad_rl.models`: model-family backends such as Transformers and future Unsloth
- `cad_rl.algorithms`: trainer engines, reward aggregation, rollout coordination
- `cad_rl.metrics`: mesh-based metric computation only

## Core Principles

- One canonical contract authority
  `cad_rl.config` is the only runtime schema layer. Do not keep two active spec systems for task/model/eval artifacts.
- One canonical workflow per lifecycle stage
  There is one obvious package entrypoint for prepare-dataset, train, resume, infer, execute/build-meshes, evaluate-metrics, and compare.
- Config drives behavior, not shell flags
  Every real run is driven by 1-3 YAML files: a shared base experiment config, a stage overlay, and an optional machine/system config. CLIs stay thin and mostly accept `--config` and `--dry-run`.
- Separate orchestration from implementation
  Pipelines compose work. Leaf modules implement model loading, dataset access, CAD execution, metric computation, reward aggregation, and registry operations.
- Post-training evaluation is first-class
  Evaluation means:
  trained checkpoint -> inference -> generated CadQuery programs -> execution to meshes/failures -> metric computation against ground truth -> summary/comparison.
  This is part of the research architecture, not an afterthought.
- Evaluation stages are separate but resumable
  Keep infer, mesh execution, and metric evaluation as distinct first-class stages with explicit handoff artifacts. This gives fast reruns, easier root-cause isolation, and cheaper debugging.
- Sample lineage beats opaque batch outputs
  The primary tracked object is a per-sample lineage record that survives across all stages with stable sample id, source asset, render artifact, generated code, execution status, mesh path, metric values, reward breakdown, and timings.
- Replay first
  Every run and every stage must be reproducible from resolved config, code revision, seed where relevant, checkpoint reference, input artifacts, and written manifests.
- Filesystem is the control plane
  Logs, manifests, lineage JSONL, summaries, and profiler outputs live on disk as the source of truth. External trackers such as Comet are mirrors only.
- Operational code stays boring
  Registry layout, artifact naming, checkpoint lookup, logging, and summary materialization are standardized and reused.
- The core stays small
  New research ideas should mostly change algorithm/reward/model/task configuration or a bounded subsystem, not the workflow skeleton.
- Optimize for delegation
  Ownership boundaries should let one engineer work on config/contracts, another on training orchestration, another on CAD execution, another on metrics, without hidden coupling.

## Config Architecture

Adopt a layered config model with grouped YAML fields and no flat JSON-like blobs as the preferred authoring style.

- Shared base experiment config
  Holds common task, dataset contract, model family/backend, CAD env defaults, metric defaults, logging defaults, profiling defaults, and shared metadata.
- Stage overlays
  `train.yaml`, `infer.yaml`, `build_meshes.yaml`, `eval.yaml`, `compare.yaml`, and `prepare_dataset.yaml` override only stage-specific fields.
- Optional system config
  Encodes local-vs-cluster paths, environment variables, launcher policy, worker counts, and hardware placement.

Operator workflow:

- 1 config: stage config already includes everything needed
- 2 configs: base experiment + stage overlay
- 3 configs: base experiment + stage overlay + system config

The config layer should expose canonical runtime types such as:

- `BaseExperimentConfig`
- `SystemConfig`
- `PrepareDatasetConfig`
- `TrainStageConfig`
- `InferStageConfig`
- `MeshStageConfig`
- `EvalStageConfig`
- `CompareStageConfig`
- shared subtypes such as `LoggingConfig`, `ProfilingConfig`, `CadExecutorConfig`, `MetricSpec`, `RewardTerm`, and `ResumePolicy`

`cad_rl.specs` should be reduced to compatibility wrappers or retired from active runtime paths.

## Reward and Metric Architecture

Metric choice and reward composition must be config-driven.

Replace hardcoded reward modes such as `10_iou` with a structured reward specification:

- reward aggregate policy such as `weighted_sum`
- reward terms as a list of:
  - metric name
  - coefficient / weight
  - transform
  - clamp policy
  - default on failure
  - enabled flag

Built-in transforms should cover the common cases needed for research without code edits:

- `identity`
- `negate`
- `scale`
- `clip`
- `log`
- `inverse_log`
- `piecewise_linear`

Metrics used for reward and metrics used for offline evaluation should resolve through the same registry, so “which metric matters” is determined by config, not by branching logic in trainer code.

Defaults should still allow current profiles to be expressed cleanly, including the old `10 * iou` and `10 * aoc_gms` cases.


### Target Config Tree

The current repo still uses flat component configs such as `experiment.*.yaml`, `task.*.yaml`, `model.*.yaml`, `algorithm.*.yaml`, `trainer.*.yaml`, and `machine.*.yaml`. The intended config tree should become:

```text
configs/
├── runs/
│   ├── dr_cppo_gms/
│   │   ├── base.yaml
│   │   ├── train.yaml
│   │   ├── infer.yaml
│   │   ├── build_meshes.yaml
│   │   └── eval.yaml
│   ├── dr_cppo_aoc_gms/
│   │   ├── base.yaml
│   │   ├── train.yaml
│   │   ├── infer.yaml
│   │   ├── build_meshes.yaml
│   │   └── eval.yaml
│   └── demo/
│       ├── base.yaml
│       ├── prepare_dataset.yaml
│       ├── train.yaml
│       ├── infer.yaml
│       ├── build_meshes.yaml
│       ├── eval.yaml
│       └── compare.yaml
├── systems/
│   ├── local.yaml
│   ├── single_node_a100.yaml
│   ├── slurm_train.yaml
│   └── slurm_eval.yaml
└── legacy/
    ├── experiment.demo.yaml
    ├── experiment.dr_cppo_constant.yaml
    ├── experiment.dr_cppo_cosine.yaml
    ├── experiment.dr_cppo_gms.yaml
    ├── experiment.dr_cppo_aoc_gms.yaml
    ├── task.cadquery_v1.yaml
    ├── task.fusion360_test_mesh_1000_infer.yaml
    ├── model.qwen2_vl.yaml
    ├── algorithm.dr_cppo_constant.yaml
    ├── algorithm.dr_cppo_cosine.yaml
    ├── algorithm.dr_cppo_gms.yaml
    ├── algorithm.dr_cppo_aoc_gms.yaml
    ├── trainer.grpo_default.yaml
    ├── trainer.grpo_aoc_gms.yaml
    └── machine.local.yaml
```

Migration rule:

- new work should be authored under `configs/runs/` and `configs/systems/`
- legacy flat configs should be loaded through compatibility adapters only
- once the new layout is stable, legacy configs can be kept as fixtures/examples or retired

## CAD Environment Boundary

Make the RL environment explicit by introducing `cad_rl.env` as the canonical boundary for code execution.

Responsibilities of `cad_rl.env`:

- execute generated CAD code into compounds/meshes
- own timeout policy, worker model, and sandbox/process isolation
- capture structured failure data, stdout/stderr, and execution timings
- expose sync execution now, with a clear path to async rollout/execution later

Responsibilities outside `cad_rl.env`:

- `cad_rl.metrics` computes metrics from meshes/execution outputs
- `cad_rl.algorithms` consumes reward-ready metric outputs
- `cad_rl.pipelines` orchestrates stage handoffs

CAD execution must not stay hidden inside metric helpers. This separation is required both for debugging and for future speedups.

## Dataset Preparation

`prepare_dataset.py` should become a first-class stage pipeline rather than a special-case script.

Its responsibilities:

- input discovery and indexing
- rendering/materialization needed for prepared datasets
- prompt/render metadata generation
- Hugging Face dataset write
- manifest, source index, and fingerprint write

Output contract:

- prepared dataset artifact on disk
- rich manifest with source roots, split, row count, prompt/render settings, normalization mode, and fingerprint

Training and inference should consume prepared artifacts plus their manifest. They should not depend on raw dataset internals.

Dataset-prep-only rendering code stays under `cad_rl.data` and must not leak into training, inference, evaluation, or runtime modules.

## Runtime, Logging, and Debugging

Every run and stage should materialize a standard directory layout with:

- resolved config
- run/stage manifest
- structured logs
- lineage records
- stage summaries
- profiler outputs

Logging design:

- use Python `logging` as the common base
- emit both readable console logs and structured JSONL event logs
- keep `print` statements out of runtime paths except in minimal CLI dry-run output

Debug mode should be config-driven and should enable:

- more verbose stage logs
- captured failing code snippets and failure context
- extra lineage fields
- fail-fast worker settings for quicker debugging loops

Comet or similar trackers may mirror metrics and assets, but the filesystem remains authoritative for resume, debugging, and comparisons.

## Performance and Future Extension Points

Leave explicit seams for future acceleration work:

- `ModelBackend`: current Transformers path, future Unsloth path
- `TrainerEngine`: current synchronous GRPO, future async GRPO
- `RolloutBackend`: local synchronous generation now, queue/worker-based rollout later
- `CadExecutorBackend`: local process pool now, remote or distributed executor later

These seams should be chosen so experimentation mostly changes config and bounded backend modules rather than pipeline structure.

## Profiling

Profiling is a first-class feature, not an ad hoc debugging step.

Support two levels:

- lightweight always-on profiling for wall time, CPU RSS, GPU memory allocated/reserved, and stage timings
- optional deep profiling via tools such as `torch.profiler` and NVML when available

Profiler outputs should be written per stage and attached to run artifacts so memory, CPU, GPU, and GPU memory regressions can be compared over time.

## Public Interfaces and Boundaries

- Public root entrypoints remain:
  `prepare_dataset.py`, `train.py`, `resume_train.py`, `infer.py`, `build_meshes.py`, `evaluate.py`, `compare_runs.py`
- Root CLIs stay thin and delegate to `cad_rl.pipelines` only.
- `cad_rl.runtime` owns artifact locations, manifests, checkpoint references, and replay metadata.
- `cad_rl.env` owns CAD execution.
- `cad_rl.metrics` owns metric computation.
- `cad_rl.algorithms` owns reward aggregation and trainer-specific logic.
- CAD execution and metric computation must never become hidden side effects inside CLI scripts.

## Test Plan

- Verification tests
  Smoke/regression tests remain a separate concern from evaluation. They check contract resolution, boundary rules, and workflow invariants.
- Config tests
  Validate base + stage overlay + system merge, grouped YAML schemas, and compatibility mapping from legacy configs.
- Reward tests
  Validate structured reward terms reproduce current shipped profiles and support new weighted combinations without code edits.
- Dataset prep tests
  Validate prepared dataset manifests, fingerprints, and the boundary between raw data and prepared artifacts.
- Environment tests
  Validate successful execution, syntax/runtime failure, and timeout behavior produce structured records.
- Evaluation pipeline tests
  Add tests for stage handoff contracts:
  inference record -> execution/mesh record -> eval record.
  Assert failed CadQuery execution is preserved as structured status, not lost.
  Assert summaries correctly exclude invalid samples while counting them.
- Replay tests
  Prove any post-training stage can be rerun from its input artifact plus manifest without retraining.
- Profiling/logging smoke tests
  Validate debug mode, structured log emission, and profiler artifact writing.
- Focused minimum suite
  Keep:
  `tests/test_config_runtime.py`
  `tests/test_specs_and_artifacts.py`
  `tests/test_training_contracts.py`
  `tests/test_dataset_boundary.py`

## Assumptions and Defaults

- `cad_rl.config` becomes the only runtime contract authority.
- Stage-specific YAMLs are first-class, but they inherit from one shared base experiment config.
- Real runs are driven by config, not by operational CLI flags.
- Reward flexibility is delivered through structured weighted metric terms, not free-form Python formulas in YAML.
- Evaluation is defined as the post-training mesh/metric pipeline, not smoke testing.
- Verification tests remain first-class, but under a different architectural bucket.
- Post-training evaluation should stay as separate resumable stages, not one opaque monolith.
- Per-sample lineage is the main debugging and analysis primitive.
- Filesystem artifacts remain the control plane; external trackers are mirrors only.
