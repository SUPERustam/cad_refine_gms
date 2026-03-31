# KISS Design for `cad_refine_gms`

## Goal

__Keep the project easy to change and easy to debug.__

The repo has one main workflow and one module per stage:

`prepare_dataset -> train/resume -> infer -> build_meshes -> evaluate -> compare`

The design should optimize for:

- one obvious place for each responsibility
- one obvious module for each stage
- thin public CLIs at the repo root
- config-driven runs
- simple stage handoffs on disk
- no speculative abstraction layers
- no wrapper packages that only forward calls

If there is only one implementation today, keep one implementation today. Add a new interface only when a second real implementation requires it.

## Project Rules

- Root Python files are public CLIs only:
  `prepare_dataset.py`, `train.py`, `resume_train.py`, `infer.py`, `build_meshes.py`, `evaluate.py`, `compare_runs.py`
- Reusable code lives under `cad_rl/`
- Use flat stage modules under `cad_rl/`: `training`, `inference`, `execution`, `evaluation`, `comparison`
- `cad_rl.data` is the only place allowed to contain dataset-prep rendering code
- optional `vis_for_norm_parts` support must stay inside `cad_rl.data`
- training, inference, evaluation, and runtime code must not depend on dataset-prep-only rendering helpers
- CLIs stay thin and delegate into those package modules
- Keep one active config model and one active runtime record model
- Do not keep duplicate package layers such as separate `pipelines`, `specs`, or tiny forwarding packages
- filesystem artifacts are the source of truth for stage outputs and resume state

## Target Repo Shape

```text
cad_refine_gms/
├── configs/
│   ├── runs/
│   │   ├── demo.yaml
│   │   ├── dr_cppo_gms.yaml
│   │   └── dr_cppo_aoc_gms.yaml
│   └── systems/
│       └── local.yaml
├── cad_rl/
│   ├── __init__.py
│   ├── config.py
│   ├── runtime.py
│   ├── modeling.py
│   ├── grpo.py
│   ├── training.py
│   ├── inference.py
│   ├── execution.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── comparison.py
│   └── data/
│       ├── __init__.py
│       ├── datasets.py
│       ├── render.py
│       └── prepare.py
├── docs/
│   └── design_doc.md
├── tests/
├── prepare_dataset.py
├── train.py
├── resume_train.py
├── infer.py
├── build_meshes.py
├── evaluate.py
└── compare_runs.py
```

Files and packages to remove after moving code:

- `cad_rl/specs/`
- `cad_rl/pipelines/`
- `cad_rl/models/`
- `cad_rl/algorithms/`
- `cad_rl/runtime/`
- `cad_rl/metrics/`

The goal is fewer layers, not bigger confusion. Keep only modules with distinct responsibilities.

## Module Ownership

### `cad_rl.config`

Owns config dataclasses, config loading, overlay merge, and serialization.

Keep the runtime contract small:

- `RunConfig`
- `SystemConfig`

`RunConfig` should contain grouped sections only:

- `task`
- `data`
- `model`
- `train`
- `infer`
- `mesh`
- `eval`
- `runtime`

Do not keep separate component-profile files for task, model, algorithm, trainer, and machine.

### `cad_rl.runtime`

Owns run directories, manifests, checkpoint selection, and record dataclasses.

Keep these runtime records:

- `RunManifest`
- `CheckpointRef`
- `InferenceRecord`
- `MeshRecord`
- `EvalRecord`
- `ComparisonReport`

### `cad_rl.modeling`

Owns the current model-family code only.

Today that means Qwen loading and processor setup. Do not introduce a backend registry until there is a second real model path.

### `cad_rl.grpo`

Owns GRPO-specific trainer internals:

- custom trainer subclass
- custom loss helpers
- top-sample selection logic

Keep this focused on TRL/GRPO behavior only.

### `cad_rl.training`

Owns train and resume orchestration:

- load resolved config
- build processor and model
- build trainer
- build reward function
- materialize run metadata
- start training or resume from checkpoint

### `cad_rl.execution`

Owns CadQuery code execution and mesh materialization:

- execute generated code
- apply timeouts
- manage worker pool
- export meshes
- return structured execution results

CadQuery execution must live here, not inside metric helpers.

### `cad_rl.metrics`

Owns geometry metrics only:

- Chamfer distance
- IoU
- normal-based metrics
- AOC-GMS helpers

This module consumes meshes or execution outputs. It does not execute code.

### `cad_rl.inference`

Owns inference only:

- inference record generation

This module should orchestrate generation using `modeling`, `runtime`, and dataset loaders.

### `cad_rl.evaluation`

Owns mesh-stage evaluation only:

- metric computation over built meshes
- evaluation record generation
- summary generation

This module should orchestrate evaluation using `metrics` and `runtime`.

### `cad_rl.comparison`

Owns comparison only:

- comparison between runs or summaries
- leaderboard/report generation

This module should stay small and consume evaluation summaries only.

### `cad_rl.data`

Owns dataset prep and dataset loading only.

Target submodules:

- `datasets.py`: prepared-dataset manifest/fingerprint and inference dataset loading
- `render.py`: all dataset-prep-only rendering helpers
- `prepare.py`: dataset-prep stage entry logic

This is the only place where optional rendering dependencies are allowed.

## Stage Contracts

Each CLI should read config, call one package function, and write artifacts to disk.

CLI to module mapping:

- `prepare_dataset.py` -> `cad_rl.data.prepare`
- `train.py` and `resume_train.py` -> `cad_rl.training`
- `infer.py` -> `cad_rl.inference`
- `build_meshes.py` -> `cad_rl.execution`
- `evaluate.py` -> `cad_rl.evaluation`
- `compare_runs.py` -> `cad_rl.comparison`

### `prepare_dataset.py`

Input:

- raw dataset source
- run config
- optional system config

Output:

- prepared dataset on disk
- prepared dataset manifest

### `train.py` and `resume_train.py`

Input:

- run config
- optional system config
- prepared dataset reference

Output:

- run directory
- resolved config snapshot
- run manifest
- checkpoints

### `infer.py`

Input:

- run config
- optional system config
- checkpoint reference
- prepared dataset split or inference source

Output:

- `InferenceRecord` JSONL

### `build_meshes.py`

Input:

- `InferenceRecord` JSONL

Output:

- exported meshes
- `MeshRecord` JSONL

### `evaluate.py`

Input:

- `MeshRecord` JSONL

Output:

- `EvalRecord` JSONL
- summary JSON

### `compare_runs.py`

Input:

- evaluation summaries

Output:

- comparison report

## CLI Contract

Every public CLI should use the same shape:

- `--config` required
- `--system` optional
- `--dry-run` optional

Avoid ad hoc workflow flags for things that belong in config.

Examples of flags that should be removed from the long-term design:

- separate task/model/algorithm profile flags
- worker-count flags that change experiment behavior
- direct path wiring between stages
- output variable naming flags

Those belong in config unless they are truly one-off debug overrides.

## Simplification Rules

- one file owns one concern
- remove duplicate task/model config types
- remove package layers that only forward calls
- do not document future seams that do not exist yet
- keep stage boundaries explicit
- keep failure records structured instead of silently dropping bad samples

## Cutover Plan

Implement the new shape as a breaking cleanup:

- replace flat legacy config files with `configs/runs/*.yaml`
- allow an optional overlay from `configs/systems/*.yaml`
- move CadQuery execution out of metric code
- delete `cad_rl.specs`
- fold tiny helper packages into single modules
- keep only the root CLIs and the simplified `cad_rl/` layout

The end state should be simpler to read than the current tree without hiding behavior inside oversized utility files.
