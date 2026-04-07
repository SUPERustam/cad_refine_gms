# KISS Design for `cad_refine_gms`

## Goal

__Keep the project easy to change and easy to debug.__

The repo has one main workflow and one module per stage:

`prepare_dataset -> train/resume -> infer -> build_meshes -> evaluate -> compare`

The design should optimize for:

- one obvious place for each responsibility
- one obvious module for each stage
- one thin public CLI at the repo root
- config-driven runs
- simple stage handoffs on disk
- no speculative abstraction layers
- no wrapper packages that only forward calls

If there is only one implementation today, keep one implementation today. Add a new interface only when a second real implementation requires it.

## Project Rules

- Root Python files are public CLIs only:
  `cli.py`
- Reusable code lives under `cad_rl/`
- Use flat stage modules under `cad_rl/`: `training`, `inference`, `execution`, `evaluation`, `comparison`
- `cad_rl.data` is the only place allowed to contain dataset-prep rendering code
- optional `vis_for_norm_parts` support must stay inside `cad_rl.data`
- training, inference, evaluation, and runtime code must not depend on dataset-prep-only rendering helpers
- The root CLI stays thin and delegates into those package modules
- Keep one active config model and one active runtime record model
- Do not keep duplicate package layers such as separate `pipelines`, `specs`, or tiny forwarding packages
- filesystem artifacts are the source of truth for stage outputs and resume state

## Target Repo Shape

```text
cad_refine_gms/
├── configs/
│   ├── demo/
│   │   ├── prepare_dataset.yaml
│   │   ├── train.yaml
│   │   ├── infer.yaml
│   │   ├── build_meshes.yaml
│   │   ├── evaluate.yaml
│   │   └── compare.yaml
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
├── cli.py
└── README.md
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

Owns config dataclasses, config loading, validation, and serialization.

Keep the runtime contract small:

- `RunConfig`
- `SystemConfig`

The active config tree should use one self-contained stage file per command invocation:

- `configs/<experiment>/prepare_dataset.yaml`
- `configs/<experiment>/train.yaml`
- `configs/<experiment>/infer.yaml`
- `configs/<experiment>/build_meshes.yaml`
- `configs/<experiment>/evaluate.yaml`
- `configs/<experiment>/compare.yaml`

Do not rely on implicit `common.yaml` merges, `system_profile`, or a separate CLI `--system` overlay.

`RunConfig` should contain grouped sections only:

- `task`
- `data`
- `model`
- `prepare`
- `train`
- `infer`
- `mesh`
- `eval`
- `compare`
- `runtime`

Do not keep separate component-profile files for task, model, algorithm, trainer, and machine.

### `cad_rl.runtime`

Owns run directories, manifests, checkpoint selection, run-scoped logging, and record dataclasses.

Keep these runtime records:

- `RunManifest`
- `CheckpointRef`
- `InferenceRecord`
- `MeshRecord`
- `EvalRecord`
- `ComparisonReport`

Keep runtime as the single source of truth for:

- `resolved_config.json`
- `manifest.json`
- checkpoint index and `latest.txt`
- `logs/<stage>.log`

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

Thin wrappers that only rename fields or forward dataclasses should be folded back into the stage entrypoints.

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
- runtime-backed checkpoint resolution for symbolic names such as `latest` and `best`

This module should orchestrate generation using `modeling`, `runtime`, and dataset loaders.

### `cad_rl.evaluation`

Owns mesh-stage evaluation only:

- metric computation over built meshes
- evaluation record generation
- summary generation

Summaries should carry explicit run and checkpoint identity so downstream comparison does not need to infer it from file paths.

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

Each CLI subcommand should read config, call one package function, and write artifacts to disk.

CLI to module mapping:

- `cli.py prepare-dataset` -> `cad_rl.data.prepare`
- `cli.py train` and `cli.py resume-train` -> `cad_rl.training`
- `cli.py infer` -> `cad_rl.inference`
- `cli.py build-meshes` -> `cad_rl.execution`
- `cli.py evaluate` -> `cad_rl.evaluation`
- `cli.py compare-runs` -> `cad_rl.comparison`

### `cli.py prepare-dataset`

Input:

- raw dataset source
- run config

Output:

- prepared dataset on disk
- prepared dataset manifest

### `cli.py train` and `cli.py resume-train`

Input:

- run config
- prepared dataset reference

Output:

- run directory
- resolved config snapshot
- run manifest
- checkpoints

### `cli.py infer`

Input:

- run config
- checkpoint reference
- prepared dataset split or inference source

Output:

- `InferenceRecord` JSONL

### `cli.py build-meshes`

Input:

- `InferenceRecord` JSONL

Output:

- exported meshes
- `MeshRecord` JSONL

### `cli.py evaluate`

Input:

- `MeshRecord` JSONL

Output:

- `EvalRecord` JSONL
- summary JSON

### `cli.py compare-runs`

Input:

- evaluation summaries

Output:

- comparison report

## CLI Contract

Every public CLI subcommand should use the same shape:

- `--config` required
- `--dry-run` optional
- `--debug` optional

Resume also keeps:

- `--checkpoint` optional

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