# CAD RL

Config-driven CAD RL research repo with a filesystem run registry and a flat stage-module layout under `cad_rl/`.

## Supported scripts

- `prepare_dataset.py`
- `train.py`
- `resume_train.py`
- `infer.py`
- `build_meshes.py`
- `evaluate.py`
- `compare_runs.py`

Everything else is internal package code under `cad_rl/`. Root-level reusable modules are no longer part of the supported interface.

## Package layout

The repo now follows one boundary rule:

- thin public CLI entrypoints stay in the repo root
- reusable Python code lives under `cad_rl/`

Current internal layout:

- `cad_rl/training`, `cad_rl/inference`, `cad_rl/execution`, `cad_rl/evaluation`, `cad_rl/comparison`: stage ownership
- `cad_rl/modeling`, `cad_rl/grpo`, `cad_rl/config`, `cad_rl/runtime`, `cad_rl/metrics`: shared runtime/model/training internals
- `cad_rl/data`: prepared dataset helpers and dataset-prep rendering path

## Config model

The active workflow is stage-config based:

- `configs/demo/common.yaml`
- `configs/demo/prepare_dataset.yaml`
- `configs/demo/train.yaml`
- `configs/demo/infer.yaml`
- `configs/demo/build_meshes.yaml`
- `configs/demo/evaluate.yaml`
- `configs/demo/compare.yaml`
- optional `configs/systems/*.yaml`

The resolved config is grouped by stage:

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

## Quickstart

Prepare a serialized Hugging Face dataset:

```bash
python prepare_dataset.py --config configs/demo/prepare_dataset.yaml --dry-run
```

`prepare_dataset.py` is a thin wrapper over `cad_rl.data`. Its rendering path may require the optional `vis_for_norm_parts` dependency depending on your dataset-prep environment.

Inspect the fully resolved training contract without launching training:

```bash
python train.py --config configs/demo/train.yaml --dry-run
```

Start a run:

```bash
python train.py --config configs/demo/train.yaml
```

Resume from the filesystem checkpoint registry:

```bash
python resume_train.py \
  --config configs/demo/train.yaml \
  --checkpoint latest
```

Run the post-training workflow:

```bash
python infer.py --config configs/demo/infer.yaml
python build_meshes.py --config configs/demo/build_meshes.yaml
python evaluate.py --config configs/demo/evaluate.yaml
python compare_runs.py --config configs/demo/compare.yaml
```

## Run artifacts

Training materializes a run directory under the configured `system.run_root`, for example `./runs/<run_id>/`, with:

- `resolved_config.json`
- `manifest.json`
- `checkpoints/`
- `artifacts/`
- `logs/`

Optional YAML mirrors are also written when `PyYAML` is available.

## Docs

- `docs/Setup.md`: environment, profiles, and dataset prep.
- `docs/RL_Practical.md`: training, resume, inference, and artifact flow.
- `docs/RL_Theory.md`: frozen algorithm core and config variants.
- `docs/Troubleshooting.md`: current repo-scoped failure modes.
