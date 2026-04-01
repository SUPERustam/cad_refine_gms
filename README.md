# CAD RL

Config-driven CAD RL research repo with a filesystem run registry and a flat stage-module layout under `cad_rl/`.

## Supported CLI

- `cli.py prepare-dataset`
- `cli.py train`
- `cli.py resume-train`
- `cli.py infer`
- `cli.py build-meshes`
- `cli.py evaluate`
- `cli.py compare-runs`

Everything else is internal package code under `cad_rl/`. The repo root keeps one public Python entrypoint only: `cli.py`.

## Package layout

The repo now follows one boundary rule:

- one thin public CLI entrypoint stays in the repo root
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
python cli.py prepare-dataset --config configs/demo/prepare_dataset.yaml --dry-run
```

`cli.py prepare-dataset` is a thin wrapper over `cad_rl.data`. Its rendering path may require the optional `vis_for_norm_parts` dependency depending on your dataset-prep environment.

Inspect the fully resolved training contract without launching training:

```bash
python cli.py train --config configs/demo/train.yaml --dry-run
```

Start a run:

```bash
python cli.py train --config configs/demo/train.yaml
```

Resume from the filesystem checkpoint registry:

```bash
python cli.py resume-train \
  --config configs/demo/train.yaml \
  --checkpoint latest
```

Run the post-training workflow:

```bash
python cli.py infer --config configs/demo/infer.yaml
python cli.py build-meshes --config configs/demo/build_meshes.yaml
python cli.py evaluate --config configs/demo/evaluate.yaml
python cli.py compare-runs --config configs/demo/compare.yaml
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
