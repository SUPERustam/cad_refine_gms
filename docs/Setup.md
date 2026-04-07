# Setup

This repo is driven by self-contained stage config documents under `configs/` and writes run state to the filesystem. Each command resolves one YAML file that already includes its `system` and `runtime` sections.

## Environment variables

No `.env` file is required for the basic local workflow.

Set environment variables only for integrations you use:

```bash
export HF_TOKEN=...
export COMET_API_KEY=...
export COMET_PROJECT_NAME=...
export COMET_WORKSPACE=...
```

The checked-in demo configs keep Comet disabled, so the Comet variables are optional unless you enable that integration in the stage config you are running.

## Expected local directories

The checked-in demo configs point at:

- `./runs` for run directories
- `./.cache/cad_rl` for cacheable artifacts

Training creates run-local subdirectories for `checkpoints/`, `artifacts/`, and `logs/` automatically.

## Config layout

Each stage config is standalone:

- `configs/demo/prepare_dataset.yaml`
- `configs/demo/train.yaml`
- `configs/demo/infer.yaml`
- `configs/demo/build_meshes.yaml`
- `configs/demo/evaluate.yaml`
- `configs/demo/compare.yaml`

To inspect the fully resolved training contract:

```bash
python cli.py train --config configs/demo/train.yaml --dry-run
```

To force debug logging for a single invocation without editing YAML:

```bash
python cli.py train --config configs/demo/train.yaml --debug --dry-run
```

## Dataset preparation

Training and inference expect prepared Hugging Face datasets on disk. The current demo task config points both `train` and `val` at:

```text
datasets/rendered_cadevolve_normalized_1_1_fixed
```

Create a prepared dataset from the stage config:

```bash
python cli.py prepare-dataset --config configs/demo/prepare_dataset.yaml
```

The command writes a serialized HF dataset plus `manifest.json` into the configured output directory. Dataset source values come from the resolved `prepare` and `data` sections.

Dataset preparation is the only part of the repo that depends on the STL visualization helper path under `cad_rl.data`. If your environment does not provide `vis_for_norm_parts`, `cli.py prepare-dataset` now fails with a targeted error explaining that the dependency is optional and dataset-prep-specific.

## Repo boundary

The supported root Python interface is intentionally limited to:

- `cli.py`

Supported workflow subcommands:

- `prepare-dataset`
- `train`
- `resume-train`
- `infer`
- `build-meshes`
- `evaluate`
- `compare-runs`

Reusable imports should come from `cad_rl/...`, not from root-level modules or the current working directory.

## Training entrypoints

Start a run:

```bash
python cli.py train --config configs/demo/train.yaml
```

Resume from the registry-managed checkpoint set:

```bash
python cli.py resume-train \
  --config configs/demo/train.yaml \
  --checkpoint latest
```

`--checkpoint` accepts `latest`, `best`, a numeric step, or an explicit checkpoint path.

All runtime-stage commands also initialize run-scoped logging under `runs/<run_id>/logs/`. Use `--debug` when you want debug-level console output and debug-level file logs for that invocation.

## Related docs

- [RL Practical Guide](RL_Practical.md)
- [RL Theory](RL_Theory.md)
- [Troubleshooting](Troubleshooting.md)
