# Setup

This repo is driven by stage config documents under `configs/` and writes run state to the filesystem. The codebase currently assumes a local system config similar to `configs/systems/local.yaml`.

## Environment variables

No `.env` file is required for the basic local workflow.

Set environment variables only for integrations you use:

```bash
export HF_TOKEN=...
export COMET_API_KEY=...
export COMET_PROJECT_NAME=...
export COMET_WORKSPACE=...
```

`configs/systems/local.yaml` currently keeps Comet disabled, so the Comet variables are optional unless you enable that integration in your system or runtime config.

## Expected local directories

The default local system config points at:

- `./runs` for run directories
- `./.cache/cad_rl` for cacheable artifacts

Training creates run-local subdirectories for `checkpoints/`, `artifacts/`, and `logs/` automatically.

## Config layout

Experiment directories compose one shared file plus stage overlays:

- `configs/demo/common.yaml`
- `configs/demo/prepare_dataset.yaml`
- `configs/demo/train.yaml`
- `configs/demo/infer.yaml`
- `configs/demo/build_meshes.yaml`
- `configs/demo/evaluate.yaml`
- `configs/demo/compare.yaml`
- optional `configs/systems/*.yaml`

To inspect the fully resolved training contract:

```bash
python train.py --config configs/demo/train.yaml --dry-run
```

## Dataset preparation

Training and inference expect prepared Hugging Face datasets on disk. The current demo task config points both `train` and `val` at:

```text
datasets/rendered_cadevolve_normalized_1_1_fixed
```

Create a prepared dataset from the stage config:

```bash
python prepare_dataset.py --config configs/demo/prepare_dataset.yaml
```

The script writes a serialized HF dataset plus `manifest.json` into the configured output directory. Dataset source values come from the resolved `prepare` and `data` sections.

Dataset preparation is the only part of the repo that depends on the STL visualization helper path under `cad_rl.data`. If your environment does not provide `vis_for_norm_parts`, `prepare_dataset.py` now fails with a targeted error explaining that the dependency is optional and dataset-prep-specific.

## Repo boundary

The supported root Python files are intentionally limited to:

- `prepare_dataset.py`
- `train.py`
- `resume_train.py`
- `infer.py`
- `build_meshes.py`
- `evaluate.py`
- `compare_runs.py`

Reusable imports should come from `cad_rl/...`, not from root-level modules or the current working directory.

## Training entrypoints

Start a run:

```bash
python train.py --config configs/demo/train.yaml
```

Resume from the registry-managed checkpoint set:

```bash
python resume_train.py \
  --config configs/demo/train.yaml \
  --checkpoint latest
```

`--checkpoint` accepts `latest`, `best`, a numeric step, or an explicit checkpoint path.

## Related docs

- [RL Practical Guide](RL_Practical.md)
- [RL Theory](RL_Theory.md)
- [Troubleshooting](Troubleshooting.md)
