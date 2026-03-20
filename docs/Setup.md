# Setup

This repo is driven by profile documents under `configs/` and writes run state to the filesystem. The codebase currently assumes a local machine profile similar to `configs/machine.local.yaml`.

## Environment variables

No `.env` file is required for the basic local workflow.

Set environment variables only for integrations you use:

```bash
export HF_TOKEN=...
export COMET_API_KEY=...
export COMET_PROJECT_NAME=...
export COMET_WORKSPACE=...
```

`configs/machine.local.yaml` currently keeps Comet disabled, so the Comet variables are optional unless you enable that integration in your machine or runtime config.

## Expected local directories

The default local machine profile points at:

- `./runs` for run directories
- `./.cache/cad_rl` for cacheable artifacts

Training creates run-local subdirectories for `checkpoints/`, `artifacts/`, and `logs/` automatically.

## Profile layout

Experiment profiles compose five component configs:

- `task`
- `model`
- `algorithm`
- `trainer`
- `machine`

The resolver supports:

- `extends` chains for layered profiles
- component references such as `task: "task.cadquery_v1.yaml"`
- nested `overrides` blocks
- `runtime` values that are carried into the resolved run contract

The main entrypoint examples are:

- `configs/experiment.demo.yaml`
- `configs/experiment.dr_cppo_constant.yaml`
- `configs/experiment.dr_cppo_cosine.yaml`
- `configs/experiment.dr_cppo_gms.yaml`
- `configs/experiment.dr_cppo_aoc_gms.yaml`

To inspect the fully resolved training contract:

```bash
python train.py --experiment configs/experiment.demo.yaml --dry-run
```

## Dataset preparation

Training and inference expect prepared Hugging Face datasets on disk. The current task profile points both `train` and `val` at:

```text
datasets/rendered_cadevolve_normalized_1_1_fixed
```

Create a prepared dataset with the supported flags in `prepare_dataset.py`:

```bash
python prepare_dataset.py \
  --split train \
  --output-dir datasets/rendered_cadevolve_normalized_1_1_fixed
```

Useful optional flags:

- `--raw-root` to point at the STL source root
- `--pickle-file` to provide annotations
- `--size` to cap the sample count
- `--shuffle` to shuffle source examples
- `--task-id` and `--model-id` to override manifest metadata

The script writes a serialized HF dataset plus `manifest.json` into the output directory.

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
python train.py --experiment configs/experiment.demo.yaml
```

Resume from the registry-managed checkpoint set:

```bash
python resume_train.py \
  --experiment configs/experiment.demo.yaml \
  --checkpoint latest
```

`--checkpoint` accepts `latest`, `best`, a numeric step, or an explicit checkpoint path.

## Related docs

- [RL Practical Guide](RL_Practical.md)
- [RL Theory](RL_Theory.md)
- [Troubleshooting](Troubleshooting.md)
