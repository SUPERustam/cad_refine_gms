# CAD RL

Profile-driven CAD RL research repo with a filesystem run registry and a frozen Dr.CPPO/GRPO training core.

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

- `cad_rl/algorithms`: GRPO trainer, CPPO loss, reward wiring
- `cad_rl/metrics`: async metric execution, mesh evaluation helpers, AOC-GMS
- `cad_rl/data`: prepared dataset helpers and dataset-prep rendering path
- `cad_rl/config`, `cad_rl/runtime`, `cad_rl/pipelines`, `cad_rl/specs`, `cad_rl/models`: config resolution, run registry, workflows, schemas, and model adapters

## Config model

Experiment profiles under `configs/` compose five main documents:

- `task`
- `model`
- `algorithm`
- `trainer`
- `machine`

`train.py` and `resume_train.py` resolve the experiment profile, materialize the resolved config into the run directory, and use that resolved contract as the source of truth for resume and downstream analysis.

The config resolver supports `extends`, component references, and nested `overrides`. Current examples include:

- `configs/experiment.demo.yaml`
- `configs/experiment.dr_cppo_constant.yaml`
- `configs/experiment.dr_cppo_cosine.yaml`
- `configs/experiment.dr_cppo_gms.yaml`
- `configs/experiment.dr_cppo_aoc_gms.yaml`

## Quickstart

Prepare a serialized Hugging Face dataset:

```bash
python prepare_dataset.py --split train
```

`prepare_dataset.py` is a thin wrapper over `cad_rl.data`. Its rendering path may require the optional `vis_for_norm_parts` dependency depending on your dataset-prep environment.

Inspect the fully resolved training contract without launching training:

```bash
python train.py --experiment configs/experiment.demo.yaml --dry-run
```

Start a run:

```bash
python train.py --experiment configs/experiment.demo.yaml
```

Resume from the filesystem checkpoint registry:

```bash
python resume_train.py \
  --experiment configs/experiment.demo.yaml \
  --checkpoint latest
```

Run the post-training workflow:

```bash
python infer.py \
  --task-profile configs/task.cadquery_v1.yaml \
  --model-profile configs/model.qwen2_vl.yaml \
  --checkpoint /path/to/checkpoint \
  --output outputs/inference.jsonl

python build_meshes.py \
  --input outputs/inference.jsonl \
  --output-dir outputs/meshes

python evaluate.py \
  --input outputs/inference.jsonl \
  --output outputs/eval.jsonl

python compare_runs.py \
  --summaries outputs/eval.summary.json other_run/eval.summary.json \
  --output outputs/compare.json
```

## Run artifacts

Training materializes a run directory under the machine profile `run_root`, for example `./runs/<run_id>/`, with:

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
