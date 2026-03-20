# Reinforcement Learning Practical Guide

This repo exposes a clean Python script surface. Training, resume, and downstream analysis are driven by resolved profiles and filesystem artifacts, not by ad hoc shell wrappers.

## Before you run anything

Follow [Setup](Setup.md) first:

- prepare the HF dataset on disk
- confirm the `task`, `model`, `algorithm`, `trainer`, and `machine` profiles you want
- inspect the resolved contract with `--dry-run` before launching long jobs

## Training

Launch a run from an experiment profile:

```bash
python train.py --experiment configs/experiment.demo.yaml
```

Inspect the resolved contract without starting training:

```bash
python train.py --experiment configs/experiment.demo.yaml --dry-run
```

The training pipeline resolves the profile, fingerprints the prepared training dataset when possible, creates a run directory under the machine profile `run_root`, and writes:

- `resolved_config.json`
- `manifest.json`
- `checkpoints/`
- `artifacts/`
- `logs/`

If `PyYAML` is available, YAML mirrors are written next to the JSON files.

## Resume

Resume uses the filesystem run registry instead of shell logic:

```bash
python resume_train.py \
  --experiment configs/experiment.demo.yaml \
  --checkpoint latest
```

Supported checkpoint references:

- `latest`
- `best`
- a numeric step such as `150`
- an explicit checkpoint path

`latest` is resolved from `checkpoints/latest.txt` when present, otherwise from the checkpoint index.

## Important config fields

The training contract comes from the resolved experiment profile. The current code paths depend most on:

| Field | Meaning |
| :--- | :--- |
| `task.prepared_datasets` | On-disk HF dataset paths by split |
| `task.output_var_name` | Expected CadQuery output variable |
| `model.base_checkpoint` | Base or SFT checkpoint family |
| `model.processor_kwargs` | Qwen processor settings |
| `algorithm.reward_config` | Reward coefficients and failure behavior |
| `algorithm.trainer_kwargs.top_samples` | Top-sample CPPO selection size |
| `algorithm.scheduler_policy` | Current scheduler variant, for example `constant` or `cosine` |
| `trainer.num_generations` | Number of completions per prompt |
| `trainer.max_completion_length` | Completion token cap |
| `machine.run_root` | Base directory for run artifacts |
| `machine.environment` | Environment variables injected before trainer setup |

## Inference

Inference currently loads a task profile and model profile directly:

```bash
python infer.py \
  --task-profile configs/task.cadquery_v1.yaml \
  --model-profile configs/model.qwen2_vl.yaml \
  --checkpoint /path/to/checkpoint \
  --split val \
  --output outputs/inference.jsonl
```

Useful flags:

- `--batch-size`
- `--num-workers`

The output is JSONL. Each row contains the task id, source mesh path, output variable name, raw model generation, wrapped code, timing, and execution placeholder metadata.

## Mesh building

Materialize meshes from inference JSONL:

```bash
python build_meshes.py \
  --input outputs/inference.jsonl \
  --output-dir outputs/meshes \
  --var-name result
```

The mesh pipeline writes:

- `outputs/meshes/mesh_records.jsonl`
- `outputs/meshes/meshes/sample_*.stl`

Rows are marked `success` or `invalid_code`.

## Evaluation

Evaluate raw generations against the source mesh paths:

```bash
python evaluate.py \
  --input outputs/inference.jsonl \
  --output outputs/eval.jsonl \
  --var-name result \
  --pool-size 16
```

The evaluation pipeline writes:

- per-sample rows to the path passed to `--output`
- an aggregate summary next to it at `*.summary.json`

Current summary fields include:

- `samples`
- `invalid_fraction`
- `iou_mean`, `iou_median`, `iou_min`, `iou_max`
- `cd_mean`, `cd_median`, `cd_min`, `cd_max`
- `missing_sample_count`

## Comparison

Compare one or more summary files:

```bash
python compare_runs.py \
  --summaries outputs/eval.summary.json other_run/eval.summary.json \
  --output outputs/compare.json
```

The report contains:

- `leaderboard`
- `checkpoint_over_time`
- `diff_report`

## Monitoring

Training writes the authoritative run state to disk. Comet is a mirror layer when enabled, not the control plane for resume.

The most useful signals to watch in training logs remain:

- reward trend
- reward variance collapse
- entropy collapse
- clipping behavior
- completion length drift

Use [Troubleshooting](Troubleshooting.md) when a run resolves cleanly but fails at data loading, checkpoint selection, generation, or metric execution time.
