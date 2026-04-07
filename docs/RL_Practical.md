# Reinforcement Learning Practical Guide

This repo exposes a clean Python script surface. Training, resume, and downstream analysis are driven by resolved stage configs and filesystem artifacts, not by ad hoc shell wrappers.

## Before you run anything

Follow [Setup](Setup.md) first:

- prepare the HF dataset on disk
- confirm the resolved stage config you want
- inspect the resolved contract with `--dry-run` before launching long jobs

## Training

Launch a run from a stage config:

```bash
python cli.py train --config configs/demo/train.yaml
```

Inspect the resolved contract without starting training:

```bash
python cli.py train --config configs/demo/train.yaml --dry-run
```

The training pipeline resolves the stage config, fingerprints the prepared training dataset when possible, creates a run directory under the configured `system.run_root`, and writes:

- `resolved_config.json`
- `manifest.json`
- `checkpoints/`
- `artifacts/`
- `logs/`

Run-scoped logging is initialized before stage execution. Use `--debug` to raise console verbosity for a single command.

## Resume

Resume uses the filesystem run registry instead of shell logic:

```bash
python cli.py resume-train \
  --config configs/demo/train.yaml \
  --checkpoint latest
```

Supported checkpoint references:

- `latest`
- `best`
- a numeric step such as `150`
- an explicit checkpoint path

`latest` is resolved from `checkpoints/latest.txt` when present, otherwise from the checkpoint index.

## Important config fields

The training contract comes from the resolved stage config. The current code paths depend most on:

| Field | Meaning |
| :--- | :--- |
| `data.prepared_datasets` | On-disk HF dataset paths by split |
| `task.output_var_name` | Expected CadQuery output variable |
| `model.base_checkpoint` | Base or SFT checkpoint family |
| `model.processor_kwargs` | Qwen processor settings |
| `train.reward` | Reward coefficients and failure behavior |
| `train.top_samples` | Top-sample CPPO selection size |
| `train.scheduler` | Current scheduler variant, for example `constant` or `cosine` |
| `train.num_generations` | Number of completions per prompt |
| `train.max_completion_length` | Completion token cap |
| `system.run_root` | Base directory for run artifacts |
| `system.environment` | Environment variables injected before trainer setup |

## Inference

Inference is stage-config driven and resolves symbolic checkpoints through the runtime registry:

```bash
python cli.py infer --config configs/demo/infer.yaml
```

The output is JSONL. Each row contains a runtime inference record including the sample id, resolved checkpoint reference, raw generation, wrapped code, timing, and metadata.

## Mesh building

Materialize meshes from inference JSONL:

```bash
python cli.py build-meshes --config configs/demo/build_meshes.yaml
```

The mesh pipeline writes:

- `outputs/meshes/mesh_records.jsonl`
- `outputs/meshes/meshes/sample_*.stl`

Rows are marked `ok` or `invalid`.

## Evaluation

Evaluate built mesh artifacts:

```bash
python cli.py evaluate --config configs/demo/evaluate.yaml
```

The evaluation pipeline writes:

- per-sample rows to `eval.output_path`
- an aggregate summary next to it at `*.summary.json`

Current summary fields include:

- `run_id`
- `checkpoint_path`
- `checkpoint_step`
- `samples`
- `invalid_fraction`
- `iou_mean`, `iou_median`, `iou_min`, `iou_max`
- `cd_mean`, `cd_median`, `cd_min`, `cd_max`
- `missing_sample_count`

## Comparison

Compare one or more summary files through the stage config:

```bash
python cli.py compare-runs --config configs/demo/compare.yaml
```

The report contains:

- `leaderboard`
- `checkpoint_history`
- `diff_report`
- `summary`

## Monitoring

Training writes the authoritative run state to disk. Comet is a mirror layer when enabled, not the control plane for resume.

The most useful signals to watch in training logs remain:

- reward trend
- reward variance collapse
- entropy collapse
- clipping behavior
- completion length drift

Use [Troubleshooting](Troubleshooting.md) when a run resolves cleanly but fails at data loading, checkpoint selection, generation, or metric execution time.
