# Troubleshooting

## `resume_train.py` cannot find a checkpoint

If resume fails with a checkpoint lookup error, verify:

- the run directory exists under the machine profile `run_root`
- `checkpoints/index.jsonl` exists for that run, or `checkpoints/latest.txt` points at a real checkpoint
- the `--checkpoint` value is one of `latest`, `best`, a recorded step, or an explicit path

Use `python train.py --experiment ... --dry-run` to confirm you are resolving the expected machine profile and run root.

## Prepared dataset path or split is missing

Training and inference load datasets from `task.prepared_datasets`.

Check:

- the resolved task profile contains the split you requested, for example `train` or `val`
- the on-disk dataset path exists
- the dataset was written with `prepare_dataset.py`

`infer.py --split val` will fail if the task profile has no `val` dataset entry.

## `prepare_dataset.py` fails with a `vis_for_norm_parts` message

Dataset preparation uses an optional rendering dependency that is intentionally isolated from training, inference, evaluation, and runtime code.

If you see an error mentioning `vis_for_norm_parts`:

- install that dependency in the dataset-prep environment
- or run dataset preparation in an environment that already provides it
- do not treat it as a training or inference dependency leak

## vLLM or generation endpoint does not come up cleanly

If generation fails around connection or startup:

- compare `trainer.vllm_server_port` with the machine profile `vllm_port`
- verify any required environment variables from `machine.environment`
- inspect the resolved contract with `--dry-run` before launching the run

The supported interface is profile-driven; do not rely on deleted shell wrappers or legacy launch scripts.

## CadQuery output variable mismatch

If generated code executes but mesh building or evaluation reports invalid code, check the expected output variable name.

The current interfaces use:

- `task.output_var_name` in the task profile
- `--var-name` in `build_meshes.py`
- `--var-name` in `evaluate.py`

These must agree with the variable assigned by the generated CadQuery program. The default task profile uses `result`.

## Import errors after local refactors

This repo no longer supports importing former root modules such as metric helpers, trainer internals, dataset helpers, or small utility modules directly from the repo root.

If local code still imports names like `metrics_async`, `grpo_trainer`, `grpo_loss`, `rewards`, `multiview_dataset`, `helper_visu`, or `utils`, update those imports to `cad_rl...` package paths instead.

## OOM or unstable training

If training runs out of memory or becomes unstable:

- reduce `trainer.per_device_train_batch_size`
- reduce `trainer.max_completion_length`
- reduce `trainer.num_generations` or `trainer.generation_batch_size`
- keep `gradient_checkpointing` enabled unless you have a reason to disable it
- verify there is enough free disk space under the run root and cache directories

## Zero loss or zero useful reward signal

Common causes:

- dataset rows point at bad mesh paths
- nearly all generations fail execution and only receive `failure_reward`
- the checkpoint is too weak for the current task
- reward coefficients do not match the intended profile variant

Inspect the generated inference JSONL and evaluation outputs before changing training code.

## Evaluation output looks incomplete

`evaluate.py` writes two files:

- the per-sample rows at the path passed to `--output`
- the aggregate summary at the same path with suffix `.summary.json`

`compare_runs.py` expects summary JSON files, not the raw per-sample JSONL.
