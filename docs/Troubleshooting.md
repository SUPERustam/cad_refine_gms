# Troubleshooting

## `cli.py resume-train` cannot find a checkpoint

If resume fails with a checkpoint lookup error, verify:

- the run directory exists under the configured `system.run_root`
- `checkpoints/index.jsonl` exists for that run, or `checkpoints/latest.txt` points at a real checkpoint
- the `--checkpoint` value is one of `latest`, `best`, a recorded step, or an explicit path

Use `python cli.py train --config configs/demo/train.yaml --dry-run` to confirm you are resolving the expected system config and run root.

## Prepared dataset path or split is missing

Training and inference load datasets from `data.prepared_datasets`.

Check:

- the resolved task config contains the split you requested, for example `train` or `val`
- the on-disk dataset path exists
- the dataset was written with `cli.py prepare-dataset`

Inference will fail if `infer.split` points at a split that is missing from `data.prepared_datasets`.

## `cli.py prepare-dataset` fails with a `vis_for_norm_parts` message

Dataset preparation uses an optional rendering dependency that is intentionally isolated from training, inference, evaluation, and runtime code.

If you see an error mentioning `vis_for_norm_parts`:

- install that dependency in the dataset-prep environment
- or run dataset preparation in an environment that already provides it
- do not treat it as a training or inference dependency leak

## vLLM or generation endpoint does not come up cleanly

If generation fails around connection or startup:

- verify `train.vllm_server_port` matches the port your generation service is actually using
- verify any required environment variables from `system.environment`
- inspect the resolved contract with `--dry-run` before launching the run

The supported interface is stage-config driven; do not rely on deleted shell wrappers or legacy launch scripts.

## CadQuery output variable mismatch

If generated code executes but mesh building or evaluation reports invalid code, check the expected output variable name.

The current interfaces use:

- `task.output_var_name` in the resolved config

Mesh building and evaluation read that value from the resolved config. It must agree with the variable assigned by the generated CadQuery program. The default demo config uses `result`.

## Import errors after local refactors

This repo no longer supports importing former root modules such as metric helpers, trainer internals, dataset helpers, or small utility modules directly from the repo root.

If local code still imports names like `metrics_async`, `grpo_trainer`, `grpo_loss`, `rewards`, `multiview_dataset`, `helper_visu`, or `utils`, update those imports to `cad_rl...` package paths instead.

## OOM or unstable training

If training runs out of memory or becomes unstable:

- reduce `train.per_device_train_batch_size`
- reduce `train.max_completion_length`
- reduce `train.num_generations` or `train.generation_batch_size`
- keep `gradient_checkpointing` enabled unless you have a reason to disable it
- verify there is enough free disk space under the run root and cache directories

## Zero loss or zero useful reward signal

Common causes:

- dataset rows point at bad mesh paths
- nearly all generations fail execution and only receive `failure_reward`
- the checkpoint is too weak for the current task
- reward coefficients do not match the intended config variant

Inspect the generated inference JSONL and evaluation outputs before changing training code.

## Evaluation output looks incomplete

`cli.py evaluate` writes two files based on the resolved `eval.output_path`:

- the per-sample rows at the configured output path
- the aggregate summary at the same path with suffix `.summary.json`

`cli.py compare-runs` reads the summary JSON paths configured in `compare.summaries`, not the raw per-sample JSONL.

## Dry-run or CLI output does not match old config docs

The current contract is stricter than the older docs:

- each command resolves one self-contained stage YAML
- there is no implicit `common.yaml`
- there is no CLI `--system` overlay
- training fields live directly under `train`, with reward settings under `train.reward`

If your local config still uses old keys such as `train.reward_config`, `train.trainer_kwargs`, or `train.scheduler_policy`, update the YAML to the current schema before running the stage.
