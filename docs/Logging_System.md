# Logging System Guide

This document explains how the RL logging system is structured, how to debug failures quickly, and what future agents must preserve when modifying it.

## What Exists

The repo now logs at three layers:

1. Shell lifecycle
   Files: `train_loop_dp_*.sh`, `shell_logging.sh`
   Captures vLLM start, training launch, exit codes, inferred signals, checkpoint selection, cleanup, and restart timing.

2. Python training lifecycle
   Files: `rl_train.py`, `rl_train_cos_sched.py`, `grpo_trainer.py`
   Captures startup config, dataset selection, resume checkpoint, rollout start/end, vLLM generation timing, and uncaught exceptions/signals.

3. Reward worker / CadQuery execution
   Files: `metrics_async.py`, `rewards.py`, `utils.py`
   Captures per-sample CadQuery failures, timeout/crash states, metric timing, failure payloads, and sampled generated code.

Comet ML remains enabled for aggregate dashboards. Local logs are the source of exact causal detail.

## Output Files

For a run named `RUN_NAME`, expect:

- `logs/RUN_NAME.log`
  Human-readable console/training log.
- `logs/RUN_NAME.jsonl`
  Structured events from shell + Python code.
- `logs/RUN_NAME.failures.jsonl`
  Failure payloads with completion text, mesh path, metrics/error payload, and correlation fields.
- `logs/vllm_server.log`
  Raw vLLM server output.

## Core Correlation Fields

When adding or reviewing logs, keep these fields consistent:

- `run_name`
- `session_id`
- `job_id`
- `hostname`
- `pid`
- `rank`
- `local_rank`
- `global_step`
- `mesh_path`
- `checkpoint`
- `event`
- `status`
- `duration_ms`

If a new log line cannot be tied back to a specific run, step, or phase, it is not sufficient.

## Fast Debugging Workflow

### 1. Find why a run exited

```sh
tail -n 50 logs/<RUN_NAME>.jsonl
rg '"event": "train_exit"|"event": "script_exit"|"event": "shell_error"' logs/<RUN_NAME>.jsonl
```

If exit code is `>=128`, the shell wrapper also logs `exit_signal`. Example: exit code `135` usually maps to signal `7` (`BUS` on Linux).

### 2. Find CadQuery failures

```sh
rg 'cadquery_execution_failed|reward_sample_failure|metrics_sample_non_ok' logs/<RUN_NAME>.jsonl logs/<RUN_NAME>.failures.jsonl
```

Then inspect:

- `global_step`
- `mesh_path`
- `sample_idx` / `generation_idx`
- serialized exception payload
- `completion` snippet in `*.failures.jsonl`

### 3. Find worker slowdowns / hangs

```sh
rg '"status": "timeout"|"status": "crash"|worker_timeout|worker_crash' logs/<RUN_NAME>.jsonl logs/<RUN_NAME>.failures.jsonl
```

Check `timings.total_ms` and nearby rollout events in the main JSONL stream.

### 4. Correlate with vLLM issues

```sh
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 35fc70d (resolve conflicts with rl_gms 2)
tail -f logs/vllm_server.log
rg 'vllm_starting|vllm_started|vllm_generation_complete|train_exit' logs/<RUN_NAME>.jsonl
```

If generation stalls, compare shell timestamps with the vLLM server log.

## Maintenance Rules

Future agents should preserve these behaviors:

1. Do not replace structured events with plain `print(...)` for failures or lifecycle transitions.
2. Do not add broad `except: pass` blocks in training or reward execution paths.
3. Any new failure path must log:
   `event`, `status`, `phase`, relevant identifiers, and serialized exception details.
4. Any new long-running stage should emit timing with `duration_ms`.
5. Any new per-sample failure should be eligible for failure-payload logging.
6. Shell wrappers must keep logging around:
   vLLM spawn, accelerate launch, train exit, cleanup, checkpoint selection.
7. Keep Comet enabled unless the user explicitly requests removal.

## When Editing Logging

If you touch any of these files:

- `logging_utils.py`
- `shell_logging.sh`
- `metrics_async.py`
- `rewards.py`
- `grpo_trainer.py`
- `rl_train.py`
- `rl_train_cos_sched.py`
- `train_loop_dp_*.sh`

then also do the following:

1. Re-run Python syntax checks:
   `python -m py_compile logging_utils.py rl_train.py rl_train_cos_sched.py grpo_trainer.py rewards.py metrics_async.py utils.py`
2. Re-run shell syntax checks:
   `bash -n train_loop_dp_gms.sh`
   `bash -n train_loop_dp_gms_resume.sh`
   `bash -n train_loop_dp_gms_resume_2.sh`
   `bash -n train_loop_dp_rustam.sh`
   `bash -n shell_logging.sh`
3. If available, add or update a focused test in `tests/`.
4. Update this document if the event schema, file layout, or debug workflow changes.

## Recommended Queries

### Recent failure payloads

```sh
<<<<<<< HEAD
<<<<<<< HEAD
tail -n 20 logs/<RUN_NAME>.failures.jsonl
=======
tail -n 20 logs_rl/<RUN_NAME>.failures.jsonl
>>>>>>> 9973a20 (resolve conflicts with rl_gms)
=======
tail -n 20 logs/<RUN_NAME>.failures.jsonl
>>>>>>> 35fc70d (resolve conflicts with rl_gms 2)
```

### Count failure types

```sh
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
<<<<<<< HEAD
<<<<<<< HEAD
path = Path("logs/<RUN_NAME>.failures.jsonl")
=======
path = Path("logs_rl/<RUN_NAME>.failures.jsonl")
>>>>>>> 9973a20 (resolve conflicts with rl_gms)
=======
path = Path("logs/<RUN_NAME>.failures.jsonl")
>>>>>>> 35fc70d (resolve conflicts with rl_gms 2)
counts = Counter()
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        counts[json.loads(line)["status"]] += 1
print(counts)
PY
```

### Inspect a single step

```sh
<<<<<<< HEAD
<<<<<<< HEAD
rg '"global_step": 68020' logs/<RUN_NAME>.jsonl logs/<RUN_NAME>.failures.jsonl
=======
rg '"global_step": 68020' logs_rl/<RUN_NAME>.jsonl logs_rl/<RUN_NAME>.failures.jsonl
>>>>>>> 9973a20 (resolve conflicts with rl_gms)
=======
rg '"global_step": 68020' logs/<RUN_NAME>.jsonl logs/<RUN_NAME>.failures.jsonl
>>>>>>> 35fc70d (resolve conflicts with rl_gms 2)
```

## Config Knobs

The main logging-related config fields are:

- `log_dir`
- `structured_logging`
- `failure_payload_logging`
- `sample_payload_logging_steps`
- `max_logged_completion_chars`
- `slow_reward_threshold_sec`
- `slow_step_threshold_sec`

Keep defaults conservative enough for long Slurm jobs. If increasing payload volume, explain the storage tradeoff in the config change.
