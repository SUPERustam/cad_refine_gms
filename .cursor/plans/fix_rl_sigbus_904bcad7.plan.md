---
name: Fix RL SIGBUS
overview: Understand why the resumed multi-GPU RL run crashes with `SIGBUS` after a few steps, then harden the training/reward pipeline and launch defaults to avoid the shared-memory / IPC failure pattern.
todos:
  - id: inspect-reward-ipc
    content: Refactor the reward worker flow in `metrics_async.py` to avoid nested per-sample process creation and reduce semaphore/shared-memory pressure.
    status: completed
  - id: add-runtime-guardrails
    content: Add lightweight logging or safeguards for effective reward concurrency and crash reporting in the RL training path.
    status: completed
  - id: tune-launch-defaults
    content: Lower risky resume-time concurrency defaults in the GMS config/script and align the script with the intended command shape.
    status: completed
  - id: verify-resume-window
    content: Run a short resume validation past the prior crash window and confirm the SIGBUS failure is gone.
    status: completed
isProject: false
---

# Fix RL SIGBUS Crash

## Working Theory

The run is resuming correctly, training for about 33 steps, and then multiple ranks die together with `Signal 7 (SIGBUS)`. The strongest evidence points to IPC/shared-memory pressure in the reward path rather than a bad checkpoint or a vLLM startup failure:

- [`/scratch/498rustam/cad_refine_m/metrics_async.py`] creates a global `NonDaemonPool`, then `get_metrics_from_texts()` submits one async task per completion, and each pool worker calls `timed_process_text()` which forks yet another child process with a `Pipe` for timeout isolation.
- [`/scratch/498rustam/cad_refine_m/grpo_trainer.py`] expands each local prompt batch by `num_generations` and uses `gather_object()` / `broadcast_object_list()` across ranks, so with the attached launch shape the process fanout and IPC volume are both high.
- [`/scratch/498rustam/cad_refine_m/logs_rl/rl_gms_train_sft_30682_resume_54000.log`] shows heavy slowdown before the crash, then leaked semaphore warnings at shutdown, which matches multiprocessing resource exhaustion much better than a simple Python exception.
- [`/scratch/498rustam/cad_refine_m/logs_rl/vllm_server.log`] shows a healthy vLLM server accepting requests and weight-sync calls, so vLLM is probably not the first failing component.

## Planned Changes

- Harden [`/scratch/498rustam/cad_refine_m/metrics_async.py`]: replace or reduce the nested fork-per-sample timeout pattern so each reward task does not create an extra child process inside a long-lived pool worker. Keep timeout protection, but bound process creation and shared-memory/semaphore usage.
- Add small diagnostics in [`/scratch/498rustam/cad_refine_m/metrics_async.py`] and/or [`/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py`] to log effective reward concurrency, batch sizes, and clear failure reasons when a worker crashes, so future `SIGBUS` events are easier to distinguish from ordinary CadQuery errors.
- Reduce unsafe launch defaults in [`/scratch/498rustam/cad_refine_m/configs/gms_config.yaml`] and the GMS loop script(s), starting with `pool_size` and likely `generation_batch_size`, so the default resumed run stays within safer IPC/memory limits.
- Align [`/scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_4.sh`] with the intended runtime shape, because the current file contents and the attached command snippet differ. I will preserve the intended resume behavior but make the heavy parameters explicit in one place to avoid drift.

## Validation

- Re-run a short resumed training segment from the same checkpoint with the safer settings.
- Confirm the job gets past the previous failure window around steps `54030-54033` without rank-wide `SIGBUS`.
- Check that reward computation still returns usable values and that shutdown no longer reports leaked semaphores, or at least reports fewer/no recurring IPC leaks.

## Likely Files

- [`/scratch/498rustam/cad_refine_m/metrics_async.py`]
- [`/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py`]
- [`/scratch/498rustam/cad_refine_m/configs/gms_config.yaml`]
- [`/scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_4.sh`]

