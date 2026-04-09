# Reinforcement Learning Practical Guide

This document provides practical instructions on how to configure, run, and monitor RL training in this project.

## Prerequisites

1.  **Environment, Configuration and Dataset**: Follow the steps in [Setup Guide](Setup.md) to activate the environment, set up the `.env` file, create necessary directories, and prepare the dataset.

## Running Training

### 1. The "Loop" Script (Recommended)
Use one of the `train_loop_dp_*.sh` scripts. These scripts handle the vLLM server lifecycle and automatic restarts/resuming.

```bash
# Example: Launching the GMS training loop
bash train_loop_dp_gms.sh
```

**What the loop script does:**
1.  Starts a `vllm-serve` instance on a dedicated GPU (e.g., `CUDA_VISIBLE_DEVICES=1`).
2.  Waits for the server to be ready.
3.  Launches `accelerate launch rl_train_cos_sched.py` on the remaining GPUs.
4.  If the training crashes or finishes, it kills the vLLM server and restarts from the latest checkpoint.

### 2. Manual Launch
For debugging, you can launch directly:
```bash
CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch rl_train.py \
    --config configs/config.yaml \
    --run_name my_experiment \
    --output_dir rl_checkpoints/my_experiment
```

## Key Configuration Parameters

These are found in `configs/*.yaml` or passed as CLI arguments:

| Parameter | Description | Recommended |
| :--- | :--- | :--- |
| `num_generations` ($G$) | Number of completions per prompt. | 16 |
| `top_samples` ($K$) | Number of samples to actually train on (CPPO). | 4 |
| `per_device_train_batch_size` | Number of *prompts* per GPU. | 1 or 4 (depends on VRAM) |
| `generation_batch_size` | Total completions in one vLLM call. | $G \times \text{batch\_size} \times \text{GPUs}$ |
| `max_completion_length` | Max tokens for CadQuery code. | 3000 - 3500 |
| `importance_sampling_level` | `token` or `sequence`. | `token` |
| `failure_reward` | Reward given if code fails to run. | 0 or -10 |
| `pool_size` | Number of CPU workers for CadQuery execution. | 16 - 40 |

## Monitoring

### 1. Metrics to Watch (Comet/WandB)
-   **`reward`**: The mean reward across the group. Should trend upwards.
-   **`reward_std`**: If this drops to zero, the model has collapsed to a single output.
-   **`entropy`**: Measures output diversity. If it drops too fast, increase `beta` or check learning rate.
-   **`clip_ratio/region_mean`**: Percentage of samples being clipped by PPO. Ideally 0.1 - 0.3.
-   **`completions/mean_length`**: Watch for "reward hacking" where the model generates extremely long/short code to exploit the reward.

### 2. Log Files
-   **Main log**: `logs/RUN_NAME.log` (Training progress and sampled code).
-   **Structured log**: `logs/RUN_NAME.jsonl` (Shell + trainer + reward events with step/run correlation).
-   **Failure payload log**: `logs/RUN_NAME.failures.jsonl` (Per-sample failure payloads and serialized exceptions).
-   **vLLM log**: `logs/vllm_server.log` (Check this if generation hangs).

For the full debugging and maintenance workflow, see [Logging System Guide](Logging_System.md).

For the full debugging and maintenance workflow, see [Logging System Guide](Logging_System.md).

## Troubleshooting

See [Troubleshooting](Troubleshooting.md) for more details.
