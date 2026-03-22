# Reinforcement Learning Theory

This document explains the theoretical framework and algorithmic implementation of Reinforcement Learning (RL) in this project.

## Overview

The goal of RL in this project is to fine-tune a Vision-Language Model (VLM) to generate high-quality **CadQuery** code from visual inputs (rendered STLs). Unlike supervised fine-tuning (SFT), which minimizes token-level cross-entropy against a ground truth, RL optimizes for a non-differentiable **reward signal** derived from the execution of the generated code.

## The RL Loop

The project uses a variant of **Group Relative Policy Optimization (GRPO)**, enhanced with a **CPPO (Clipped PPO)** top-K selection mechanism.

```mermaid
graph TD
    Dataset[Dataset: Image + Prompt] --> Rollout[Rollout: Generate G completions]
    Rollout --> Execute[Execute CadQuery Code]
    Execute --> Reward[Compute Reward: IoU / Chamfer / AOC-GMS]
    Reward --> Advantage[Compute Group-Relative Advantages]
    Advantage --> TopK[Select Top-K Samples by |Advantage|]
    TopK --> Loss[Compute CPPO Loss]
    Loss --> Optimizer[Optimizer Step]
    Optimizer --> Dataset
```

### 1. Grouped Sampling (GRPO)
For each prompt in a batch, the model generates $G$ independent completions (rollouts). In this project, $G$ is typically set to 16.
The rewards $r_i$ for these $G$ completions are computed, and then normalized within the group to produce advantages $A_i$:

$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$

This group-relative approach removes the need for a separate value-function (critic) model, significantly reducing VRAM requirements.

### 2. Top-K Selection (CPPO Variant)
To improve training stability and focus on the most informative samples, we implement a custom selection mechanism in `grpo_loss.py`:
- We select the `top_samples` (e.g., 4 out of 16) that have the highest **absolute advantage**.
- These samples represent the "best" and "worst" performers relative to the group mean, providing the strongest gradient signal.

### 3. Reward Function
The reward is the ground truth of the RL process. In this project, it is defined in `rewards.py`:
- **Execution Check**: If the code fails to execute or doesn't produce a valid mesh, it receives a `failure_reward` (e.g., -10 or 0).
- **Geometric Metrics**: If execution succeeds, we compute:
    - **IoU (Intersection over Union)**: Volumetric overlap between predicted and GT meshes.
    - **Chamfer Distance**: Point-cloud similarity.
    - **AOC-GMS**: Area Over the Curve of Global Multiview Similarity (surface normal alignment).
- **Clipping**: Final rewards are typically clipped to a range like $[-10, 10]$.

## Distributed Execution and vLLM

To accelerate the bottleneck of generation, we use **vLLM** in a server-client architecture.

```mermaid
sequenceDiagram
    participant T as Trainer (Accelerate Rank 0)
    participant V as vLLM Server
    participant W as Training Workers (All Ranks)
    participant P as Metrics Pool (Async)

    T->>V: Send Batch of Prompts
    V->>V: Generate G completions per prompt
    V-->>T: Return Completion IDs
    T->>W: Broadcast Completions to all Workers
    W->>P: Execute CadQuery & Compute Rewards
    P-->>W: Return Reward Tensors
    W->>W: Compute CPPO Loss & Gradients
    W->>W: All-Reduce Gradients & Update Model
```

### Key Implementation Details

- **Importance Sampling**: We support both `token` and `sequence` level importance sampling. `sequence` level averages the log-probability ratio over the completion length before applying the PPO clip.
- **Alignment**: `steps_per_generation` is aligned with `gradient_accumulation_steps` to ensure that the "old" log-probabilities used in the PPO ratio are consistent with the model version that generated them.
- **VLM Handling**: The trainer handles multimodal inputs (images + text) by passing pixel values and grid information through the rollout and loss computation phases.
