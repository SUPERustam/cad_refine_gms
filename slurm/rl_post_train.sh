#!/bin/bash
#SBATCH --job-name=rl-post-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --output=slurm/rl_post_train.out
#SBATCH --error=slurm/rl_post_train.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

CONFIG_PATH=$1

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

eval "$(python -m cad_rl.vllm_server env --config "${CONFIG_PATH}")"

python -m cad_rl.vllm_server serve --config "${CONFIG_PATH}" \
  > "slurm/vllm_${SLURM_JOB_ID}.out" \
  2> "slurm/vllm_${SLURM_JOB_ID}.err" &
VLLM_PID=$!

CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
accelerate launch \
  --num_processes "${WORLD_SIZE}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  cli.py train \
  --config "${CONFIG_PATH}"
