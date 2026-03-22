#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

RUN_NAME="fusion360_test_mesh_1000"


CHECKPOINT="$HOME/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600"
STL_DIR="/scratch/498rustam/datasets/fusion360_test_mesh_1000"
OUT_DIR="$HOME/cad_refine_m/predictions/fusion360_test_mesh_1000"

# Disable vLLM v1 engine to avoid "Cannot re-initialize CUDA in forked subprocess" error
export VLLM_USE_V1=0

python benchmark/inference_vllm.py \
  --stl_dir "${STL_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${CHECKPOINT}" \
  --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7