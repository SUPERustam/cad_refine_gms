#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs) # export all environment variables from .env file

set -uo pipefail

RUN_NAME="fusion360_test_mesh_1000"

CHECKPOINT="/scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600"
STL_DIR="/scratch/498rustam/datasets/fusion360_test_mesh_1000"
OUT_DIR="/scratch/498rustam/cad_refine_m/predictions/fusion360_test_mesh_1000"

# Disable vLLM v1 engine to avoid "Cannot re-initialize CUDA in forked subprocess" error
export VLLM_USE_V1=0

CMD='python benchmark/inference_vllm.py \
      --stl_dir ${STL_DIR} \
      --out_dir ${OUT_DIR} \
      --model_path ${CHECKPOINT} \
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7'

eval "$CMD"