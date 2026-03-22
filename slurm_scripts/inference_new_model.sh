#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs) # export all environment variables from .env file

set -uo pipefail

RUN_NAME="deepcad_1000_run"

CHECKPOINT="/scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600"
STL_DIR="/scratch/498rustam/datasets/deepcad_test_mesh_1000"
OUT_DIR="/scratch/498rustam/cad_refine_m/predictions/deepcad_test_mesh_1000"

CMD='python benchmark/inference.py \
      --stl_dir ${STL_DIR} \
      --out_dir ${OUT_DIR} \
      --model_path ${CHECKPOINT} \
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7'

eval "$CMD"