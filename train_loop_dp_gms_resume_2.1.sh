#!/usr/bin/env bash

source .env
set -euo pipefail

RUN_NAME="rl_gms_train_sft_30682_resume_68000"
BASE_DIR="/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_68000/"
CHECKPOINT="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"

LAUNCH_SCRIPT="/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py"
CONFIG_FILE="/scratch/498rustam/cad_refine_m/configs/gms_config.yaml"

LOG_FILE="/scratch/498rustam/cad_refine_m/logs/${RUN_NAME}.log"
VLLM_LOG="/scratch/498rustam/cad_refine_m/logs/vllm_server.log"

ACCELERATE_LOG_LEVEL=INFO
NCCL_DEBUG=INFO

RESUME="/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_54000/checkpoint-68000" # RL latest checkpoint

METRICS_VAR_NAME='r' # for Cadrille format

CMD='script --flush ${LOG_FILE} \
--command "COMET_API_KEY=${COMET_API_KEY} COMET_PROJECT_NAME=${COMET_PROJECT_NAME} COMET_WORKSPACE=${COMET_WORKSPACE} CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} \
--output_dir ${BASE_DIR} --run_name ${RUN_NAME} --sft_path ${CHECKPOINT} --resume_ckpt_path ${RESUME}"'


# Lower --gpu-memory-utilization vs default 0.9 to reduce KV-cache footprint and peak load on the vLLM GPU.
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 \
  --gpu-memory-utilization 0.75 >"$VLLM_LOG" 2>&1 &
sleep 80
eval "$CMD"