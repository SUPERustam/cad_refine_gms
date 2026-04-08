#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs) # export all environment variables from .env file

set -uo pipefail

DELAY=5
BASE_DIR="/scratch/498rustam/cad_refine_m/rl_checkpoints/"
CHECKPOINT="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"
RUN_NAME="rl_gms_train_sft_30682"

LAUNCH_SCRIPT="/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py"
CONFIG_FILE="/scratch/498rustam/cad_refine_m/configs/gms_config.yaml"

LOG_FILE="/scratch/498rustam/cad_refine_m/logs/${RUN_NAME}.log"
VLLM_LOG="/scratch/498rustam/cad_refine_m/logs/vllm_server.log"

ACCELERATE_LOG_LEVEL=INFO
NCCL_DEBUG=INFO

RESUME="False"

export METRICS_VAR_NAME='r' # for Cadrille format

CMD='script --flush ${LOG_FILE} \
--command "COMET_API_KEY=${COMET_API_KEY} COMET_PROJECT_NAME=${COMET_PROJECT_NAME} COMET_WORKSPACE=${COMET_WORKSPACE} CUDA_VISIBLE_DEVICES=2,3,4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} --importance_sampling_level token --output_dir ${BASE_DIR} \
--run_name ${RUN_NAME} --use_vllm true --max_completion_length 3500 --temperature 1 --top_p 1 --top_k 100 \
--pool_size 40 --num_generations 16 --top_samples 4 --generation_batch_size 384 --per_device_train_batch_size 1 --max_prompt_length 600 \
--sft_path ${CHECKPOINT} \
--learning_rate 1e-5 --failure_reward 0 --gradient_accumulation_steps 2  --resume_ckpt_path ${RESUME} --save_steps 400"'

while true; do
  CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 >"$VLLM_LOG" 2>&1 &
  sleep 80
  eval "$CMD"

  pkill -9 -f 'cadtrl|VLLM|vllm'  || true

  number=$(ls -d "$BASE_DIR"/checkpoint-* 2>/dev/null \
    | sed 's/.*checkpoint-//' \
    | sort -n \
    | tail -n 1|| true)

  if [[ -n "$number" ]]; then
    CHECKPOINT="$BASE_DIR/checkpoint-$number"
    echo "[$(date)] New CHECKPOINT: $CHECKPOINT"
  fi
  RESUME=$CHECKPOINT

  echo "[$(date)] exited; restarting in ${DELAY}s..."
  sleep "$DELAY"
done