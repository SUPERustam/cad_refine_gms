#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs) # export all environment variables from .env file

set -euo pipefail

DELAY=5
BASE_DIR="/scratch/498rustam/cad_refine_m/rl_checkpoints_resume_16400/"
CHECKPOINT="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"
RUN_NAME="rl_gms_train_sft_30682_resume_16400"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

LAUNCH_SCRIPT="/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py"
CONFIG_FILE="/scratch/498rustam/cad_refine_m/configs/gms_config.yaml"

LOG_FILE="/scratch/498rustam/cad_refine_m/logs_rl/${RUN_NAME}.log"
VLLM_LOG="/scratch/498rustam/cad_refine_m/logs_rl/vllm_server.log"

ACCELERATE_LOG_LEVEL=INFO
NCCL_DEBUG=INFO

RESUME="/scratch/498rustam/cad_refine_m/rl_checkpoints/checkpoint-16400" # RL latest checkpoint

export METRICS_VAR_NAME='r' # for Cadrille format
export RUN_NAME LOG_FILE VLLM_LOG

source "$(dirname "$0")/shell_logging.sh"
shell_logging_init

on_err() {
  local exit_code=$?
  shell_log_event "shell_error" "error" "shell command failed" "line=$1" "command=$2" "exit_code=$exit_code"
}

on_exit() {
  local exit_code=$?
  shell_log_event "script_exit" "exit" "training loop exiting" "exit_code=$exit_code" "checkpoint=$CHECKPOINT" "resume=$RESUME"
}

trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR
trap 'shell_log_event "signal_received" "signal" "SIGINT received"; exit 130' INT
trap 'shell_log_event "signal_received" "signal" "SIGTERM received"; exit 143' TERM
trap 'on_exit' EXIT

shell_log_event "script_started" "ok" "starting training loop" "base_dir=$BASE_DIR" "config_file=$CONFIG_FILE" "launch_script=$LAUNCH_SCRIPT"

CMD='script --flush ${LOG_FILE} \
--command "COMET_API_KEY=${COMET_API_KEY} COMET_PROJECT_NAME=${COMET_PROJECT_NAME} COMET_WORKSPACE=${COMET_WORKSPACE} CUDA_VISIBLE_DEVICES=2,3,4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} --importance_sampling_level token --output_dir ${BASE_DIR} \
--run_name ${RUN_NAME} --use_vllm true --max_completion_length 3500 --temperature 1 --top_p 1 --top_k 100 \
--pool_size 40 --num_generations 16 --top_samples 4 --generation_batch_size 384 --per_device_train_batch_size 1 --max_prompt_length 600 \
--sft_path ${CHECKPOINT} \
--learning_rate 1e-5 --failure_reward 0 --gradient_accumulation_steps 2  --resume_ckpt_path ${RESUME} --save_steps 400 \
--save_total_limit ${SAVE_TOTAL_LIMIT}"'

while true; do
  shell_log_event "vllm_starting" "ok" "starting vllm server" "cuda_visible_devices=1"
  CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  shell_log_event "vllm_started" "ok" "vllm server process spawned" "pid=$VLLM_PID"
  sleep 80
  shell_log_event "train_launch" "ok" "launching accelerate job" "resume=$RESUME" "checkpoint=$CHECKPOINT"
  set +e
  eval "$CMD"
  TRAIN_EXIT_CODE=$?
  set -e
  TRAIN_EXIT_SIGNAL=""
  if [[ "$TRAIN_EXIT_CODE" -ge 128 ]]; then
    TRAIN_EXIT_SIGNAL="$(kill -l "$((TRAIN_EXIT_CODE - 128))" 2>/dev/null || true)"
  fi
  shell_log_event "train_exit" "exit" "accelerate job finished" "exit_code=$TRAIN_EXIT_CODE" "exit_signal=$TRAIN_EXIT_SIGNAL" "resume=$RESUME" "checkpoint=$CHECKPOINT"

  pkill -9 -f 'cadtrl|VLLM|vllm'  || true
  shell_log_event "cleanup_complete" "ok" "killed matching training/vllm processes"

  number=$(ls -d "$BASE_DIR"/checkpoint-* 2>/dev/null \
    | sed 's/.*checkpoint-//' \
    | sort -n \
    | tail -n 1|| true)

  if [[ -n "$number" ]]; then
    CHECKPOINT="$BASE_DIR/checkpoint-$number"
    echo "[$(date)] New CHECKPOINT: $CHECKPOINT"
    shell_log_event "checkpoint_selected" "ok" "new checkpoint selected" "checkpoint=$CHECKPOINT"
  fi
  RESUME=$CHECKPOINT

  echo "[$(date)] exited; restarting in ${DELAY}s..."
  shell_log_event "restart_sleep" "ok" "sleep before restart" "delay_sec=$DELAY"
  sleep "$DELAY"

  break # don't restart the loop: too many dashboards in Comet
done
