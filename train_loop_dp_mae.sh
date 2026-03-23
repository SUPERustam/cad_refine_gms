#!/usr/bin/env bash
# Single-run RL training launcher using MAE render metric.
# All training/reward parameters live in configs/mae_config.yaml.
# Set SFT_PATH, OUTPUT_DIR, RUN_NAME, RESUME as needed before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
fi

CONFIG_FILE="${SCRIPT_DIR}/configs/mae_config.yaml"
LAUNCH_SCRIPT="${SCRIPT_DIR}/rl_train_cos_sched.py"
LOG_DIR="${SCRIPT_DIR}/logs_rl"
VLLM_LOG="${LOG_DIR}/vllm_server.log"

OUTPUT_DIR="${SCRIPT_DIR}/rl_checkpoints/rl_mae_train"
RUN_NAME="grpo_mae_render_1"
SFT_PATH="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"
RESUME="False"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
VLLM_WAIT="80"

export METRICS_VAR_NAME="r"

CUDA_VLLM="0"
CUDA_TRAIN="1,2,3"

mkdir -p "${LOG_DIR}"

cleanup() {
  pkill -9 -f 'cadtrl|VLLM|vllm' 2>/dev/null || true
}
trap cleanup EXIT



echo "[$(date)] Starting vLLM server on GPU ${CUDA_VLLM}..."
CUDA_VISIBLE_DEVICES="${CUDA_VLLM}" trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
sleep "${VLLM_WAIT}"

if [[ -z "${SFT_PATH}" ]]; then
  echo "Error: SFT_PATH must be set (path to SFT checkpoint)" >&2
  exit 1
fi

echo "[$(date)] Launching training with config ${CONFIG_FILE}..."
script --flush "${LOG_FILE}" --command \
  "COMET_API_KEY=${COMET_API_KEY:-} COMET_PROJECT_NAME=${COMET_PROJECT_NAME:-} COMET_WORKSPACE=${COMET_WORKSPACE:-} \
   CUDA_VISIBLE_DEVICES=${CUDA_TRAIN} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} \
   --output_dir ${OUTPUT_DIR} --run_name ${RUN_NAME} \
   --sft_path ${SFT_PATH} --resume_ckpt_path ${RESUME}"

echo "[$(date)] Training finished."
