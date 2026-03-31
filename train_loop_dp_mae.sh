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

RUN_NAME="grpo_mae_render_1_resume_10400_2"
OUTPUT_DIR="${SCRIPT_DIR}/rl_checkpoints/${RUN_NAME}"

SAVE_TOTAL_LIMIT="3"
SFT_PATH="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"
RESUME="/scratch/hsegrif/cad_refine_m/rl_checkpoints/rl_mae_train/checkpoint-10400"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
VLLM_WAIT="80"

export METRICS_VAR_NAME="r"

# Slurm/file logs are not a color terminal: Rich (TRL log_completions) and tqdm emit
# ANSI escapes that look like garbage in *.out; NO_COLOR disables them.
export NO_COLOR=1

# Full prompt/completion text in logs (Rich tables truncate columns). See grpo_trainer.TopSampleGRPOTrainer.log
export GRPO_LOG_FULL_COMPLETIONS=1
# Compare multiple completions for the same prompt (num_generations rollouts per input). Optional:
export GRPO_LOG_GROUP_BY_PROMPT=1
# export GRPO_LOG_PROMPT_GROUPS=2   # how many distinct prompts to print (default 1)

# PyVista/VTK: off-screen renders (bitmaps for metrics). Inherited by training + metric workers.
export PYVISTA_OFF_SCREEN=true
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

# VTK 9.x: metric workers clear CUDA, so EGL often has no device → VTK falls back to X11
# and logs "vtkXOpenGLRenderWindow ... bad X server connection" unless DISPLAY exists.
# Wrapping training in xvfb-run provides a virtual X server (no window on your desktop).
XVFB_PREFIX=()
if [[ "${PYVISTA_USE_XVFB:-1}" != "0" ]] && command -v xvfb-run >/dev/null 2>&1; then
  XVFB_PREFIX=(xvfb-run -a -s "-screen 0 4096x4096x24")
elif [[ "${PYVISTA_USE_XVFB:-1}" != "0" ]]; then
  echo "[train_loop_dp_mae] Note: xvfb-run not in PATH; install the xvfb package on the cluster to silence VTK X11 warnings on CPU metric workers." >&2
fi

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
"${XVFB_PREFIX[@]}" script --flush "${LOG_FILE}" --command \
  "COMET_API_KEY=${COMET_API_KEY:-} COMET_PROJECT_NAME=${COMET_PROJECT_NAME:-} COMET_WORKSPACE=${COMET_WORKSPACE:-} \
   CUDA_VISIBLE_DEVICES=${CUDA_TRAIN} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} \
   --output_dir ${OUTPUT_DIR} --run_name ${RUN_NAME} \
   --sft_path ${SFT_PATH} --resume_ckpt_path ${RESUME} \
   --save_total_limit ${SAVE_TOTAL_LIMIT}"

echo "[$(date)] Training finished."
