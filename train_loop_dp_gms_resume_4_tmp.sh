#!/usr/bin/env bash
# RL GMS train loop with optional staging of read-heavy paths to /tmp and
# checkpoint-triggered sync-back of writable outputs to scratch.
#
# Paths and toggles: edit configs/train_loop_dp_gms_resume_4_tmp.env (or set
# TRAIN_LOOP_CONFIG to another file before launching).
# For Slurm helpers that only need the same path variables:
#   TRAIN_LOOP_PATHS_ONLY=1 source ./train_loop_dp_gms_resume_4_tmp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
fi

TRAIN_LOOP_CONFIG="${TRAIN_LOOP_CONFIG:-$SCRIPT_DIR/configs/train_loop_dp_gms_resume_4_tmp.env}"
if [[ ! -f "$TRAIN_LOOP_CONFIG" ]]; then
  echo "error: TRAIN_LOOP_CONFIG not found: $TRAIN_LOOP_CONFIG" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$TRAIN_LOOP_CONFIG"

# Per-job session under TMP_ROOT (uses PID; not set in config file)
TMP_SESSION="${TMP_SESSION:-$TMP_ROOT/session_${RUN_NAME}_$$}"

# ---------------------------------------------------------------------------
# Paths-only mode: load config then return (for Slurm/debug scripts that source this file)
# ---------------------------------------------------------------------------
if [[ "${TRAIN_LOOP_PATHS_ONLY:-}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

set -euo pipefail

# shellcheck source=scripts/tmp_staging_lib.sh
source "$SCRIPT_DIR/scripts/tmp_staging_lib.sh"

export ACCELERATE_LOG_LEVEL
export NCCL_DEBUG
export METRICS_VAR_NAME

# ---------------------------------------------------------------------------
# Resolve dataset scratch path (same rule as Python: dp_f360 in output dir)
# ---------------------------------------------------------------------------
_data_scratch_path() {
  local out="$1"
  if [[ "$out" == *"dp_f360"* ]]; then
    echo "$DATASET_SCRATCH_DP_F360"
  else
    echo "$DATASET_SCRATCH_DEFAULT"
  fi
}

DATASET_SCRATCH="$(_data_scratch_path "$BASE_DIR_SCRATCH")"

# ---------------------------------------------------------------------------
# Runtime paths (default = scratch); overridden when staging is enabled
# ---------------------------------------------------------------------------
BASE_DIR="$BASE_DIR_SCRATCH"
CHECKPOINT="$CHECKPOINT_SCRATCH"
RESUME="$RESUME_SCRATCH"
LOG_FILE="$LOG_FILE_SCRATCH"
VLLM_LOG="$VLLM_LOG_SCRATCH"

CKPT_WATCH_PID=""
_tmp_staging_final_sync() {
  set +e
  if [[ -n "${CKPT_WATCH_PID:-}" ]] && kill -0 "$CKPT_WATCH_PID" 2>/dev/null; then
    kill "$CKPT_WATCH_PID" 2>/dev/null || true
    wait "$CKPT_WATCH_PID" 2>/dev/null || true
  fi
  if [[ "${ENABLE_TMP_STAGING:-0}" == "1" ]]; then
    echo "[tmp_staging] final sync (EXIT)" >&2
    if [[ "${SYNC_BACK_BASE_DIR:-0}" == "1" && -n "${BASE_DIR_TMP:-}" ]]; then
      tmp_staging_sync_back "$BASE_DIR_TMP" "$BASE_DIR_SCRATCH"
    fi
    if [[ "${SYNC_BACK_LOGS:-0}" == "1" ]]; then
      [[ -n "${LOG_FILE_TMP:-}" && -f "$LOG_FILE_TMP" ]] && mkdir -p "$(dirname "$LOG_FILE_SCRATCH")" && cp -a "$LOG_FILE_TMP" "$LOG_FILE_SCRATCH" || true
      [[ -n "${VLLM_LOG_TMP:-}" && -f "$VLLM_LOG_TMP" ]] && mkdir -p "$(dirname "$VLLM_LOG_SCRATCH")" && cp -a "$VLLM_LOG_TMP" "$VLLM_LOG_SCRATCH" || true
    fi
    if [[ "${SYNC_BACK_HF_CACHE:-0}" == "1" && -n "${HF_HOME_TMP:-}" ]]; then
      tmp_staging_sync_back "$HF_HOME_TMP" "$HF_CACHE_SCRATCH"
    fi
  fi
  set -e
}

if [[ "${ENABLE_TMP_STAGING:-0}" == "1" ]]; then
  mkdir -p "$TMP_SESSION"
  echo "[tmp_staging] session: $TMP_SESSION" >&2

  BASE_DIR_TMP="$TMP_SESSION/train_output"
  CHECKPOINT_TMP="$TMP_SESSION/sft_checkpoint"
  RESUME_TMP="$TMP_SESSION/resume_ckpt"
  DATASET_TMP="$TMP_SESSION/hf_dataset"
  HF_HOME_TMP="$TMP_SESSION/hf_home"
  LOG_FILE_TMP="$TMP_SESSION/logs/${RUN_NAME}.log"
  VLLM_LOG_TMP="$TMP_SESSION/logs/vllm_server.log"

  if [[ "${STAGE_BASE_DIR_INPUT:-0}" == "1" ]]; then
    tmp_staging_copy_tree "$BASE_DIR_SCRATCH" "$BASE_DIR_TMP"
  else
    mkdir -p "$BASE_DIR_TMP"
  fi
  BASE_DIR="$BASE_DIR_TMP"

  if [[ "${STAGE_CHECKPOINT:-0}" == "1" ]]; then
    tmp_staging_copy_tree "$CHECKPOINT_SCRATCH" "$CHECKPOINT_TMP"
    CHECKPOINT="$CHECKPOINT_TMP"
  fi

  if [[ "${STAGE_RESUME:-0}" == "1" ]]; then
    tmp_staging_copy_tree "$RESUME_SCRATCH" "$RESUME_TMP"
    RESUME="$RESUME_TMP"
  fi

  if [[ "${STAGE_DATASET:-0}" == "1" ]]; then
    tmp_staging_copy_tree "$DATASET_SCRATCH" "$DATASET_TMP"
    export HF_DATASET_OVERRIDE="$DATASET_TMP"
  else
    unset HF_DATASET_OVERRIDE 2>/dev/null || true
  fi

  if [[ "${STAGE_HF_CACHE:-0}" == "1" ]]; then
    mkdir -p "$HF_HOME_TMP"
    if [[ -d "$HF_CACHE_SCRATCH" ]]; then
      tmp_staging_copy_tree "$HF_CACHE_SCRATCH" "$HF_HOME_TMP"
    fi
    export HF_HOME="$HF_HOME_TMP"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME_TMP}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME_TMP}/transformers"
  fi

  if [[ "${STAGE_LOGS_TO_TMP:-0}" == "1" ]]; then
    mkdir -p "$(dirname "$LOG_FILE_TMP")" "$(dirname "$VLLM_LOG_TMP")"
    LOG_FILE="$LOG_FILE_TMP"
    VLLM_LOG="$VLLM_LOG_TMP"
  fi

  trap '_tmp_staging_final_sync' EXIT

  if [[ "${SYNC_BACK_BASE_DIR:-0}" == "1" ]]; then
    tmp_staging_checkpoint_watcher_loop "$BASE_DIR_TMP" "$BASE_DIR_SCRATCH" \
      "$CKPT_WATCH_POLL_SEC" "$CKPT_STABLE_DELAY_SEC" &
    CKPT_WATCH_PID=$!
  fi
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$VLLM_LOG")"

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model "${VLLM_MODEL_ID}" \
  --max_model_len "${VLLM_MAX_MODEL_LEN}" \
  >"$VLLM_LOG" 2>&1 &
sleep 80

# `script -c` accepts only one command string; pass env the same way as the original script.
_SCRIPT_INNER="COMET_API_KEY=${COMET_API_KEY} COMET_PROJECT_NAME=${COMET_PROJECT_NAME} COMET_WORKSPACE=${COMET_WORKSPACE} \
CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[[ -n "${HF_DATASET_OVERRIDE:-}" ]] && _SCRIPT_INNER+=" HF_DATASET_OVERRIDE=${HF_DATASET_OVERRIDE}"
[[ -n "${HF_HOME:-}" ]] && _SCRIPT_INNER+=" HF_HOME=${HF_HOME}"
[[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]] && _SCRIPT_INNER+=" HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
[[ -n "${TRANSFORMERS_CACHE:-}" ]] && _SCRIPT_INNER+=" TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
_SCRIPT_INNER+=" accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} \
  --output_dir ${BASE_DIR} --run_name ${RUN_NAME} --sft_path ${CHECKPOINT} \
  --resume_ckpt_path ${RESUME} --save_total_limit ${SAVE_TOTAL_LIMIT}"

script --flush -c "${_SCRIPT_INNER}" "$LOG_FILE"

# rerun
# pkill -9 -f 'cadtrl|VLLM|vllm'  || true
# number=$(ls -d "$BASE_DIR"/checkpoint-* 2>/dev/null \
#   | sed 's/.*checkpoint-//' \
#   | sort -n \
#   | tail -n 1|| true)

# if [[ -n "$number" ]]; then
#   CHECKPOINT="$BASE_DIR/checkpoint-$number"
#   echo "[$(date)] New CHECKPOINT: $CHECKPOINT"
# fi
# RESUME=$CHECKPOINT
