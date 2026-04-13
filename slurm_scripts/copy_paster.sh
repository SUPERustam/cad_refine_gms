#!/usr/bin/env bash

source .env
set -euo pipefail


RUN_NAME="rl_gms_train_sft_30682_resume_68000"
BASE_DIR="/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_68000/"

CHECKPOINT="/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LAUNCH_SCRIPT="/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py"
CONFIG_FILE="/scratch/498rustam/cad_refine_m/configs/gms_config.yaml"

LOG_FILE="/scratch/498rustam/cad_refine_m/logs/${RUN_NAME}.log"
VLLM_LOG="/scratch/498rustam/cad_refine_m/logs/vllm_server.log"

RESUME="/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_54000/checkpoint-68000"



# -------------------

mkdir -p "/tmp/superustam_daily"

cp -r "${BASE_DIR}" "/tmp/superustam_daily/$(basename "${BASE_DIR}")"

printf "[1] Copied base directory to tmp\n"

cp -r "${CHECKPOINT}" "/tmp/superustam_daily/$(basename "${CHECKPOINT}")"

printf "[2] Copied checkpoint to tmp\n"

cp -r "${RESUME}" "/tmp/superustam_daily/$(basename "${RESUME}")"

printf "[3] Copied resume to tmp\n"

# cp -r "${LOG_FILE}" "/tmp/superustam_daily/$(basename "${LOG_FILE}")"

# printf "[6] Copied log file to tmp\n"

# cp -r "${VLLM_LOG}" "/tmp/superustam_daily/$(basename "${VLLM_LOG}")"

# printf "[7] Copied vllm log file to tmp\n"

# ------------------------------------------------------------
cp -r "/tmp/superustam_daily/$(basename "${BASE_DIR}")" "/scratch/498rustam/cad_refine_m/temp/$(basename "${BASE_DIR}")"
printf "[8] Copied base directory back to scratch\n"


cp -r "/tmp/superustam_daily/$(basename "${RESUME}")" "/scratch/498rustam/cad_refine_m/temp/$(basename "${RESUME}")"
printf "[9] Copied resume back to scratch\n"


# cp -r "/tmp/superustam_daily/$(basename "${LOG_FILE}")" "/scratch/498rustam/cad_refine_m/temp/$(basename "${LOG_FILE}")"
# printf "[11] Copied log file back to scratch\n"


# cp -r "/tmp/superustam_daily/$(basename "${VLLM_LOG}")" "/scratch/498rustam/cad_refine_m/temp/$(basename "${VLLM_LOG}")"
# printf "[12] Copied vllm log file back to scratch"