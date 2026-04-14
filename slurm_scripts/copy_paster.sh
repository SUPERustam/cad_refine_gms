#!/usr/bin/env bash
# Standalone copy round-trip for debugging: uses the same path variables as
# train_loop_dp_gms_resume_4.sh (single source of truth).
#
# Edit paths only in ../train_loop_dp_gms_resume_4.sh (User configuration block).
# Optional env:
#   COPY_PASTER_TMP_ROOT  base under /tmp for staging (default: /tmp/cad_refine_m_staging_paste)
#   COPY_PASTER_BACK_ROOT where to copy BASE_DIR + RESUME back on scratch (default: $REPO_ROOT/temp)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TRAIN_LOOP_PATHS_ONLY=1
# shellcheck source=../train_loop_dp_gms_resume_4.sh
source "$REPO_ROOT/train_loop_dp_gms_resume_4.sh"
unset TRAIN_LOOP_PATHS_ONLY

# shellcheck source=../scripts/tmp_staging_lib.sh
source "$REPO_ROOT/scripts/tmp_staging_lib.sh"

DEST_PARENT="${COPY_PASTER_TMP_ROOT:-/tmp/cad_refine_m_staging_paste}"
BACK_ROOT="${COPY_PASTER_BACK_ROOT:-$REPO_ROOT/temp}"
mkdir -p "$DEST_PARENT" "$BACK_ROOT"

STAGE_ROOT="$DEST_PARENT/manual_copy_$$"
mkdir -p "$STAGE_ROOT"

echo "[copy_paster] staging to $STAGE_ROOT"

tmp_staging_copy_tree "$BASE_DIR_SCRATCH" "$STAGE_ROOT/$(basename "${BASE_DIR_SCRATCH%/}")"
printf "[1] Copied BASE_DIR_SCRATCH to tmp\n"

tmp_staging_copy_tree "$CHECKPOINT_SCRATCH" "$STAGE_ROOT/$(basename "${CHECKPOINT_SCRATCH%/}")"
printf "[2] Copied CHECKPOINT_SCRATCH to tmp\n"

tmp_staging_copy_tree "$RESUME_SCRATCH" "$STAGE_ROOT/$(basename "${RESUME_SCRATCH%/}")"
printf "[3] Copied RESUME_SCRATCH to tmp\n"

if [[ "${STAGE_DATASET:-0}" == "1" ]]; then
  _ds="$DATASET_SCRATCH_DEFAULT"
  [[ "$BASE_DIR_SCRATCH" == *"dp_f360"* ]] && _ds="$DATASET_SCRATCH_DP_F360"
  tmp_staging_copy_tree "$_ds" "$STAGE_ROOT/$(basename "$_ds")"
  printf "[4] Copied dataset to tmp\n"
fi

if [[ "${STAGE_HF_CACHE:-0}" == "1" && -d "${HF_CACHE_SCRATCH:-}" ]]; then
  tmp_staging_copy_tree "$HF_CACHE_SCRATCH" "$STAGE_ROOT/hf_cache"
  printf "[5] Copied HF cache to tmp\n"
fi

echo "[copy_paster] syncing writable outputs back to $BACK_ROOT"

tmp_staging_sync_back "$STAGE_ROOT/$(basename "${BASE_DIR_SCRATCH%/}")" "$BACK_ROOT/$(basename "${BASE_DIR_SCRATCH%/}")"
printf "[8] Synced base dir back to scratch temp\n"

tmp_staging_sync_back "$STAGE_ROOT/$(basename "${RESUME_SCRATCH%/}")" "$BACK_ROOT/$(basename "${RESUME_SCRATCH%/}")"
printf "[9] Synced resume tree back to scratch temp\n"

echo "[copy_paster] done"
