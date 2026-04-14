#!/usr/bin/env bash
# Compare /tmp staging session dirs vs scratch after train_loop_dp_gms_resume_4_tmp.sh.
# Uses the same config as the train loop (TRAIN_LOOP_CONFIG / configs/train_loop_dp_gms_resume_4_tmp.env).
#
# Usage:
#   scripts/tmp_staging_status.sh                    # latest session_${RUN_NAME}_* under TMP_ROOT
#   scripts/tmp_staging_status.sh /tmp/.../session_foo_12345
#   TRAIN_LOOP_CONFIG=/path/to.env scripts/tmp_staging_status.sh
# Options:
#   --dry-run-sync   run rsync -n for train_output tmp -> scratch (shows drift; needs rsync)
#   --grep-logs      print [tmp_staging] lines from LOG_FILE_SCRATCH if present

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LOOP_CONFIG="${TRAIN_LOOP_CONFIG:-$ROOT/configs/train_loop_dp_gms_resume_4_tmp.env}"
DRY_RUN_SYNC=0
GREP_LOGS=0
SESSION_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-sync) DRY_RUN_SYNC=1 ;;
    --grep-logs) GREP_LOGS=1 ;;
    -h|--help)
      sed -n '1,22p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$SESSION_ARG" ]]; then
        if [[ -d "$1" ]]; then
          SESSION_ARG="$(cd "$1" && pwd)"
        elif [[ "$1" == /* ]]; then
          SESSION_ARG="$1"
        else
          echo "error: unknown argument or missing directory: $1" >&2
          exit 2
        fi
      else
        echo "error: unexpected extra argument: $1" >&2
        exit 2
      fi
      ;;
  esac
  shift
done

if [[ ! -f "$TRAIN_LOOP_CONFIG" ]]; then
  echo "error: TRAIN_LOOP_CONFIG not found: $TRAIN_LOOP_CONFIG" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$TRAIN_LOOP_CONFIG"

_data_scratch_path() {
  local out="$1"
  if [[ "$out" == *"dp_f360"* ]]; then
    echo "$DATASET_SCRATCH_DP_F360"
  else
    echo "$DATASET_SCRATCH_DEFAULT"
  fi
}
DATASET_SCRATCH="$(_data_scratch_path "$BASE_DIR_SCRATCH")"

# shellcheck source=/dev/null
source "$ROOT/scripts/tmp_staging_lib.sh"

_dir_stats() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    printf '%-28s %s\n' "$label" "(missing) $path"
    return 1
  fi
  local bytes files
  bytes="$(du -sb "$path" 2>/dev/null | awk '{print $1}')" || bytes="?"
  files="$(find "$path" -xdev 2>/dev/null | wc -l | tr -d ' ')"
  printf '%-28s %12s bytes  %8s files  %s\n' "$label" "$bytes" "$files" "$path"
  return 0
}

_latest_session_dir() {
  local best="" best_t=0
  local d t
  shopt -s nullglob
  for d in "$TMP_ROOT"/session_"${RUN_NAME}"_*; do
    [[ -d "$d" ]] || continue
    t="$(stat -c %Y "$d" 2>/dev/null || stat -f %m "$d" 2>/dev/null || echo 0)"
    if ((t >= best_t)); then
      best_t="$t"
      best="$d"
    fi
  done
  shopt -u nullglob
  echo "${best:-}"
}

issues=0

echo "=== tmp staging status ==="
echo "config: $TRAIN_LOOP_CONFIG"
echo "ENABLE_TMP_STAGING=$ENABLE_TMP_STAGING  TMP_ROOT=$TMP_ROOT  RUN_NAME=$RUN_NAME"
echo "BASE_DIR_SCRATCH=$BASE_DIR_SCRATCH"
echo

if [[ "${ENABLE_TMP_STAGING:-0}" != "1" ]]; then
  echo "note: ENABLE_TMP_STAGING is not 1 in config; /tmp session layout may be from an older run or manual copy."
fi

SESSION="$SESSION_ARG"
if [[ -z "$SESSION" ]]; then
  SESSION="$(_latest_session_dir)"
fi

if [[ -z "$SESSION" || ! -d "$SESSION" ]]; then
  echo "error: no staging session directory found under $TMP_ROOT (pattern session_${RUN_NAME}_*)" >&2
  echo "       pass explicit path: $0 /tmp/cad_refine_m_staging/session_${RUN_NAME}_<pid>" >&2
  exit 2
fi

echo "SESSION=$SESSION"
echo

BASE_TMP="$SESSION/train_output"
CK_TMP="$SESSION/sft_checkpoint"
RS_TMP="$SESSION/resume_ckpt"
DS_TMP="$SESSION/hf_dataset"
HF_TMP="$SESSION/hf_home"

echo "--- staged on /tmp (session) ---"
_dir_stats "train_output (tmp)" "$BASE_TMP" || issues=$((issues + 1))
_dir_stats "sft_checkpoint (tmp)" "$CK_TMP" || true
_dir_stats "resume_ckpt (tmp)" "$RS_TMP" || true
[[ -d "$DS_TMP" ]] && _dir_stats "hf_dataset (tmp)" "$DS_TMP" || echo "hf_dataset (tmp)            (not present / STAGE_DATASET off)"
[[ -d "$HF_TMP" ]] && _dir_stats "hf_home (tmp)" "$HF_TMP" || echo "hf_home (tmp)               (not present / STAGE_HF_CACHE off)"
echo

echo "--- scratch targets ---"
_dir_stats "train_output (scratch)" "${BASE_DIR_SCRATCH%/}" || issues=$((issues + 1))
_dir_stats "sft_checkpoint (scratch)" "${CHECKPOINT_SCRATCH%/}" || true
_dir_stats "resume_ckpt (scratch)" "${RESUME_SCRATCH%/}" || true
echo

n_tmp="$(tmp_staging_latest_checkpoint_num "$BASE_TMP")"
n_sc="$(tmp_staging_latest_checkpoint_num "${BASE_DIR_SCRATCH%/}")"
echo "--- checkpoints (highest checkpoint-N) ---"
echo "tmp train_output latest N:     ${n_tmp:-<none>}"
echo "scratch train_output latest N: ${n_sc:-<none>}"

if [[ -n "$n_tmp" && -n "$n_sc" ]]; then
  if ((10#$n_tmp > 10#$n_sc)); then
    echo "WARN: tmp has a newer checkpoint than scratch; sync-back may be incomplete or still in progress." >&2
    issues=$((issues + 1))
  elif ((10#$n_tmp == 10#$n_sc)); then
    if [[ -n "$n_tmp" ]]; then
      t_ts="$(stat -c %Y "$BASE_TMP/checkpoint-$n_tmp" 2>/dev/null || stat -f %m "$BASE_TMP/checkpoint-$n_tmp" 2>/dev/null || echo 0)"
      s_ts="$(stat -c %Y "${BASE_DIR_SCRATCH%/}/checkpoint-$n_sc" 2>/dev/null || stat -f %m "${BASE_DIR_SCRATCH%/}/checkpoint-$n_sc" 2>/dev/null || echo 0)"
      if [[ "$t_ts" -gt "$s_ts" ]]; then
        echo "WARN: same checkpoint id but tmp dir is newer than scratch; final sync may not have run." >&2
        issues=$((issues + 1))
      else
        echo "OK: scratch checkpoint is same or newer by mtime (heuristic)."
      fi
    fi
  else
    echo "OK: scratch checkpoint N >= tmp N."
  fi
elif [[ -n "$n_tmp" && -z "$n_sc" ]]; then
  echo "WARN: checkpoints exist under tmp but none on scratch yet." >&2
  issues=$((issues + 1))
fi
echo

if [[ "$DRY_RUN_SYNC" == "1" ]]; then
  if command -v rsync >/dev/null 2>&1; then
    echo "--- rsync dry-run (tmp/train_output -> scratch) ---"
    rsync -n -a "$BASE_TMP/" "${BASE_DIR_SCRATCH%/}/" 2>/dev/null | head -n 50 || true
    echo "(truncated to 50 lines; full diff may be large)"
  else
    echo "rsync not installed; skip --dry-run-sync"
  fi
  echo
fi

if [[ "$GREP_LOGS" == "1" ]]; then
  echo "--- [tmp_staging] in $LOG_FILE_SCRATCH ---"
  if [[ -f "$LOG_FILE_SCRATCH" ]]; then
    rg '\[tmp_staging\]' "$LOG_FILE_SCRATCH" 2>/dev/null || grep '\[tmp_staging\]' "$LOG_FILE_SCRATCH" || echo "(no matches — messages may be on Slurm stderr only)"
  else
    echo "(log file missing)"
  fi
  echo
fi

if [[ "$issues" -eq 0 ]]; then
  echo "summary: no heuristic issues detected (check rsync/timestamps manually if unsure)."
else
  echo "summary: $issues issue(s) detected (see WARN lines)."
fi
exit "$issues"
