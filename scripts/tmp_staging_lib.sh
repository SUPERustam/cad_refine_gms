#!/usr/bin/env bash
# Shared helpers for staging paths to /tmp and syncing writable outputs back.
# Sourced by train_loop_dp_gms_resume_4.sh (and optionally slurm_scripts/copy_paster.sh).

# Copy tree from SRC to DST (mkdir -p DST parent). Uses rsync if available, else cp -a.
tmp_staging_copy_tree() {
  local src="$1"
  local dst="$2"
  [[ -e "$src" ]] || { echo "[tmp_staging] skip missing: $src" >&2; return 0; }
  mkdir -p "$(dirname "$dst")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src/" "$dst/"
  else
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
  fi
}

# Incremental sync SRC -> DST (writable outputs). No --delete unless caller passes extra rsync args.
tmp_staging_sync_back() {
  local src="$1"
  local dst="$2"
  shift 2
  [[ -d "$src" ]] || return 0
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$@" "$src/" "$dst/"
  else
    cp -a "$src/." "$dst/"
  fi
}

# Highest numeric N among checkpoint-N under DIR; empty if none.
tmp_staging_latest_checkpoint_num() {
  local dir="$1"
  local n=""
  local d
  shopt -s nullglob
  for d in "$dir"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    local base="${d##*/}"
    local num="${base#checkpoint-}"
    [[ "$num" =~ ^[0-9]+$ ]] || continue
    if [[ -z "$n" ]] || ((10#$num > 10#$n)); then
      n="$num"
    fi
  done
  shopt -u nullglob
  echo "${n:-}"
}

# Loop: when checkpoint number under tmp_out increases, wait and sync tmp_out -> scratch_out.
# Args: tmp_out scratch_out poll_sec stable_delay
tmp_staging_checkpoint_watcher_loop() {
  local tmp_out="$1"
  local scratch_out="$2"
  local poll_sec="${3:-30}"
  local stable_delay="${4:-5}"
  local last=""
  mkdir -p "$scratch_out"
  while true; do
    local cur
    cur="$(tmp_staging_latest_checkpoint_num "$tmp_out")"
    if [[ -n "$cur" && "$cur" != "$last" ]]; then
      sleep "$stable_delay"
      # Sync only after a stable step save (avoids copying half-written checkpoint dirs)
      if [[ -f "$tmp_out/checkpoint-$cur/trainer_state.json" ]]; then
        last="$cur"
        echo "[tmp_staging] checkpoint-$cur observed; syncing $tmp_out -> $scratch_out" >&2
        tmp_staging_sync_back "$tmp_out" "$scratch_out"
      fi
    fi
    sleep "$poll_sec"
  done
}
