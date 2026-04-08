#!/usr/bin/env bash

shell_logging_init() {
  : "${RUN_NAME:?RUN_NAME is required}"
  : "${LOG_FILE:?LOG_FILE is required}"
  LOG_DIR="${LOG_DIR:-$(dirname "$LOG_FILE")}"
  mkdir -p "$LOG_DIR"
  export CAD_LOG_RUN_NAME="${CAD_LOG_RUN_NAME:-$RUN_NAME}"
  export CAD_LOG_DIR="${CAD_LOG_DIR:-$LOG_DIR}"
  export CAD_LOG_SESSION_ID="${CAD_LOG_SESSION_ID:-${RUN_NAME}-$(date +%s)-$$}"
  export STRUCTURED_LOG_FILE="${STRUCTURED_LOG_FILE:-$LOG_DIR/${RUN_NAME}.jsonl}"
}

shell_log_event() {
  local event="$1"
  local status="${2:-ok}"
  local message="${3:-}"
  python3 -c 'import json, os, sys, socket, datetime
event, status, message = sys.argv[1:4]
payload = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event": event,
    "status": status,
    "message": message,
    "run_name": os.getenv("RUN_NAME", ""),
    "session_id": os.getenv("CAD_LOG_SESSION_ID", ""),
    "job_id": os.getenv("SLURM_JOB_ID", ""),
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    "log_file": os.getenv("LOG_FILE", ""),
    "vllm_log": os.getenv("VLLM_LOG", ""),
}
for item in sys.argv[4:]:
    if "=" in item:
        key, value = item.split("=", 1)
        payload[key] = value
with open(os.getenv("STRUCTURED_LOG_FILE"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
' "$event" "$status" "$message" "${@:4}"
}
