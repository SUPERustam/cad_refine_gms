import contextlib
import contextvars
import json
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


_EVENT_CONTEXT = contextvars.ContextVar("cad_logging_context", default={})
_LOGGER_INITIALIZED = False


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _runtime_defaults():
    return {
        "session_id": os.getenv("CAD_LOG_SESSION_ID", ""),
        "run_name": os.getenv("CAD_LOG_RUN_NAME", ""),
        "job_id": os.getenv("SLURM_JOB_ID", ""),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "rank": os.getenv("RANK", ""),
        "local_rank": os.getenv("LOCAL_RANK", ""),
        "world_size": os.getenv("WORLD_SIZE", ""),
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Exception):
        return serialize_exception(value)
    return repr(value)


def serialize_exception(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def truncate_text(text, max_chars):
    if text is None:
        return None
    if max_chars is None or max_chars < 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


class JsonlEventHandler(logging.Handler):
    def __init__(self, path):
        super().__init__(level=logging.INFO)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, record):
        payload = getattr(record, "event_payload", None)
        if payload is None:
            payload = {
                "timestamp": _utc_now(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        line = json.dumps(payload, ensure_ascii=True, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def bind_context(**kwargs):
    current = dict(_EVENT_CONTEXT.get())
    current.update({k: v for k, v in kwargs.items() if v is not None})
    token = _EVENT_CONTEXT.set(current)
    return token


def reset_context(token):
    _EVENT_CONTEXT.reset(token)


@contextlib.contextmanager
def scoped_context(**kwargs):
    token = bind_context(**kwargs)
    try:
        yield
    finally:
        reset_context(token)


def build_event(event, status=None, level="INFO", message=None, **fields):
    payload = _runtime_defaults()
    payload.update(_EVENT_CONTEXT.get())
    payload.update({k: v for k, v in fields.items() if v is not None})
    payload["timestamp"] = _utc_now()
    payload["event"] = event
    payload["status"] = status or payload.get("status") or "ok"
    payload["level"] = level
    if message is not None:
        payload["message"] = message
    return payload


def log_event(logger, event, status=None, level=logging.INFO, message=None, **fields):
    payload = build_event(
        event,
        status=status,
        level=logging.getLevelName(level),
        message=message,
        **fields,
    )
    text = message or f"{event} status={payload['status']}"
    logger.log(level, text, extra={"event_payload": payload})
    return payload


def ensure_session_id(run_name=None):
    session_id = os.getenv("CAD_LOG_SESSION_ID")
    if session_id:
        return session_id
    run_prefix = (run_name or os.getenv("CAD_LOG_RUN_NAME") or "run").replace(" ", "_")
    session_id = f"{run_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    os.environ["CAD_LOG_SESSION_ID"] = session_id
    return session_id


def setup_logging(
    run_name,
    log_dir="logs",
    structured_logging=True,
    failure_payload_logging=True,
    text_logging=True,
):
    global _LOGGER_INITIALIZED
    logger = logging.getLogger("cad_refine")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    session_id = ensure_session_id(run_name)
    os.environ["CAD_LOG_RUN_NAME"] = run_name
    os.environ["CAD_LOG_DIR"] = log_dir

    if _LOGGER_INITIALIZED:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if text_logging:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_path = Path(log_dir)
    if structured_logging:
        logger.addHandler(JsonlEventHandler(log_path / f"{run_name}.jsonl"))
    if failure_payload_logging:
        failure_logger = logging.getLogger("cad_refine.failures")
        failure_logger.setLevel(logging.INFO)
        failure_logger.propagate = False
        if not failure_logger.handlers:
            failure_logger.addHandler(
                JsonlEventHandler(log_path / f"{run_name}.failures.jsonl")
            )

    _LOGGER_INITIALIZED = True
    bind_context(run_name=run_name, session_id=session_id)
    log_event(
        logger,
        "logging_initialized",
        log_dir=str(log_dir),
        structured_logging=structured_logging,
        failure_payload_logging=failure_payload_logging,
    )
    return logger


def get_logger():
    return logging.getLogger("cad_refine")


def get_failure_logger():
    return logging.getLogger("cad_refine.failures")


def log_failure_payload(event, **fields):
    logger = get_failure_logger()
    if not logger.handlers:
        return None
    payload = build_event(
        event, level="ERROR", status=fields.pop("status", "error"), **fields
    )
    logger.info(event, extra={"event_payload": payload})
    return payload


def install_excepthooks(logger):
    def _handle_exception(exc_type, exc, tb):
        serialized = {
            "type": exc_type.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
        }
        log_event(
            logger,
            "uncaught_exception",
            status="error",
            level=logging.ERROR,
            exception=serialized,
        )

    def _handle_thread_exception(args):
        serialized = serialize_exception(args.exc_value)
        serialized["thread"] = getattr(args.thread, "name", "")
        log_event(
            logger,
            "thread_exception",
            status="error",
            level=logging.ERROR,
            exception=serialized,
        )

    sys.excepthook = _handle_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = _handle_thread_exception


def install_signal_handlers(logger, signals_to_handle=None):
    if signals_to_handle is None:
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]

    def _handler(signum, frame):
        signame = signal.Signals(signum).name
        log_event(
            logger,
            "process_signal",
            status="signal",
            level=logging.WARNING,
            signal=signame,
            frame=str(frame.f_code.co_name) if frame is not None else "",
        )
        raise SystemExit(128 + signum)

    for sig in signals_to_handle:
        try:
            signal.signal(sig, _handler)
        except Exception:
            continue
