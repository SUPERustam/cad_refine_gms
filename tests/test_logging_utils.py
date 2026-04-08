import json
import logging

from logging_utils import JsonlEventHandler, build_event


def test_build_event_includes_core_fields():
    payload = build_event("unit_test_event", status="ok", custom_field="value")
    assert payload["event"] == "unit_test_event"
    assert payload["status"] == "ok"
    assert payload["custom_field"] == "value"
    assert "timestamp" in payload


def test_jsonl_handler_writes_event_payload(tmp_path):
    logger = logging.getLogger("cad_refine_test_logger")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    log_path = tmp_path / "events.jsonl"
    logger.addHandler(JsonlEventHandler(log_path))
    payload = build_event("handler_test", status="ok", sample_idx=3)
    logger.info("handler_test", extra={"event_payload": payload})

    data = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert data["event"] == "handler_test"
    assert data["sample_idx"] == 3
