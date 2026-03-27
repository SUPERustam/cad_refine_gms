from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None

from cad_rl.algorithms.frozen import configure_process_environment


def _extract_assistant_text(decoded: str) -> str:
    """Assistant span between chat markers (matches Qwen2-VL training / inference_cad_model example)."""
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    if start_tag in decoded:
        part = decoded.split(start_tag, maxsplit=1)[1]
        if part.startswith("\n"):
            part = part[1:]
        if end_tag in part:
            part = part.split(end_tag, maxsplit=1)[0]
        return part.strip()
    return decoded.strip()


def _qwen_batch_messages_texts_and_images(batch, processor):
    """Shared Qwen2-VL prompt strings and vision inputs (matches training collate)."""
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": sample["image"]}]}]
        for sample in batch
    ]
    texts = [
        processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    if process_vision_info is None:
        images = [sample["image"] for sample in batch]
        videos = None
    else:
        images, videos = process_vision_info(messages)
    mesh_paths = [str(sample["mesh_path"]) for sample in batch]
    return texts, images, videos, mesh_paths


def _collate_for_qwen(batch, processor):
    texts, images, videos, mesh_paths = _qwen_batch_messages_texts_and_images(
        batch, processor
    )
    if process_vision_info is None:
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    else:
        inputs = processor(
            text=texts, images=images, videos=videos, padding=True, return_tensors="pt"
        )
    inputs["mesh_path"] = mesh_paths
    return inputs


def generate_inference_records(
    model,
    processor,
    dataset,
    output_path: str | Path,
    generation_config: dict,
    task_spec,
) -> Path:
    configure_process_environment()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    var_name = getattr(task_spec, "output_var_name", None) or getattr(
        task_spec, "output_variable", "result"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        loader = DataLoader(
            dataset,
            batch_size=generation_config.get("batch_size", 8),
            shuffle=False,
            num_workers=generation_config.get("num_workers", 0),
            collate_fn=lambda batch: _collate_for_qwen(batch, processor),
        )
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                    if key != "mesh_path"
                }
                started = time.time()
                generated = model.generate(
                    **inputs, **generation_config["generate_kwargs"]
                )
                # Full-sequence decode + assistant span (same as examples/inference_cad_model.py).
                # Do not use attention_mask.sum() as a slice index with left padding — it counts
                # real tokens but ignores leading pads, so it corrupts the prompt/completion boundary.
                texts = []
                for row_ids in generated:
                    decoded = processor.tokenizer.decode(
                        row_ids, skip_special_tokens=False
                    )
                    texts.append(_extract_assistant_text(decoded))
                for mesh_path, text in zip(batch["mesh_path"], texts):
                    record = {
                        "task_id": getattr(task_spec, "task_id", "unknown"),
                        "mesh_path": mesh_path,
                        "output_variable": var_name,
                        "raw_generation": text,
                        "wrapped_code": text,
                        "timing_sec": time.time() - started,
                        "execution": {"status": "not_run"},
                    }
                    handle.write(json.dumps(record) + "\n")
    return output_path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_jsonl(records: list[dict], key: str) -> dict:
    values = [
        record[key]
        for record in records
        if key in record and isinstance(record[key], (int, float))
    ]
    if not values:
        return {
            "count": len(records),
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    values = sorted(values)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return {
        "count": len(records),
        "mean": sum(values) / len(values),
        "median": median,
        "min": values[0],
        "max": values[-1],
    }
