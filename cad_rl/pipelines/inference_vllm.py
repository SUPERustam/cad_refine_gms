"""vLLM-backed batch inference for Qwen2-VL (used by ``infer.py`` only)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from cad_rl.algorithms.frozen import configure_process_environment
from cad_rl.pipelines.inference import (
    _extract_assistant_text,
    _qwen_batch_messages_texts_and_images,
)


def hf_generate_kwargs_to_vllm_sampling_params(tokenizer: Any, hf_kwargs: dict) -> Any:
    """
    Map Hugging Face ``generate()`` kwargs to vLLM ``SamplingParams`` (vLLM 0.10+).

    Drops keys that only apply to transformers (e.g. ``pad_token_id``).
    """
    from vllm import SamplingParams

    kw = dict(hf_kwargs)
    max_tokens = int(kw.pop("max_new_tokens", 2048))
    do_sample = bool(kw.pop("do_sample", True))
    temperature = float(kw.pop("temperature", 0.7))
    if not do_sample:
        temperature = 0.0
    top_p = float(kw.pop("top_p", 1.0))
    top_k = kw.pop("top_k", -1)
    repetition_penalty = float(kw.pop("repetition_penalty", 1.0))
    kw.pop("pad_token_id", None)
    eos_token_id = kw.pop("eos_token_id", None)
    bad_words_ids = kw.pop("bad_words_ids", None)

    sp_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None and int(top_k) >= 0:
        sp_kwargs["top_k"] = int(top_k)

    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            sp_kwargs["stop_token_ids"] = [eos_token_id]
        else:
            sp_kwargs["stop_token_ids"] = list(eos_token_id)

    if bad_words_ids:
        bad_words: list[str] = []
        for seq in bad_words_ids:
            if not seq:
                continue
            bad_words.append(tokenizer.decode(list(seq), skip_special_tokens=False))
        if bad_words:
            sp_kwargs["bad_words"] = bad_words

    return SamplingParams(**sp_kwargs)


def _collate_prompts_images_paths(batch, processor):
    texts, images, _videos, mesh_paths = _qwen_batch_messages_texts_and_images(
        batch, processor
    )
    return {"texts": texts, "images": images, "mesh_path": mesh_paths}


def generate_inference_records_vllm(
    llm: Any,
    processor: Any,
    dataset: Any,
    output_path: str | Path,
    generation_config: dict,
    task_spec: Any,
) -> Path:
    """
    Same JSONL records as :func:`generate_inference_records`, using in-process vLLM.

    ``generation_config`` must include ``sampling_params`` (vLLM ``SamplingParams``)
    plus optional ``batch_size``, ``num_workers``.
    """
    configure_process_environment()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    var_name = getattr(task_spec, "output_var_name", None) or getattr(
        task_spec, "output_variable", "result"
    )
    sampling_params = generation_config["sampling_params"]

    with output_path.open("w", encoding="utf-8") as handle:
        loader = DataLoader(
            dataset,
            batch_size=generation_config.get("batch_size", 8),
            shuffle=False,
            num_workers=generation_config.get("num_workers", 0),
            collate_fn=lambda batch: _collate_prompts_images_paths(batch, processor),
        )
        for batch in loader:
            texts = batch["texts"]
            images = batch["images"]
            mesh_paths = batch["mesh_path"]
            vllm_inputs = [
                {"prompt": prompt, "multi_modal_data": {"image": img}}
                for prompt, img in zip(texts, images)
            ]
            started = time.time()
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            elapsed = time.time() - started
            for mesh_path, o in zip(mesh_paths, outputs):
                raw = o.outputs[0].text
                text = _extract_assistant_text(raw)
                record = {
                    "task_id": getattr(task_spec, "task_id", "unknown"),
                    "mesh_path": mesh_path,
                    "output_variable": var_name,
                    "raw_generation": text,
                    "wrapped_code": text,
                    "timing_sec": elapsed,
                    "execution": {"status": "not_run"},
                }
                handle.write(json.dumps(record) + "\n")
    return output_path
