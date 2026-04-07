from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional model dependency
    import torch
except Exception:  # pragma: no cover - lightweight environments
    torch = None

try:  # pragma: no cover - optional model dependency
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except Exception:  # pragma: no cover - lightweight environments
    AutoProcessor = None
    Qwen2VLForConditionalGeneration = None

from cad_rl.config import ModelSpec


DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if torch is None:
        raise RuntimeError("torch is required to resolve model dtype")
    aliases = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in aliases:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return aliases[dtype_name]


def create_processor(model_id: str = DEFAULT_MODEL_ID, **processor_kwargs):
    if AutoProcessor is None:
        raise RuntimeError("transformers is required to create the processor")
    kwargs = {
        "trust_remote_code": True,
        "resized_width": 14 * 17 * 2,
        "resized_height": 14 * 17 * 4,
        "padding_side": "left",
    }
    kwargs.update(processor_kwargs)
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def create_processor_from_spec(spec: ModelSpec):
    processor_name = spec.processor_name or spec.base_checkpoint
    return create_processor(processor_name, **dict(spec.processor_kwargs))


def load_qwen_model(model_path: str, **model_kwargs):
    if torch is None or Qwen2VLForConditionalGeneration is None:
        raise RuntimeError("torch and transformers are required to load the model")
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
    }
    kwargs.update(model_kwargs)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        **kwargs,
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model


def load_model_from_spec(spec: ModelSpec, checkpoint_path: str | None = None):
    return load_qwen_model(
        checkpoint_path or spec.base_checkpoint,
        torch_dtype=resolve_torch_dtype(spec.torch_dtype),
        attn_implementation=spec.attn_implementation,
        trust_remote_code=spec.trust_remote_code,
    )


def build_generation_kwargs(processor: Any) -> dict[str, Any]:
    bad_words = [
        "<|image_pad|>",
        "<|vision_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
    ]
    ids = [
        processor.tokenizer.convert_tokens_to_ids(token)
        for token in bad_words
        if processor.tokenizer.convert_tokens_to_ids(token)
        != processor.tokenizer.unk_token_id
    ]
    return {"bad_words": bad_words, "stop_token_ids": ids}
