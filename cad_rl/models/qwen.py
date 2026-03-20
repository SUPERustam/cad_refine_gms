from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from cad_rl.specs.task import GenerationDefaults, ModelSpec


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    aliases = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in aliases:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return aliases[dtype_name]


def load_qwen_processor(spec: ModelSpec | None = None):
    from transformers import AutoProcessor

    spec = spec or ModelSpec()
    return AutoProcessor.from_pretrained(
        spec.processor_id,
        trust_remote_code=spec.trust_remote_code,
        resized_width=spec.processor_render.resized_width,
        resized_height=spec.processor_render.resized_height,
        padding_side=spec.processor_render.padding_side,
    )


def load_qwen_model(spec: ModelSpec | None = None, pretrained_path: str | None = None):
    from transformers import Qwen2VLForConditionalGeneration

    spec = spec or ModelSpec()
    return Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=pretrained_path or spec.model_id,
        torch_dtype=_resolve_torch_dtype(spec.torch_dtype),
        attn_implementation=spec.attn_implementation,
        trust_remote_code=spec.trust_remote_code,
    )


def build_qwen_generation_kwargs(
    processor: Any,
    defaults: GenerationDefaults | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = defaults or GenerationDefaults()
    bad_words = list(defaults.bad_words)
    stop_token_ids = []
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        unk = getattr(tokenizer, "unk_token_id", None)
        for token in bad_words:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk:
                stop_token_ids.append(token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != unk:
            stop_token_ids.append(im_end_id)

    payload: dict[str, Any] = {
        "bad_words": bad_words,
        "stop_token_ids": sorted(set(stop_token_ids)),
        "max_new_tokens": defaults.max_new_tokens,
        "do_sample": defaults.do_sample,
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "top_k": defaults.top_k,
        "repetition_penalty": defaults.repetition_penalty,
    }
    if overrides:
        payload.update(dict(overrides))
    return payload


@dataclass(frozen=True)
class QwenAdapter:
    spec: ModelSpec = field(default_factory=ModelSpec)

    def load_processor(self):
        return load_qwen_processor(self.spec)

    def load_model(self, pretrained_path: str | None = None):
        return load_qwen_model(self.spec, pretrained_path=pretrained_path)

    def generation_kwargs(
        self, processor: Any, overrides: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return build_qwen_generation_kwargs(
            processor, self.spec.generation_defaults, overrides=overrides
        )
