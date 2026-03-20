from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptRenderSpec:
    template_id: str = "qwen2vl_single_image_chat"
    user_role: str = "user"
    content_type: str = "image"
    include_image_payload: bool = False
    add_generation_prompt: bool = True
    padding_side: str = "left"
    resized_width: int = 14 * 17 * 2
    resized_height: int = 14 * 17 * 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptRenderSpec":
        return cls(**dict(data))

    def render_prompt(self, processor: Any, image: Any) -> str:
        content: dict[str, Any] = {"type": self.content_type}
        if self.include_image_payload:
            content["image"] = image
        message = [
            {
                "role": self.user_role,
                "content": [content],
            }
        ]
        return processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )


@dataclass(frozen=True)
class GenerationDefaults:
    bad_words: tuple[str, ...] = (
        "<|image_pad|>",
        "<|vision_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
    )
    max_new_tokens: int = 4000
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bad_words"] = list(self.bad_words)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationDefaults":
        payload = dict(data)
        payload["bad_words"] = tuple(payload.get("bad_words", ()))
        return cls(**payload)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    processor_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    family_id: str = "qwen_vl"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    generation_defaults: GenerationDefaults = field(default_factory=GenerationDefaults)
    processor_render: PromptRenderSpec = field(default_factory=PromptRenderSpec)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation_defaults"] = self.generation_defaults.to_dict()
        data["processor_render"] = self.processor_render.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        payload = dict(data)
        payload["generation_defaults"] = GenerationDefaults.from_dict(
            payload.get("generation_defaults", {})
        )
        payload["processor_render"] = PromptRenderSpec.from_dict(
            payload.get("processor_render", {})
        )
        return cls(**payload)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prepared_hf_dataset_splits: Mapping[str, str]
    raw_dataset_root: str | None = None
    raw_dataset_pickle: str | None = None
    prompt_render: PromptRenderSpec = field(default_factory=PromptRenderSpec)
    output_var_name: str = "result"
    normalization_mode: str = "fixed"
    default_eval_tiers: tuple[str, ...] = ("quick", "standard", "full")
    default_eval_split: str = "val"
    default_model: ModelSpec = field(default_factory=ModelSpec)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prepared_hf_dataset_splits"] = dict(self.prepared_hf_dataset_splits)
        data["prompt_render"] = self.prompt_render.to_dict()
        data["default_eval_tiers"] = list(self.default_eval_tiers)
        data["default_model"] = self.default_model.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        payload = dict(data)
        payload["prepared_hf_dataset_splits"] = dict(
            payload.get("prepared_hf_dataset_splits", {})
        )
        payload["prompt_render"] = PromptRenderSpec.from_dict(
            payload.get("prompt_render", {})
        )
        payload["default_eval_tiers"] = tuple(
            payload.get("default_eval_tiers", ("quick", "standard", "full"))
        )
        payload["default_model"] = ModelSpec.from_dict(payload.get("default_model", {}))
        return cls(**payload)

    def prepared_dataset_path(self, split: str = "train") -> str:
        return self.prepared_hf_dataset_splits[split]

    def render_prompt(self, processor: Any, image: Any) -> str:
        return self.prompt_render.render_prompt(processor, image)


def default_model_spec() -> ModelSpec:
    return ModelSpec()


def default_task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="cadevolve_qwen_v1",
        prepared_hf_dataset_splits={
            "train": "/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_fixed",
            "train_dp_f360": "/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_deepcadf360",
        },
        raw_dataset_root="/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/deepcad_fusion_train",
        raw_dataset_pickle="/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/deepcad_fusion_train/pkls/train_small.pkl",
        prompt_render=PromptRenderSpec(),
        output_var_name="result",
        normalization_mode="fixed",
        default_eval_tiers=("quick", "standard", "full"),
        default_eval_split="val",
        default_model=ModelSpec(),
    )
