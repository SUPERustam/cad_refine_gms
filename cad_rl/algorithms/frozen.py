import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.optim as optim
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
)

from cad_rl.algorithms.rewards import get_reward_function

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_SEED = 16


@dataclass
class FrozenRewardConfig:
    failure_reward: float = -10.0
    iou_coef: float = 10.0
    cd_coef: float = 0.0
    auc_coef: float = 0.0
    aoc_gms_coef: float = 0.0
    get_nc: bool = False
    nc_n_points: int = 16384
    nc_tol: int = 5
    print_sample_steps: int = 25
    pool_size: int = 16
    r_mode: str = "10_iou"
    get_aoc_gms: bool = False
    aoc_gms_n_points: int = 8192
    aoc_gms_n_angles: int = 125
    aoc_gms_rel_tol: float = 0.05
    aoc_gms_cube_trick: bool = True
    aoc_gms_pc_cache_enable: bool = False
    aoc_gms_upper_bound_tol_rt: int = 25
    aoc_gms_autofix_sampling: bool = False

    def nc_params(self) -> dict[str, Any]:
        return {
            "get_nc": self.get_nc,
            "n_points": self.nc_n_points,
            "tol": self.nc_tol,
            "get_aoc_gms": self.get_aoc_gms,
            "aoc_gms_n_points": self.aoc_gms_n_points,
            "aoc_gms_rel_tol": self.aoc_gms_rel_tol,
            "aoc_gms_n_angles": self.aoc_gms_n_angles,
            "aoc_gms_cube_trick": self.aoc_gms_cube_trick,
            "aoc_gms_pc_cache_enable": self.aoc_gms_pc_cache_enable,
            "aoc_gms_upper_bound_tol_rt": self.aoc_gms_upper_bound_tol_rt,
            "aoc_gms_autofix_sampling": self.aoc_gms_autofix_sampling,
        }


@dataclass
class FrozenTrainingConfig:
    sft_path: str
    clip_cov: bool = False
    top_samples: int = 4
    resume_ckpt_path: str = ""
    scheduler: str = "constant"
    scheduler_training_steps: int = 200000


def configure_process_environment() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")
    os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_processor(model_id: str = DEFAULT_MODEL_ID, **processor_kwargs):
    kwargs = {
        "trust_remote_code": True,
        "resized_width": 14 * 17 * 2,
        "resized_height": 14 * 17 * 4,
        "padding_side": "left",
    }
    kwargs.update(processor_kwargs)
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def load_qwen_model(model_path: str, **model_kwargs):
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


def create_optimizer_and_scheduler(
    model, learning_rate: float, scheduler_name: str, num_training_steps: int
):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = get_constant_schedule(optimizer)
    return optimizer, scheduler


def build_reward_function(config: FrozenRewardConfig):
    return get_reward_function(
        failure_reward=config.failure_reward,
        iou_coef=config.iou_coef,
        cd_coef=config.cd_coef,
        auc_coef=config.auc_coef,
        aoc_gms_coef=config.aoc_gms_coef,
        nc_params=config.nc_params(),
        mode=config.r_mode,
        print_every=config.print_sample_steps,
    )


def build_generation_kwargs(processor) -> dict[str, Any]:
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


def compute_training_steps(grpo_args, dataset_length: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        grpo_args.num_train_epochs
        * dataset_length
        * grpo_args.num_iterations
        // (
            grpo_args.gradient_accumulation_steps
            * grpo_args.per_device_train_batch_size
            * world_size
        )
    )


def resolve_resume_path(training: FrozenTrainingConfig) -> bool | str:
    if training.resume_ckpt_path in {"", "False"}:
        return False
    return training.resume_ckpt_path


def reward_config_from_mapping(raw: dict[str, Any]) -> FrozenRewardConfig:
    fields = FrozenRewardConfig.__dataclass_fields__
    return FrozenRewardConfig(**{key: raw[key] for key in raw if key in fields})


def training_config_from_mapping(raw: dict[str, Any]) -> FrozenTrainingConfig:
    fields = FrozenTrainingConfig.__dataclass_fields__
    return FrozenTrainingConfig(**{key: raw[key] for key in raw if key in fields})


def serialize_dataclass(instance) -> dict[str, Any]:
    return asdict(instance)
