from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RepeatSampler, nanmax, nanmin, nanstd
from trl.trainer.utils import pad
from transformers import Trainer, get_constant_schedule, get_cosine_schedule_with_warmup

from cad_rl.execution import close_pool, execute_generated_codes, init_pool
from cad_rl.metrics import (
    compute_metrics_from_execution,
    reward_from_auc_blend,
    reward_from_metrics,
)
from cad_rl.utils import _maybe_print_sample

DEFAULT_SEED = 16
_DEFAULT_VAR_NAME = os.getenv("METRICS_VAR_NAME", "result")


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


def compute_training_steps(grpo_args: GRPOConfig, dataset_length: int) -> int:
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


def get_reward_function(
    failure_reward,
    iou_coef=10,
    cd_coef=0,
    auc_coef=0,
    aoc_gms_coef=0,
    nc_params=None,
    mode="10_iou",
    print_every=50,
    var_name=None,
):
    def combined_reward(completions, mesh_path, trainer_state=None, **kwargs):
        vn = var_name or _DEFAULT_VAR_NAME
        rewards = []
        execution_results = execute_generated_codes(
            completions,
            mesh_path,
            var_name=vn,
        )
        pred_metrics = [
            compute_metrics_from_execution(
                result,
                n_points=8192,
                normal_params=nc_params,
            )
            for result in execution_results
        ]
        for metrics in pred_metrics:
            if metrics is None or metrics.get("iou") is None:
                reward = failure_reward
            else:
                use_aoc_gms = (
                    nc_params and nc_params.get("get_aoc_gms") and aoc_gms_coef > 0
                )
                if use_aoc_gms:
                    auc_gms_value = metrics.get("auc_gms")
                    if auc_gms_value is None:
                        reward = failure_reward
                    else:
                        gms_reward = reward_from_metrics(
                            metrics.get("cd"),
                            metrics.get("iou"),
                            auc_gms=auc_gms_value,
                            mode="10_auc_gms",
                        )
                        reward = float(
                            np.clip((aoc_gms_coef / 10.0) * gms_reward, -10.0, 10.0)
                        )
                else:
                    reward = reward_from_metrics(
                        metrics.get("cd"),
                        metrics.get("iou"),
                        auc=metrics.get("auc"),
                        mode=mode,
                    )
            if not math.isfinite(reward):
                reward = failure_reward
            rewards.append(float(reward))

        top_idx = rewards.index(max(rewards))
        _maybe_print_sample(
            completions[top_idx],
            mesh_path[top_idx],
            step=trainer_state.global_step,
            every=print_every,
        )
        return rewards

    return combined_reward


def adv_select_top_samples(self, inputs, num_generations: int, top_samples: int):
    completion_ids = inputs["completion_ids"]
    group_size = self.num_generations
    batch_size = completion_ids.size(0)
    assert batch_size % group_size == 0, (
        f"inputs must be B * num_generations, got B : {batch_size} G : {group_size}"
    )
    num_prompts = batch_size // group_size
    device = completion_ids.device

    advantages = inputs["advantages"].view(batch_size, -1).squeeze(-1)
    abs_advantages = advantages.abs().view(num_prompts, group_size)
    _, top_indices = torch.topk(abs_advantages, top_samples, dim=1)
    row_indices = (
        torch.arange(num_prompts, device=device).unsqueeze(1).expand(-1, top_samples)
    )
    flat = (row_indices * group_size + top_indices).reshape(-1)

    def take(value):
        return value[flat] if value is not None else None

    new_inputs = {}
    for key, value in inputs.items():
        if value is None:
            new_inputs[key] = None
            continue
        if key in {"pixel_values", "pixel_values_videos"}:
            packed = value.size(0) // batch_size
            depth = value.size(1)
            new_inputs[key] = value.view(batch_size, -1, depth)[flat].reshape(-1, depth)
        else:
            new_inputs[key] = take(value)

    return new_inputs


def cppo_compute_loss(
    self, model, inputs, return_outputs=False, clip_cov=False, has_videos=False
):
    completion_ids_full = inputs["completion_ids"]
    logits_to_keep = completion_ids_full.size(1)
    selected_inputs = self.select_top_samples(
        inputs, num_generations=self.num_generations, top_samples=self.top_samples
    )

    prompt_ids, prompt_mask = (
        selected_inputs["prompt_ids"],
        selected_inputs["prompt_mask"],
    )
    completion_ids, completion_mask = (
        selected_inputs["completion_ids"],
        selected_inputs["completion_mask"],
    )
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    per_token_logps, entropies = self._get_per_token_logps_and_entropies(
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy=True,
        pixel_values=selected_inputs.get("pixel_values"),
        image_grid_thw=selected_inputs.get("image_grid_thw"),
        pixel_attention_mask=selected_inputs.get("pixel_attention_mask"),
        image_sizes=selected_inputs.get("image_sizes"),
    )

    advantages = selected_inputs["advantages"]
    old_per_token_logps = selected_inputs.get("old_per_token_logps")
    old_per_token_logps = (
        per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
    )

    log_ratio = per_token_logps - old_per_token_logps
    if self.importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif self.importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * completion_mask).sum(
            -1
        ) / completion_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {self.importance_sampling_level}."
        )

    coef_1 = torch.exp(log_importance_weights)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)

    if clip_cov:
        cov_lb = 1
        cov_hb = 5
        select_ratio = 2e-4
        covs = (per_token_logps - per_token_logps.mean()) * (
            advantages.unsqueeze(1) - advantages.mean()
        )
        mask = (covs > cov_lb) & (covs < cov_hb)
        all_idx = torch.nonzero(mask).reshape(-1)
        select_num = int(select_ratio * per_token_logps.numel())
        if all_idx.numel() >= select_num > 0:
            perm = torch.randperm(all_idx.numel(), device=all_idx.device)
            clip_idx = all_idx[perm[:select_num]]
            per_token_loss1[clip_idx] = per_token_loss1[clip_idx].detach()
            per_token_loss2[clip_idx] = per_token_loss2[clip_idx].detach()

    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    if self.loss_type == "grpo":
        loss = (
            (per_token_loss * completion_mask).sum(-1)
            / completion_mask.sum(-1).clamp(min=1.0)
        ).mean()
        loss = loss / self.current_gradient_accumulation_steps
    else:
        loss = (per_token_loss * completion_mask).sum() / (
            per_token_loss.size(0) * self.max_completion_length
        )
        loss = loss / self.current_gradient_accumulation_steps

    mode = "train" if self.model.training else "eval"
    completion_token_count = completion_mask.sum().clamp(min=1.0)

    def masked_batch_mean(x):
        if x.shape[1] == 1:
            return x.mean()
        return (x * completion_mask).sum() / completion_token_count

    mean_entropy = masked_batch_mean(entropies)
    self._metrics[mode]["entropy"].append(
        self.accelerator.gather(mean_entropy).nanmean().item()
    )
    is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped
    low_clip = masked_batch_mean(is_low_clipped.float())
    high_clip = masked_batch_mean(is_high_clipped.float())
    clip_ratio = masked_batch_mean(is_region_clipped.float())
    gathered_low_clip = self.accelerator.gather(low_clip)
    self._metrics[mode]["clip_ratio/low_mean"].append(
        gathered_low_clip.nanmean().item()
    )
    self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
    gathered_high_clip = self.accelerator.gather(high_clip)
    self._metrics[mode]["clip_ratio/high_mean"].append(
        gathered_high_clip.nanmean().item()
    )
    self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
    gathered_clip_ratio = self.accelerator.gather(clip_ratio)
    self._metrics[mode]["clip_ratio/region_mean"].append(
        gathered_clip_ratio.nanmean().item()
    )
    return loss


class TopSampleGRPOTrainer(GRPOTrainer):
    def __init__(self, top_samples, clip_cov=False, **kwargs):
        super().__init__(**kwargs)
        self.top_samples = top_samples
        self.clip_cov = clip_cov

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.model.gradient_checkpointing_enable()
        return cppo_compute_loss(
            self, model, inputs, return_outputs=return_outputs, clip_cov=self.clip_cov
        )

    def select_top_samples(self, inputs, num_generations: int, top_samples):
        return adv_select_top_samples(self, inputs, num_generations, top_samples)

    def _generate_and_score_completions(self, inputs):
        if getattr(self, "_cached_step", None) == self.state.global_step:
            return self._cached_rollout
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        repeated_inputs = self.repeat_inputs(inputs)
        inputs = repeated_inputs
        prompts_text = [x["prompt"] for x in inputs]
        original_prompts = copy.deepcopy(prompts_text)

        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)
                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    ordered_set_of_images = (
                        all_images[:: self.num_generations] if has_images else None
                    )
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts_text),
                    (self.accelerator.process_index + 1) * len(prompts_text),
                )
                completion_ids = completion_ids[process_slice]

            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                nullcontext(),
            ):
                unwrapped_model.gradient_checkpointing_disable()
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = (
                    prompt_ids,
                    prompt_mask,
                )
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_ids_list = [
            [token.item() for token, mask_row in zip(row, mask) if mask_row]
            for row, mask in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        rewards_per_func = self._calculate_rewards(
            inputs, original_prompts, completions, completion_ids_list
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
            "all_process_advantages": all_process_advantages,
            "completion_lengths": completion_lengths,
        }


__all__ = [
    "DEFAULT_SEED",
    "FrozenRewardConfig",
    "FrozenTrainingConfig",
    "GRPOConfig",
    "TopSampleGRPOTrainer",
    "build_reward_function",
    "compute_training_steps",
    "configure_process_environment",
    "create_optimizer_and_scheduler",
    "reward_config_from_mapping",
    "seed_everything",
    "serialize_dataclass",
    "training_config_from_mapping",
]
