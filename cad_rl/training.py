from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

import cad_rl.config
import cad_rl.data
import cad_rl.modeling
import cad_rl.runtime

try:  # pragma: no cover - optional training dependency
    import cad_rl.grpo
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.grpo = None  # type: ignore[attr-defined]

try:  # pragma: no cover - optional training dependency
    from datasets import load_from_disk
except Exception:  # pragma: no cover - lightweight environments
    load_from_disk = None

try:  # pragma: no cover - optional training dependency
    from trl.trainer.grpo_config import GRPOConfig
except Exception:  # pragma: no cover - lightweight environments
    GRPOConfig = None


class TrainingConfigError(ValueError):
    pass


_TRAINER_CLASS_NAME = "TopSampleGRPOTrainer"
_GRPO_CONFIG_KEYS = (
    "output_dir",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_completion_length",
    "log_completions",
    "logging_steps",
    "num_generations",
    "generation_batch_size",
    "report_to",
    "run_name",
    "num_train_epochs",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "temperature",
    "top_p",
    "top_k",
    "importance_sampling_level",
    "loss_type",
    "epsilon",
    "num_iterations",
    "scale_rewards",
    "learning_rate",
    "use_vllm",
    "vllm_server_port",
    "bf16",
    "gradient_checkpointing",
    "remove_unused_columns",
    "ddp_find_unused_parameters",
    "beta",
    "weight_decay",
)
_REWARD_KEYS = (
    "failure_reward",
    "iou_coef",
    "cd_coef",
    "auc_coef",
    "aoc_gms_coef",
    "get_nc",
    "nc_n_points",
    "nc_tol",
    "print_sample_steps",
    "pool_size",
    "r_mode",
    "get_aoc_gms",
    "aoc_gms_n_points",
    "aoc_gms_n_angles",
    "aoc_gms_rel_tol",
    "aoc_gms_cube_trick",
    "aoc_gms_pc_cache_enable",
    "aoc_gms_upper_bound_tol_rt",
    "aoc_gms_autofix_sampling",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _required_attr(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise TrainingConfigError(f"Missing required training field: train.{name}")
    value = getattr(obj, name)
    if value is None:
        raise TrainingConfigError(f"Missing required training field: train.{name}")
    return value


def _optional_attr(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        return None
    return getattr(obj, name)


def _mapping_from_section(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, Mapping):
        return dict(section)
    if is_dataclass(section):
        return cad_rl.config.to_serializable(section)
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }
    raise TrainingConfigError(
        f"Unsupported training reward section type: {type(section)!r}"
    )


def _build_grpo_config_payload(train: Any) -> dict[str, Any]:
    payload = {
        "output_dir": _required_attr(train, "output_dir"),
        "per_device_train_batch_size": int(
            _required_attr(train, "per_device_train_batch_size")
        ),
        "gradient_accumulation_steps": int(
            _required_attr(train, "gradient_accumulation_steps")
        ),
        "max_completion_length": int(_required_attr(train, "max_completion_length")),
        "log_completions": bool(_required_attr(train, "log_completions")),
        "logging_steps": int(_required_attr(train, "logging_steps")),
        "num_generations": int(_required_attr(train, "num_generations")),
        "generation_batch_size": int(_required_attr(train, "generation_batch_size")),
        "report_to": tuple(str(item) for item in _required_attr(train, "report_to")),
        "run_name": str(_required_attr(train, "run_name")),
        "num_train_epochs": int(_required_attr(train, "num_train_epochs")),
        "save_strategy": str(_required_attr(train, "save_strategy")),
        "save_steps": int(_required_attr(train, "save_steps")),
        "save_total_limit": _optional_attr(train, "save_total_limit"),
        "temperature": float(_required_attr(train, "temperature")),
        "top_p": float(_required_attr(train, "top_p")),
        "top_k": int(_required_attr(train, "top_k")),
        "importance_sampling_level": str(
            _required_attr(train, "importance_sampling_level")
        ),
        "loss_type": str(_required_attr(train, "loss_type")),
        "epsilon": float(_required_attr(train, "epsilon")),
        "num_iterations": int(_required_attr(train, "num_iterations")),
        "scale_rewards": bool(_required_attr(train, "scale_rewards")),
        "learning_rate": float(_required_attr(train, "learning_rate")),
        "use_vllm": bool(_required_attr(train, "use_vllm")),
        "vllm_server_port": _optional_attr(train, "vllm_server_port"),
        "bf16": bool(_required_attr(train, "bf16")),
        "gradient_checkpointing": bool(_required_attr(train, "gradient_checkpointing")),
        "remove_unused_columns": bool(_required_attr(train, "remove_unused_columns")),
        "ddp_find_unused_parameters": bool(
            _required_attr(train, "ddp_find_unused_parameters")
        ),
        "beta": float(_required_attr(train, "beta")),
        "weight_decay": float(_required_attr(train, "weight_decay")),
    }
    unknown = set(payload) - set(_GRPO_CONFIG_KEYS)
    if unknown:
        raise TrainingConfigError(f"Unexpected GRPO config keys: {sorted(unknown)!r}")
    return payload


def _build_grpo_config(train: Any):
    if GRPOConfig is None:
        raise RuntimeError("trl is required to build GRPO config")
    return GRPOConfig(**_build_grpo_config_payload(train))


def _build_reward_config_payload(train: Any) -> dict[str, Any]:
    raw = _mapping_from_section(getattr(train, "reward", None))
    unknown = set(raw) - set(_REWARD_KEYS)
    if unknown:
        raise TrainingConfigError(f"Unexpected reward config keys: {sorted(unknown)!r}")
    return {key: raw[key] for key in _REWARD_KEYS if key in raw}


def _build_reward_config(train: Any):
    if cad_rl.grpo is None:
        raise RuntimeError("cad_rl.grpo dependencies are not available")
    return cad_rl.grpo.FrozenRewardConfig(**_build_reward_config_payload(train))


def _build_training_config_payload(resolved: cad_rl.config.RunConfig) -> dict[str, Any]:
    return {
        "sft_path": resolved.model.sft_checkpoint or resolved.model.base_checkpoint,
        "clip_cov": bool(_required_attr(resolved.train, "clip_cov")),
        "top_samples": int(_required_attr(resolved.train, "top_samples")),
        "resume_ckpt_path": resolved.runtime.resume_path,
        "scheduler": str(_required_attr(resolved.train, "scheduler")),
        "scheduler_training_steps": int(resolved.runtime.scheduler_training_steps),
    }


def _build_training_config(resolved: cad_rl.config.RunConfig):
    if cad_rl.grpo is None:
        raise RuntimeError("cad_rl.grpo dependencies are not available")
    return cad_rl.grpo.FrozenTrainingConfig(**_build_training_config_payload(resolved))


def build_trainer(
    grpo_args,
    reward_args,
    training_args,
    dataset_path: str,
    *,
    processor_name: str,
    processor_kwargs: Mapping[str, Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
):
    if load_from_disk is None:
        raise RuntimeError("datasets is required to build the trainer")
    if cad_rl.grpo is None:
        raise RuntimeError("cad_rl.grpo dependencies are not available")
    cad_rl.grpo.configure_process_environment()
    cad_rl.grpo.seed_everything()
    cad_rl.grpo.init_pool(reward_args.pool_size)

    processor = cad_rl.modeling.create_processor(
        processor_name, **dict(processor_kwargs or {})
    )
    model = cad_rl.modeling.load_qwen_model(
        training_args.sft_path, **dict(model_kwargs or {})
    )
    dataset = load_from_disk(dataset_path)
    optimizer, scheduler = cad_rl.grpo.create_optimizer_and_scheduler(
        model=model,
        learning_rate=grpo_args.learning_rate,
        scheduler_name=training_args.scheduler,
        num_training_steps=training_args.scheduler_training_steps,
    )
    reward_fn = cad_rl.grpo.build_reward_function(reward_args)

    grpo_args.generation_kwargs = cad_rl.modeling.build_generation_kwargs(processor)
    grpo_args.steps_per_generation = grpo_args.gradient_accumulation_steps
    grpo_args.max_steps = cad_rl.grpo.compute_training_steps(grpo_args, len(dataset))
    grpo_args.num_train_epochs = 0

    trainer = cad_rl.grpo.TopSampleGRPOTrainer(
        clip_cov=training_args.clip_cov,
        top_samples=training_args.top_samples,
        model=model,
        processing_class=processor,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        args=grpo_args,
        optimizers=(optimizer, scheduler),
    )
    trainer._cad_close_pool = cad_rl.grpo.close_pool  # type: ignore[attr-defined]
    return trainer


def build_trainer_from_resolved(
    resolved: cad_rl.config.RunConfig,
    grpo_args,
    dataset_split: str | None = None,
):
    for key, value in resolved.system.environment.items():
        os.environ[str(key)] = str(value)
    if resolved.system.world_size is not None:
        os.environ.setdefault("WORLD_SIZE", str(resolved.system.world_size))
    if resolved.system.cache_dir:
        os.environ.setdefault("HF_HOME", resolved.system.cache_dir)

    active_split = dataset_split or resolved.runtime.dataset_split
    training_args = _build_training_config(resolved)
    reward_args = _build_reward_config(resolved.train)
    processor_name = resolved.model.processor_name or resolved.model.base_checkpoint
    processor_kwargs = dict(resolved.model.processor_kwargs)
    model_kwargs = {
        "torch_dtype": cad_rl.modeling.resolve_torch_dtype(resolved.model.torch_dtype),
        "attn_implementation": resolved.model.attn_implementation,
        "trust_remote_code": resolved.model.trust_remote_code,
    }
    return build_trainer(
        grpo_args=grpo_args,
        reward_args=reward_args,
        training_args=training_args,
        dataset_path=resolved.data.prepared_datasets[active_split],
        processor_name=processor_name,
        processor_kwargs=processor_kwargs,
        model_kwargs=model_kwargs,
    )


def train_from_resolved_config(
    resolved: cad_rl.config.RunConfig,
    *,
    resume_reference: str | int | None = None,
):
    if cad_rl.grpo is None:
        raise RuntimeError("cad_rl.grpo dependencies are not available")
    machine_root = Path(resolved.system.run_root)
    run_id = str(
        resolved.runtime.run_id
        or f"{resolved.experiment_id}_{resolved.stage}_{_utc_stamp()}"
    )
    run_dir = machine_root / run_id
    registry = cad_rl.runtime.RunRegistry(machine_root)
    fingerprint = None
    dataset_path = resolved.data.prepared_datasets.get(resolved.runtime.dataset_split)
    if dataset_path:
        try:
            fingerprint = cad_rl.data.fingerprint_prepared_dataset(
                dataset_path
            ).fingerprint
        except Exception:
            fingerprint = None
    manifest = cad_rl.runtime.RunManifest(
        run_id=run_id,
        resolved_config_path="resolved_config.json",
        git_revision=resolved.runtime.git_revision,
        seed=resolved.runtime.seed,
        dataset_fingerprint=fingerprint,
        checkpoint_inventory=(),
        latest_checkpoint=None,
        comet_experiment_id=resolved.runtime.comet_experiment_id,
        task_id=resolved.task.task_id,
        model_id=resolved.model.base_checkpoint,
        trainer_id=_TRAINER_CLASS_NAME,
        machine_profile_id=resolved.system.profile_id,
    )
    cad_rl.runtime.materialize_resolved_run(run_dir, resolved, manifest)

    grpo_args = _build_grpo_config(resolved.train)
    trainer = build_trainer_from_resolved(resolved, grpo_args)
    resume_from_checkpoint: bool | str = False
    if resume_reference is not None:
        checkpoint = cad_rl.runtime.select_checkpoint(
            run_id, registry, resume_reference
        )
        resume_from_checkpoint = checkpoint.path
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        cad_rl.grpo.close_pool()
    return {"run_id": run_id, "run_dir": str(run_dir), "manifest": manifest}
