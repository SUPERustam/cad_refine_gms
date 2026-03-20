from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

from datasets import load_from_disk
import torch
from trl.trainer.grpo_config import GRPOConfig

from cad_rl.algorithms.frozen import (
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    FrozenRewardConfig,
    FrozenTrainingConfig,
    build_generation_kwargs,
    build_reward_function,
    compute_training_steps,
    configure_process_environment,
    create_optimizer_and_scheduler,
    create_processor,
    load_qwen_model,
    seed_everything,
)
from cad_rl.config import (
    ResolvedExperimentConfig,
    RunManifest,
    resolve_experiment_profile,
    to_serializable,
)
from cad_rl.data import fingerprint_prepared_dataset
from cad_rl.runtime import RunRegistry, materialize_resolved_run, select_checkpoint
from cad_rl.algorithms.grpo_trainer import TopSampleGRPOTrainer
from cad_rl.metrics.async_metrics import close_pool, init_pool


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _torch_dtype(name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)


def _build_grpo_config(mapping: Mapping[str, Any]) -> GRPOConfig:
    return GRPOConfig(**dict(mapping))


def _build_grpo_config_from_resolved(resolved: ResolvedExperimentConfig) -> GRPOConfig:
    payload = to_serializable(resolved.trainer)
    payload["importance_sampling_level"] = resolved.algorithm.importance_sampling_mode
    payload["loss_type"] = resolved.algorithm.loss_mode
    return _build_grpo_config(payload)


def _build_training_config(resolved: ResolvedExperimentConfig) -> FrozenTrainingConfig:
    trainer_kwargs = dict(resolved.algorithm.trainer_kwargs)
    runtime = dict(resolved.runtime)
    return FrozenTrainingConfig(
        sft_path=resolved.model.sft_checkpoint or resolved.model.base_checkpoint,
        clip_cov=bool(trainer_kwargs.get("clip_cov", False)),
        top_samples=int(trainer_kwargs.get("top_samples", 4)),
        resume_ckpt_path=str(runtime.get("resume_path", "")),
        scheduler=resolved.algorithm.scheduler_policy,
        scheduler_training_steps=int(runtime.get("scheduler_training_steps", 200000)),
    )


def _build_reward_config(resolved: ResolvedExperimentConfig) -> FrozenRewardConfig:
    return FrozenRewardConfig(**dict(resolved.algorithm.reward_config))


def _build_run_manifest(
    run_id: str, resolved: ResolvedExperimentConfig, fingerprint: str | None
) -> RunManifest:
    runtime = dict(resolved.runtime)
    return RunManifest(
        run_id=run_id,
        resolved_config_path="resolved_config.json",
        git_revision=runtime.get("git_revision"),
        seed=int(runtime.get("seed", DEFAULT_SEED)),
        dataset_fingerprint=fingerprint,
        checkpoint_inventory=(),
        latest_checkpoint=None,
        comet_experiment_id=runtime.get("comet_experiment_id"),
        task_id=resolved.task.task_id,
        model_id=resolved.model.base_checkpoint,
        algorithm_id=resolved.algorithm.trainer_type,
        machine_profile_id=resolved.machine.profile_id,
    )


def build_trainer(
    grpo_args,
    reward_args: FrozenRewardConfig,
    training_args: FrozenTrainingConfig,
    dataset_path: str,
    *,
    processor_name: str = DEFAULT_MODEL_ID,
    processor_kwargs: Mapping[str, Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
):
    configure_process_environment()
    seed_everything()
    init_pool(reward_args.pool_size)

    processor = create_processor(processor_name, **dict(processor_kwargs or {}))
    model = load_qwen_model(training_args.sft_path, **dict(model_kwargs or {}))
    dataset = load_from_disk(dataset_path)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=grpo_args.learning_rate,
        scheduler_name=training_args.scheduler,
        num_training_steps=training_args.scheduler_training_steps,
    )
    reward_fn = build_reward_function(reward_args)

    grpo_args.generation_kwargs = build_generation_kwargs(processor)
    grpo_args.steps_per_generation = grpo_args.gradient_accumulation_steps
    grpo_args.max_steps = compute_training_steps(grpo_args, len(dataset))
    grpo_args.num_train_epochs = 0

    trainer = TopSampleGRPOTrainer(
        clip_cov=training_args.clip_cov,
        top_samples=training_args.top_samples,
        model=model,
        processing_class=processor,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        args=grpo_args,
        optimizers=(optimizer, scheduler),
    )
    return trainer


def build_trainer_from_resolved(
    resolved: ResolvedExperimentConfig,
    grpo_args: GRPOConfig,
    dataset_split: str = "train",
):
    for key, value in resolved.machine.environment.items():
        os.environ[str(key)] = str(value)
    if resolved.machine.world_size is not None:
        os.environ.setdefault("WORLD_SIZE", str(resolved.machine.world_size))
    if resolved.machine.cache_dir:
        os.environ.setdefault("HF_HOME", resolved.machine.cache_dir)

    training_args = _build_training_config(resolved)
    reward_args = _build_reward_config(resolved)
    processor_name = resolved.model.processor_name or resolved.model.base_checkpoint
    processor_kwargs = dict(resolved.model.processor_kwargs)
    model_kwargs = {
        "torch_dtype": _torch_dtype(
            dict(resolved.extras).get(
                "torch_dtype", getattr(resolved.model, "torch_dtype", "bfloat16")
            )
        ),
        "attn_implementation": dict(resolved.extras).get(
            "attn_implementation", "flash_attention_2"
        ),
        "trust_remote_code": resolved.model.trust_remote_code,
    }
    return build_trainer(
        grpo_args=grpo_args,
        reward_args=reward_args,
        training_args=training_args,
        dataset_path=resolved.task.prepared_datasets[dataset_split],
        processor_name=processor_name,
        processor_kwargs=processor_kwargs,
        model_kwargs=model_kwargs,
    )


def train_from_resolved_config(
    resolved: ResolvedExperimentConfig,
    grpo_args: GRPOConfig,
    *,
    resume_reference: str | int | None = None,
):
    machine_root = Path(resolved.machine.run_root)
    run_id = str(
        resolved.runtime.get("run_id", f"{resolved.profile_id}_{_utc_stamp()}")
    )
    run_dir = machine_root / run_id
    registry = RunRegistry(machine_root)
    fingerprint = None
    dataset_path = resolved.task.prepared_datasets.get("train")
    if dataset_path:
        try:
            fingerprint = fingerprint_prepared_dataset(dataset_path).fingerprint
        except Exception:
            fingerprint = None
    manifest = _build_run_manifest(run_id, resolved, fingerprint)
    materialize_resolved_run(run_dir, resolved, manifest)

    trainer = build_trainer_from_resolved(resolved, grpo_args)
    resume_from_checkpoint: bool | str = False
    if resume_reference is not None:
        checkpoint = select_checkpoint(run_id, registry, resume_reference)
        resume_from_checkpoint = checkpoint.path
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        close_pool()
    return {"run_id": run_id, "run_dir": str(run_dir), "manifest": manifest}


def load_resolved_and_grpo(
    experiment_path: str | Path,
) -> tuple[ResolvedExperimentConfig, GRPOConfig]:
    resolved = resolve_experiment_profile(
        experiment_path, profiles_root=Path(experiment_path).parent
    )
    grpo_args = _build_grpo_config_from_resolved(resolved)
    return resolved, grpo_args


def export_training_contract(config: ResolvedExperimentConfig) -> dict[str, Any]:
    return {
        "task": asdict(config.task),
        "model": asdict(config.model),
        "algorithm": to_serializable(config.algorithm),
        "trainer": to_serializable(config.trainer),
        "machine_profile": getattr(config.machine, "profile_id", None),
        "runtime": dict(config.runtime),
    }
