from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

from cad_rl.config import RunConfig, resolve_run_config, to_serializable
from cad_rl.data import fingerprint_prepared_dataset
from cad_rl.runtime import (
    RunManifest,
    RunRegistry,
    materialize_resolved_run,
    select_checkpoint,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_grpo_config(mapping: Mapping[str, Any]):
    from trl.trainer.grpo_config import GRPOConfig

    return GRPOConfig(**dict(mapping))


def build_grpo_config_from_resolved(resolved: RunConfig):
    payload = to_serializable(resolved.train)
    payload["importance_sampling_level"] = resolved.train.importance_sampling_mode
    payload["loss_type"] = resolved.train.loss_mode
    return _build_grpo_config(payload)


def _build_training_config(resolved: RunConfig):
    from cad_rl.grpo import FrozenTrainingConfig

    return FrozenTrainingConfig(
        sft_path=resolved.model.sft_checkpoint or resolved.model.base_checkpoint,
        clip_cov=bool(resolved.train.trainer_kwargs.get("clip_cov", False)),
        top_samples=int(resolved.train.trainer_kwargs.get("top_samples", 4)),
        resume_ckpt_path=resolved.runtime.resume_path,
        scheduler=resolved.train.scheduler_policy,
        scheduler_training_steps=resolved.runtime.scheduler_training_steps,
    )


def _build_reward_config(resolved: RunConfig):
    from cad_rl.grpo import FrozenRewardConfig

    return FrozenRewardConfig(**dict(resolved.train.reward_config))


def _build_run_manifest(
    run_id: str, resolved: RunConfig, fingerprint: str | None
) -> RunManifest:
    return RunManifest(
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
        trainer_id=resolved.train.trainer_type,
        machine_profile_id=resolved.system.profile_id,
    )


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
    from datasets import load_from_disk

    from cad_rl.grpo import (
        TopSampleGRPOTrainer,
        build_reward_function,
        close_pool,
        compute_training_steps,
        configure_process_environment,
        create_optimizer_and_scheduler,
        init_pool,
        seed_everything,
    )
    from cad_rl.modeling import (
        build_generation_kwargs,
        create_processor,
        load_qwen_model,
    )

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
    trainer._cad_close_pool = close_pool  # type: ignore[attr-defined]
    return trainer


def build_trainer_from_resolved(
    resolved: RunConfig,
    grpo_args,
    dataset_split: str | None = None,
):
    from cad_rl.modeling import resolve_torch_dtype

    for key, value in resolved.system.environment.items():
        os.environ[str(key)] = str(value)
    if resolved.system.world_size is not None:
        os.environ.setdefault("WORLD_SIZE", str(resolved.system.world_size))
    if resolved.system.cache_dir:
        os.environ.setdefault("HF_HOME", resolved.system.cache_dir)

    active_split = dataset_split or resolved.runtime.dataset_split
    training_args = _build_training_config(resolved)
    reward_args = _build_reward_config(resolved)
    processor_name = resolved.model.processor_name or resolved.model.base_checkpoint
    processor_kwargs = dict(resolved.model.processor_kwargs)
    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(resolved.model.torch_dtype),
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


def resolve_training_config(
    experiment_path: str | Path,
    *,
    system: str | Path | None = None,
) -> RunConfig:
    return resolve_run_config(
        experiment_path, profiles_root=Path(experiment_path).parents[1], system=system
    )


def train_from_resolved_config(
    resolved: RunConfig,
    *,
    resume_reference: str | int | None = None,
):
    from cad_rl.grpo import close_pool

    machine_root = Path(resolved.system.run_root)
    run_id = str(
        resolved.runtime.run_id
        or f"{resolved.experiment_id}_{resolved.stage}_{_utc_stamp()}"
    )
    run_dir = machine_root / run_id
    registry = RunRegistry(machine_root)
    fingerprint = None
    dataset_path = resolved.data.prepared_datasets.get(resolved.runtime.dataset_split)
    if dataset_path:
        try:
            fingerprint = fingerprint_prepared_dataset(dataset_path).fingerprint
        except Exception:
            fingerprint = None
    manifest = _build_run_manifest(run_id, resolved, fingerprint)
    materialize_resolved_run(run_dir, resolved, manifest)

    grpo_args = build_grpo_config_from_resolved(resolved)
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


def export_training_contract(config: RunConfig) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "config_path": config.config_path,
        "task": asdict(config.task),
        "data": to_serializable(config.data),
        "model": asdict(config.model),
        "train": to_serializable(config.train),
        "runtime": config.runtime.to_dict(),
        "system": to_serializable(config.system),
    }
