from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, MutableMapping):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value.items())
    raise TypeError(f"Expected mapping, got {type(value)!r}")


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return [to_serializable(item) for item in sorted(value, key=str)]
    return value


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    prepared_datasets: Mapping[str, str] = field(default_factory=dict)
    prompt_profile_id: str = "default_prompt"
    render_profile_id: str = "default_render"
    output_var_name: str = "result"
    normalization_mode: str = "fixed"
    default_eval_suites: tuple[str, ...] = ("quick", "standard", "full")
    prompt_convention: str | None = None
    render_convention: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskSpec":
        payload = _dict(data)
        aliases = {
            "prepared_dataset_splits": "prepared_datasets",
            "output_variable_name": "output_var_name",
            "var_name": "output_var_name",
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)
        return cls(
            task_id=str(payload["task_id"]),
            prepared_datasets={
                str(key): str(value)
                for key, value in _dict(payload.get("prepared_datasets")).items()
            },
            prompt_profile_id=str(payload.get("prompt_profile_id", "default_prompt")),
            render_profile_id=str(payload.get("render_profile_id", "default_render")),
            output_var_name=str(payload.get("output_var_name", "result")),
            normalization_mode=str(payload.get("normalization_mode", "fixed")),
            default_eval_suites=_tuple(
                payload.get("default_eval_suites", ("quick", "standard", "full"))
            ),
            prompt_convention=payload.get("prompt_convention"),
            render_convention=payload.get("render_convention"),
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_family_adapter_id: str
    base_checkpoint: str
    sft_checkpoint: str | None = None
    processor_name: str | None = None
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generation_defaults: Mapping[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelSpec":
        payload = _dict(data)
        return cls(
            model_family_adapter_id=str(payload["model_family_adapter_id"]),
            base_checkpoint=str(payload["base_checkpoint"]),
            sft_checkpoint=None
            if payload.get("sft_checkpoint") is None
            else str(payload.get("sft_checkpoint")),
            processor_name=None
            if payload.get("processor_name") is None
            else str(payload.get("processor_name")),
            processor_kwargs=_dict(payload.get("processor_kwargs")),
            generation_defaults=_dict(payload.get("generation_defaults")),
            trust_remote_code=bool(payload.get("trust_remote_code", True)),
        )


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    trainer_type: str
    loss_mode: str
    importance_sampling_mode: str
    top_k_policy: str
    optimizer_policy: str
    scheduler_policy: str
    reward_config: Mapping[str, Any] = field(default_factory=dict)
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AlgorithmSpec":
        payload = _dict(data)
        aliases = {
            "top_samples_policy": "top_k_policy",
            "top_sample_policy": "top_k_policy",
            "reward": "reward_config",
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)
        return cls(
            trainer_type=str(payload["trainer_type"]),
            loss_mode=str(payload["loss_mode"]),
            importance_sampling_mode=str(payload["importance_sampling_mode"]),
            top_k_policy=str(payload["top_k_policy"]),
            optimizer_policy=str(payload["optimizer_policy"]),
            scheduler_policy=str(payload["scheduler_policy"]),
            reward_config=_dict(payload.get("reward_config")),
            trainer_kwargs=_dict(payload.get("trainer_kwargs")),
        )


@dataclass(frozen=True, slots=True)
class MachineProfile:
    profile_id: str
    run_root: str
    cache_dir: str | None = None
    checkpoints_root: str = "checkpoints"
    artifact_root: str = "artifacts"
    logs_root: str = "logs"
    vllm_port: int | None = None
    world_size: int | None = None
    cpu_workers: int = 4
    vllm_placement_policy: str = "local"
    comet_enabled: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MachineProfile":
        payload = _dict(data)
        return cls(
            profile_id=str(payload["profile_id"]),
            run_root=str(payload["run_root"]),
            cache_dir=None
            if payload.get("cache_dir") is None
            else str(payload.get("cache_dir")),
            checkpoints_root=str(payload.get("checkpoints_root", "checkpoints")),
            artifact_root=str(payload.get("artifact_root", "artifacts")),
            logs_root=str(payload.get("logs_root", "logs")),
            vllm_port=None
            if payload.get("vllm_port") is None
            else int(payload.get("vllm_port")),
            world_size=None
            if payload.get("world_size") is None
            else int(payload.get("world_size")),
            cpu_workers=4
            if payload.get("cpu_workers") is None
            else int(payload.get("cpu_workers")),
            vllm_placement_policy=str(payload.get("vllm_placement_policy", "local")),
            comet_enabled=bool(payload.get("comet_enabled", False)),
            environment={
                str(key): str(value)
                for key, value in _dict(payload.get("environment")).items()
            },
        )


@dataclass(frozen=True, slots=True)
class TrainerSpec:
    use_vllm: bool = True
    vllm_server_port: int | None = 8000
    bf16: bool = True
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    beta: float = 0.0
    weight_decay: float = 0.0
    output_dir: str = "models/test_cadrecodev2"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 3000
    log_completions: bool = False
    logging_steps: int = 5
    num_generations: int = 16
    generation_batch_size: int = 64
    report_to: tuple[str, ...] = ("comet_ml",)
    run_name: str = "grpo_cadrecodev2_0"
    num_train_epochs: int = 20
    save_strategy: str = "steps"
    save_steps: int = 150
    save_total_limit: int | None = None
    temperature: float = 1.0
    top_p: float = 0.99
    top_k: int = 50
    importance_sampling_level: str = "sequence"
    loss_type: str = "dr_grpo"
    epsilon: float = 0.1
    num_iterations: int = 3
    scale_rewards: bool = False
    learning_rate: float = 3e-5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainerSpec":
        payload = _dict(data)
        report_to = payload.get("report_to", ("comet_ml",))
        if isinstance(report_to, (str, bytes, bytearray)):
            report_to = (str(report_to),)
        else:
            report_to = _tuple(report_to)
        return cls(
            use_vllm=bool(payload.get("use_vllm", True)),
            vllm_server_port=None
            if payload.get("vllm_server_port") is None
            else int(payload.get("vllm_server_port")),
            bf16=bool(payload.get("bf16", True)),
            gradient_checkpointing=bool(payload.get("gradient_checkpointing", True)),
            remove_unused_columns=bool(payload.get("remove_unused_columns", False)),
            ddp_find_unused_parameters=bool(
                payload.get("ddp_find_unused_parameters", False)
            ),
            beta=float(payload.get("beta", 0.0)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            output_dir=str(payload.get("output_dir", "models/test_cadrecodev2")),
            per_device_train_batch_size=int(
                payload.get("per_device_train_batch_size", 4)
            ),
            gradient_accumulation_steps=int(
                payload.get("gradient_accumulation_steps", 1)
            ),
            max_completion_length=int(payload.get("max_completion_length", 3000)),
            log_completions=bool(payload.get("log_completions", False)),
            logging_steps=int(payload.get("logging_steps", 5)),
            num_generations=int(payload.get("num_generations", 16)),
            generation_batch_size=int(payload.get("generation_batch_size", 64)),
            report_to=tuple(str(item) for item in report_to),
            run_name=str(payload.get("run_name", "grpo_cadrecodev2_0")),
            num_train_epochs=int(payload.get("num_train_epochs", 20)),
            save_strategy=str(payload.get("save_strategy", "steps")),
            save_steps=int(payload.get("save_steps", 150)),
            save_total_limit=None
            if payload.get("save_total_limit") is None
            else int(payload.get("save_total_limit")),
            temperature=float(payload.get("temperature", 1.0)),
            top_p=float(payload.get("top_p", 0.99)),
            top_k=int(payload.get("top_k", 50)),
            importance_sampling_level=str(
                payload.get("importance_sampling_level", "sequence")
            ),
            loss_type=str(payload.get("loss_type", "dr_grpo")),
            epsilon=float(payload.get("epsilon", 0.1)),
            num_iterations=int(payload.get("num_iterations", 3)),
            scale_rewards=bool(payload.get("scale_rewards", False)),
            learning_rate=float(payload.get("learning_rate", 3e-5)),
        )


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    run_id: str
    step: int | None
    path: str
    kind: str = "specific"
    parent_checkpoint: str | None = None
    score: float | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointRef":
        payload = _dict(data)
        return cls(
            run_id=str(payload["run_id"]),
            step=None if payload.get("step") is None else int(payload.get("step")),
            path=str(payload["path"]),
            kind=str(payload.get("kind", "specific")),
            parent_checkpoint=payload.get("parent_checkpoint"),
            score=None if payload.get("score") is None else float(payload.get("score")),
            created_at=payload.get("created_at"),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    resolved_config_path: str
    git_revision: str | None = None
    seed: int | None = None
    dataset_fingerprint: str | None = None
    machine_profile_id: str | None = None
    task_id: str | None = None
    model_id: str | None = None
    algorithm_id: str | None = None
    checkpoint_inventory: tuple[CheckpointRef, ...] = ()
    latest_checkpoint: str | None = None
    comet_experiment_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunManifest":
        payload = _dict(data)
        inventory = tuple(
            CheckpointRef.from_mapping(item)
            if not isinstance(item, CheckpointRef)
            else item
            for item in payload.get("checkpoint_inventory", [])
        )
        return cls(
            run_id=str(payload["run_id"]),
            resolved_config_path=str(payload["resolved_config_path"]),
            git_revision=payload.get("git_revision"),
            seed=None if payload.get("seed") is None else int(payload.get("seed")),
            dataset_fingerprint=payload.get("dataset_fingerprint"),
            machine_profile_id=payload.get("machine_profile_id"),
            task_id=payload.get("task_id"),
            model_id=payload.get("model_id"),
            algorithm_id=payload.get("algorithm_id"),
            checkpoint_inventory=inventory,
            latest_checkpoint=payload.get("latest_checkpoint"),
            comet_experiment_id=payload.get("comet_experiment_id"),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    run_id: str
    checkpoint: CheckpointRef
    sample_id: str
    prompt: str | None = None
    raw_generation: str | None = None
    wrapped_code: str | None = None
    render_path: str | None = None
    output_text: str | None = None
    status: str = "ok"
    duration_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InferenceRecord":
        payload = _dict(data)
        checkpoint = payload["checkpoint"]
        if not isinstance(checkpoint, CheckpointRef):
            checkpoint = CheckpointRef.from_mapping(checkpoint)
        return cls(
            run_id=str(payload["run_id"]),
            checkpoint=checkpoint,
            sample_id=str(payload["sample_id"]),
            prompt=payload.get("prompt"),
            raw_generation=payload.get("raw_generation"),
            wrapped_code=payload.get("wrapped_code"),
            render_path=payload.get("render_path"),
            output_text=payload.get("output_text"),
            status=str(payload.get("status", "ok")),
            duration_s=None
            if payload.get("duration_s") is None
            else float(payload.get("duration_s")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class MeshRecord:
    run_id: str
    checkpoint: CheckpointRef
    sample_id: str
    mesh_path: str | None = None
    status: str = "ok"
    failure_reason: str | None = None
    duration_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MeshRecord":
        payload = _dict(data)
        checkpoint = payload["checkpoint"]
        if not isinstance(checkpoint, CheckpointRef):
            checkpoint = CheckpointRef.from_mapping(checkpoint)
        return cls(
            run_id=str(payload["run_id"]),
            checkpoint=checkpoint,
            sample_id=str(payload["sample_id"]),
            mesh_path=payload.get("mesh_path"),
            status=str(payload.get("status", "ok")),
            failure_reason=payload.get("failure_reason"),
            duration_s=None
            if payload.get("duration_s") is None
            else float(payload.get("duration_s")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class EvalRecord:
    run_id: str
    checkpoint: CheckpointRef
    task_id: str
    suite: str
    split: str
    metric_name: str
    value: float
    sample_count: int
    invalid_count: int = 0
    failure_count: int = 0
    summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalRecord":
        payload = _dict(data)
        checkpoint = payload["checkpoint"]
        if not isinstance(checkpoint, CheckpointRef):
            checkpoint = CheckpointRef.from_mapping(checkpoint)
        return cls(
            run_id=str(payload["run_id"]),
            checkpoint=checkpoint,
            task_id=str(payload["task_id"]),
            suite=str(payload["suite"]),
            split=str(payload["split"]),
            metric_name=str(payload["metric_name"]),
            value=float(payload["value"]),
            sample_count=int(payload["sample_count"]),
            invalid_count=int(payload.get("invalid_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            summary=_dict(payload.get("summary")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    report_id: str
    run_ids: tuple[str, ...]
    checkpoint_refs: tuple[CheckpointRef, ...]
    leaderboard: tuple[Mapping[str, Any], ...] = ()
    checkpoint_history: tuple[Mapping[str, Any], ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComparisonReport":
        payload = _dict(data)
        checkpoints = tuple(
            CheckpointRef.from_mapping(item)
            if not isinstance(item, CheckpointRef)
            else item
            for item in payload.get("checkpoint_refs", [])
        )
        return cls(
            report_id=str(payload["report_id"]),
            run_ids=_tuple(payload.get("run_ids")),
            checkpoint_refs=checkpoints,
            leaderboard=tuple(_dict(item) for item in payload.get("leaderboard", [])),
            checkpoint_history=tuple(
                _dict(item) for item in payload.get("checkpoint_history", [])
            ),
            summary=_dict(payload.get("summary")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class ResolvedExperimentConfig:
    profile_id: str
    task: TaskSpec
    model: ModelSpec
    algorithm: AlgorithmSpec
    machine: MachineProfile
    trainer: TrainerSpec = field(default_factory=TrainerSpec)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResolvedExperimentConfig":
        payload = _dict(data)
        return cls(
            profile_id=str(payload["profile_id"]),
            task=TaskSpec.from_mapping(payload["task"]),
            model=ModelSpec.from_mapping(payload["model"]),
            algorithm=AlgorithmSpec.from_mapping(payload["algorithm"]),
            machine=MachineProfile.from_mapping(payload["machine"]),
            trainer=TrainerSpec.from_mapping(payload.get("trainer")),
            runtime=_dict(payload.get("runtime")),
            extras=_dict(payload.get("extras")),
        )
