from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


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
    output_var_name: str = "result"
    default_eval_suites: tuple[str, ...] = ("quick", "standard", "full")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TaskSpec":
        payload = _dict(data)
        aliases = {
            "output_variable_name": "output_var_name",
            "var_name": "output_var_name",
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)
        return cls(
            task_id=str(payload["task_id"]),
            output_var_name=str(payload.get("output_var_name", "result")),
            default_eval_suites=_tuple(
                payload.get("default_eval_suites", ("quick", "standard", "full"))
            ),
        )


@dataclass(frozen=True, slots=True)
class DataSpec:
    prepared_datasets: Mapping[str, str] = field(default_factory=dict)
    prompt_profile_id: str = "default_prompt"
    render_profile_id: str = "default_render"
    normalization_mode: str = "fixed"
    prompt_convention: str | None = None
    render_convention: str | None = None
    hf_dataset: bool = True
    raw_dataset_root: str | None = None
    raw_dataset_pickle: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DataSpec":
        payload = _dict(data)
        aliases = {
            "prepared_dataset_splits": "prepared_datasets",
            "prepared_hf_dataset_splits": "prepared_datasets",
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)
        return cls(
            prepared_datasets={
                str(key): str(value)
                for key, value in _dict(payload.get("prepared_datasets")).items()
            },
            prompt_profile_id=str(payload.get("prompt_profile_id", "default_prompt")),
            render_profile_id=str(payload.get("render_profile_id", "default_render")),
            normalization_mode=str(payload.get("normalization_mode", "fixed")),
            prompt_convention=payload.get("prompt_convention"),
            render_convention=payload.get("render_convention"),
            hf_dataset=bool(payload.get("hf_dataset", True)),
            raw_dataset_root=None
            if payload.get("raw_dataset_root") is None
            else str(payload.get("raw_dataset_root")),
            raw_dataset_pickle=None
            if payload.get("raw_dataset_pickle") is None
            else str(payload.get("raw_dataset_pickle")),
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
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ModelSpec":
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
            torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
            attn_implementation=str(
                payload.get("attn_implementation", "flash_attention_2")
            ),
        )


@dataclass(frozen=True, slots=True)
class PrepareSpec:
    split: str = "train"
    output_path: str | None = None
    raw_root: str | None = None
    pickle_file: str | None = None
    shuffle: bool = False
    size: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PrepareSpec":
        payload = _dict(data)
        return cls(
            split=str(payload.get("split", "train")),
            output_path=None
            if payload.get("output_path") is None
            else str(payload.get("output_path")),
            raw_root=None
            if payload.get("raw_root") is None
            else str(payload.get("raw_root")),
            pickle_file=None
            if payload.get("pickle_file") is None
            else str(payload.get("pickle_file")),
            shuffle=bool(payload.get("shuffle", False)),
            size=None if payload.get("size") is None else int(payload.get("size")),
        )


@dataclass(frozen=True, slots=True)
class TrainSpec:
    trainer_type: str
    loss_mode: str
    importance_sampling_mode: str
    top_k_policy: str
    optimizer_policy: str
    scheduler_policy: str
    reward_config: Mapping[str, Any] = field(default_factory=dict)
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)
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
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TrainSpec":
        payload = _dict(data)
        report_to = payload.get("report_to", ("comet_ml",))
        if isinstance(report_to, (str, bytes, bytearray)):
            report_to = (str(report_to),)
        else:
            report_to = _tuple(report_to)
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
class InferSpec:
    split: str = "val"
    output_path: str | None = None
    checkpoint: str | None = None
    checkpoint_kind: str = "specific"
    batch_size: int = 8
    num_workers: int = 0
    raw_recursive: bool = False
    max_samples: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "InferSpec":
        payload = _dict(data)
        checkpoint = payload.get("checkpoint_path", payload.get("checkpoint"))
        return cls(
            split=str(payload.get("split", "val")),
            output_path=None
            if payload.get("output_path") is None
            else str(payload.get("output_path")),
            checkpoint=None if checkpoint is None else str(checkpoint),
            checkpoint_kind=str(payload.get("checkpoint_kind", "specific")),
            batch_size=int(payload.get("batch_size", 8)),
            num_workers=int(payload.get("num_workers", 0)),
            raw_recursive=bool(payload.get("raw_recursive", False)),
            max_samples=None
            if payload.get("max_samples") is None
            else int(payload.get("max_samples")),
        )


@dataclass(frozen=True, slots=True)
class MeshSpec:
    input_path: str | None = None
    output_dir: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MeshSpec":
        payload = _dict(data)
        return cls(
            input_path=None
            if payload.get("input_path") is None
            else str(payload.get("input_path")),
            output_dir=None
            if payload.get("output_dir") is None
            else str(payload.get("output_dir")),
        )


@dataclass(frozen=True, slots=True)
class EvalSpec:
    input_path: str | None = None
    output_path: str | None = None
    split: str = "val"
    suite: str = "standard"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EvalSpec":
        payload = _dict(data)
        return cls(
            input_path=None
            if payload.get("input_path") is None
            else str(payload.get("input_path")),
            output_path=None
            if payload.get("output_path") is None
            else str(payload.get("output_path")),
            split=str(payload.get("split", "val")),
            suite=str(payload.get("suite", "standard")),
        )


@dataclass(frozen=True, slots=True)
class CompareSpec:
    summaries: tuple[str, ...] = ()
    output_path: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CompareSpec":
        payload = _dict(data)
        return cls(
            summaries=tuple(str(item) for item in _tuple(payload.get("summaries"))),
            output_path=None
            if payload.get("output_path") is None
            else str(payload.get("output_path")),
        )


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    seed: int = 16
    dataset_split: str = "train"
    run_id: str | None = None
    git_revision: str | None = None
    comet_experiment_id: str | None = None
    dataset_fingerprint: str | None = None
    resume_path: str = ""
    scheduler_training_steps: int = 200000
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RuntimeSpec":
        payload = _dict(data)
        extras_payload = _dict(payload.get("extras"))
        consumed = {
            "seed",
            "dataset_split",
            "run_id",
            "git_revision",
            "comet_experiment_id",
            "dataset_fingerprint",
            "resume_path",
            "scheduler_training_steps",
        }
        return cls(
            seed=int(payload.get("seed", 16)),
            dataset_split=str(payload.get("dataset_split", "train")),
            run_id=None if payload.get("run_id") is None else str(payload.get("run_id")),
            git_revision=None
            if payload.get("git_revision") is None
            else str(payload.get("git_revision")),
            comet_experiment_id=None
            if payload.get("comet_experiment_id") is None
            else str(payload.get("comet_experiment_id")),
            dataset_fingerprint=None
            if payload.get("dataset_fingerprint") is None
            else str(payload.get("dataset_fingerprint")),
            resume_path=str(payload.get("resume_path", "")),
            scheduler_training_steps=int(payload.get("scheduler_training_steps", 200000)),
            extras={
                **{k: deepcopy(v) for k, v in payload.items() if k not in consumed | {"extras"}},
                **{k: deepcopy(v) for k, v in extras_payload.items()},
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "seed": self.seed,
            "dataset_split": self.dataset_split,
            "run_id": self.run_id,
            "git_revision": self.git_revision,
            "comet_experiment_id": self.comet_experiment_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "resume_path": self.resume_path,
            "scheduler_training_steps": self.scheduler_training_steps,
        }
        payload.update(self.extras)
        return payload


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
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MachineProfile":
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
            cpu_workers=int(payload.get("cpu_workers", 4)),
            vllm_placement_policy=str(payload.get("vllm_placement_policy", "local")),
            comet_enabled=bool(payload.get("comet_enabled", False)),
            environment={
                str(key): str(value)
                for key, value in _dict(payload.get("environment")).items()
            },
        )


SystemConfig = MachineProfile


@dataclass(frozen=True, slots=True)
class RunConfig:
    profile_id: str
    experiment_id: str
    stage: str
    config_path: str
    task: TaskSpec
    data: DataSpec
    model: ModelSpec
    prepare: PrepareSpec
    train: TrainSpec
    infer: InferSpec
    mesh: MeshSpec
    eval: EvalSpec
    compare: CompareSpec
    runtime: RuntimeSpec
    system: SystemConfig
    common_config_path: str | None = None
    system_config_path: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        payload = _dict(data)
        system_payload = payload.get("system", payload.get("machine"))
        return cls(
            profile_id=str(payload["profile_id"]),
            experiment_id=str(payload.get("experiment_id", payload["profile_id"])),
            stage=str(payload.get("stage", "unknown")),
            config_path=str(payload.get("config_path", "")),
            task=TaskSpec.from_mapping(payload.get("task")),
            data=DataSpec.from_mapping(payload.get("data")),
            model=ModelSpec.from_mapping(payload.get("model")),
            prepare=PrepareSpec.from_mapping(payload.get("prepare")),
            train=TrainSpec.from_mapping(payload.get("train")),
            infer=InferSpec.from_mapping(payload.get("infer")),
            mesh=MeshSpec.from_mapping(payload.get("mesh")),
            eval=EvalSpec.from_mapping(payload.get("eval")),
            compare=CompareSpec.from_mapping(payload.get("compare")),
            runtime=RuntimeSpec.from_mapping(payload.get("runtime")),
            system=SystemConfig.from_mapping(system_payload),
            common_config_path=payload.get("common_config_path"),
            system_config_path=payload.get("system_config_path"),
            extras=_dict(payload.get("extras")),
        )

    @property
    def machine(self) -> SystemConfig:
        return self.system


ResolvedExperimentConfig = RunConfig


DEFAULT_PROFILE_SUFFIXES = (".yaml", ".yml", ".json")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_mapping(text: str) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(text)  # type: ignore[no-untyped-call]
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Profile document must be a mapping, got {type(loaded)!r}")
    return dict(loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(
    reference: str | Path, profiles_root: Path, relative_to: Path | None = None
) -> Path:
    candidate = Path(reference)
    if candidate.exists():
        return candidate
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_roots = []
    if relative_to is not None:
        search_roots.append(relative_to)
    search_roots.append(profiles_root)

    if candidate.suffix:
        for root in search_roots:
            resolved = root / candidate
            if resolved.exists():
                return resolved
    else:
        for root in search_roots:
            for suffix in DEFAULT_PROFILE_SUFFIXES:
                resolved = root / f"{candidate.name}{suffix}"
                if resolved.exists():
                    return resolved
            resolved = root / candidate
            if resolved.exists():
                return resolved

    if relative_to is not None:
        candidate = relative_to / candidate
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve profile reference: {reference}")


def load_profile_document(
    reference: str | Path,
    profiles_root: str | Path = "configs",
    *,
    _relative_to: Path | None = None,
) -> dict[str, Any]:
    root = Path(profiles_root)
    path = _resolve_path(reference, root, relative_to=_relative_to)
    data = _load_mapping(_read_text(path))
    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, (str, Path)):
        extends = [extends]

    merged: dict[str, Any] = {}
    for item in extends:
        base_doc = load_profile_document(
            item, profiles_root=root, _relative_to=path.parent
        )
        merged = _deep_merge(merged, base_doc)
    return _deep_merge(merged, data)


def _load_system_document(
    base_dir: Path,
    profiles_root: Path,
    document: Mapping[str, Any],
    system_reference: str | Path | None,
) -> tuple[dict[str, Any], str | None]:
    system_doc = dict(document.get("system", {}))
    reference = document.get("system_profile") if system_reference is None else system_reference
    if reference is None:
        return system_doc, None

    resolved = _resolve_path(reference, profiles_root, relative_to=base_dir)
    loaded = load_profile_document(
        resolved, profiles_root=profiles_root, _relative_to=resolved.parent
    )
    return _deep_merge(loaded, system_doc), str(resolved)


def _ensure_section(section_name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"Section {section_name!r} must be an inline mapping in the active config tree"
    )


def resolve_run_config(
    reference: str | Path,
    profiles_root: str | Path = "configs",
    *,
    system: str | Path | None = None,
) -> RunConfig:
    root = Path(profiles_root)
    config_path = _resolve_path(reference, root)
    stage_document = load_profile_document(config_path, profiles_root=root)
    base_dir = config_path.parent
    common_path: Path | None = None
    common_document: dict[str, Any] = {}
    if config_path.stem != "common":
        candidate = base_dir / "common.yaml"
        if candidate.exists():
            common_path = candidate
            common_document = load_profile_document(candidate, profiles_root=root)
    document = _deep_merge(common_document, stage_document)
    overrides = _dict(document.pop("overrides", {}))
    stage = str(document.get("stage", config_path.stem))
    experiment_id = str(document.get("experiment_id", base_dir.name))
    profile_id = str(document.get("profile_id", experiment_id))

    task_doc = _ensure_section("task", document.get("task"))
    data_doc = _ensure_section("data", document.get("data"))
    model_doc = _ensure_section("model", document.get("model"))
    prepare_doc = _ensure_section("prepare", document.get("prepare"))
    train_doc = _ensure_section("train", document.get("train"))
    infer_doc = _ensure_section("infer", document.get("infer"))
    mesh_doc = _ensure_section("mesh", document.get("mesh"))
    eval_doc = _ensure_section("eval", document.get("eval"))
    compare_doc = _ensure_section("compare", document.get("compare"))
    runtime_doc = _ensure_section("runtime", document.get("runtime"))

    system_doc, system_config_path = _load_system_document(
        base_dir, root, document, system
    )
    if "profile_id" not in system_doc:
        system_doc["profile_id"] = str(document.get("system_id", "default"))

    task_doc = _deep_merge(task_doc, _dict(overrides.get("task")))
    data_doc = _deep_merge(data_doc, _dict(overrides.get("data")))
    model_doc = _deep_merge(model_doc, _dict(overrides.get("model")))
    prepare_doc = _deep_merge(prepare_doc, _dict(overrides.get("prepare")))
    train_doc = _deep_merge(train_doc, _dict(overrides.get("train")))
    infer_doc = _deep_merge(infer_doc, _dict(overrides.get("infer")))
    mesh_doc = _deep_merge(mesh_doc, _dict(overrides.get("mesh")))
    eval_doc = _deep_merge(eval_doc, _dict(overrides.get("eval")))
    compare_doc = _deep_merge(compare_doc, _dict(overrides.get("compare")))
    runtime_doc = _deep_merge(runtime_doc, _dict(overrides.get("runtime")))
    system_doc = _deep_merge(
        system_doc, _dict(overrides.get("system", overrides.get("machine")))
    )

    return RunConfig(
        profile_id=profile_id,
        experiment_id=experiment_id,
        stage=stage,
        config_path=str(config_path),
        common_config_path=None if common_path is None else str(common_path),
        system_config_path=system_config_path,
        task=TaskSpec.from_mapping(task_doc),
        data=DataSpec.from_mapping(data_doc),
        model=ModelSpec.from_mapping(model_doc),
        prepare=PrepareSpec.from_mapping(prepare_doc),
        train=TrainSpec.from_mapping(train_doc),
        infer=InferSpec.from_mapping(infer_doc),
        mesh=MeshSpec.from_mapping(mesh_doc),
        eval=EvalSpec.from_mapping(eval_doc),
        compare=CompareSpec.from_mapping(compare_doc),
        runtime=RuntimeSpec.from_mapping(runtime_doc),
        system=SystemConfig.from_mapping(system_doc),
        extras={
            k: deepcopy(v)
            for k, v in document.items()
            if k
            not in {
                "profile_id",
                "experiment_id",
                "stage",
                "task",
                "data",
                "model",
                "prepare",
                "train",
                "infer",
                "mesh",
                "eval",
                "compare",
                "runtime",
                "system",
                "system_profile",
                "overrides",
            }
        },
    )


def resolve_experiment_profile(
    reference: str | Path,
    profiles_root: str | Path = "configs",
) -> RunConfig:
    return resolve_run_config(reference, profiles_root=profiles_root)


class ProfileResolver:
    def __init__(self, profiles_root: str | Path = "configs") -> None:
        self.profiles_root = Path(profiles_root)

    def load(self, reference: str | Path) -> dict[str, Any]:
        return load_profile_document(reference, profiles_root=self.profiles_root)

    def resolve(
        self, reference: str | Path, *, system: str | Path | None = None
    ) -> RunConfig:
        return resolve_run_config(
            reference, profiles_root=self.profiles_root, system=system
        )

    def replace_root(self, profiles_root: str | Path) -> "ProfileResolver":
        return ProfileResolver(profiles_root)


__all__ = [
    "CompareSpec",
    "DataSpec",
    "EvalSpec",
    "InferSpec",
    "MachineProfile",
    "MeshSpec",
    "ModelSpec",
    "PrepareSpec",
    "ProfileResolver",
    "ResolvedExperimentConfig",
    "RunConfig",
    "RuntimeSpec",
    "SystemConfig",
    "TaskSpec",
    "TrainSpec",
    "load_profile_document",
    "resolve_experiment_profile",
    "resolve_run_config",
    "to_serializable",
]
