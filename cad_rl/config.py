from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
import yaml


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


class ConfigError(ValueError):
    pass


def _strict_section(
    section_name: str,
    value: Mapping[str, Any] | None,
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> dict[str, Any]:
    if value is None:
        raise ConfigError(f"Missing required section {section_name!r}")
    payload = _dict(value)
    allowed = set(required) | set(optional)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown keys in section {section_name!r}: {', '.join(unknown)}"
        )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ConfigError(
            f"Missing required keys in section {section_name!r}: {', '.join(missing)}"
        )
    return payload


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
        payload = _strict_section(
            "task",
            data,
            required=("task_id",),
            optional=("output_var_name", "default_eval_suites"),
        )
        return cls(
            task_id=str(payload["task_id"]),
            output_var_name=str(payload.get("output_var_name", "result")),
            default_eval_suites=_tuple(
                payload.get("default_eval_suites", ("quick", "standard", "full"))
            ),
        )


@dataclass(frozen=True, slots=True)
class DataSpec:
    prepared_datasets: Mapping[str, str]
    prompt_profile_id: str
    render_profile_id: str
    normalization_mode: str
    prompt_convention: str | None = None
    render_convention: str | None = None
    hf_dataset: bool = True
    raw_dataset_root: str | None = None
    raw_dataset_pickle: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DataSpec":
        payload = _strict_section(
            "data",
            data,
            required=(
                "prepared_datasets",
                "prompt_profile_id",
                "render_profile_id",
                "normalization_mode",
            ),
            optional=(
                "prompt_convention",
                "render_convention",
                "hf_dataset",
                "raw_dataset_root",
                "raw_dataset_pickle",
            ),
        )
        return cls(
            prepared_datasets={
                str(key): str(value)
                for key, value in _dict(payload.get("prepared_datasets")).items()
            },
            prompt_profile_id=str(payload["prompt_profile_id"]),
            render_profile_id=str(payload["render_profile_id"]),
            normalization_mode=str(payload["normalization_mode"]),
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
        payload = _strict_section(
            "model",
            data,
            required=(
                "base_checkpoint",
                "sft_checkpoint",
                "processor_name",
                "processor_kwargs",
                "generation_defaults",
                "trust_remote_code",
                "torch_dtype",
                "attn_implementation",
            ),
        )
        return cls(
            base_checkpoint=str(payload["base_checkpoint"]),
            sft_checkpoint=None
            if payload.get("sft_checkpoint") is None
            else str(payload.get("sft_checkpoint")),
            processor_name=None
            if payload.get("processor_name") is None
            else str(payload.get("processor_name")),
            processor_kwargs=_dict(payload["processor_kwargs"]),
            generation_defaults=_dict(payload["generation_defaults"]),
            trust_remote_code=bool(payload["trust_remote_code"]),
            torch_dtype=str(payload["torch_dtype"]),
            attn_implementation=str(payload["attn_implementation"]),
        )


@dataclass(frozen=True, slots=True)
class PrepareSpec:
    split: str
    output_path: str | None = None
    raw_root: str | None = None
    pickle_file: str | None = None
    shuffle: bool = False
    size: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PrepareSpec":
        payload = _strict_section(
            "prepare",
            data,
            required=("split",),
            optional=("output_path", "raw_root", "pickle_file", "shuffle", "size"),
        )
        return cls(
            split=str(payload["split"]),
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
class RewardSpec:
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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RewardSpec":
        payload = _strict_section(
            "train.reward",
            data,
            required=(
                "failure_reward",
                "iou_coef",
                "cd_coef",
                "auc_coef",
                "aoc_gms_coef",
                "r_mode",
            ),
            optional=(
                "get_nc",
                "nc_n_points",
                "nc_tol",
                "print_sample_steps",
                "pool_size",
                "get_aoc_gms",
                "aoc_gms_n_points",
                "aoc_gms_n_angles",
                "aoc_gms_rel_tol",
                "aoc_gms_cube_trick",
                "aoc_gms_pc_cache_enable",
                "aoc_gms_upper_bound_tol_rt",
                "aoc_gms_autofix_sampling",
            ),
        )
        return cls(
            failure_reward=float(payload["failure_reward"]),
            iou_coef=float(payload["iou_coef"]),
            cd_coef=float(payload["cd_coef"]),
            auc_coef=float(payload["auc_coef"]),
            aoc_gms_coef=float(payload["aoc_gms_coef"]),
            get_nc=bool(payload.get("get_nc", False)),
            nc_n_points=int(payload.get("nc_n_points", 16384)),
            nc_tol=int(payload.get("nc_tol", 5)),
            print_sample_steps=int(payload.get("print_sample_steps", 25)),
            pool_size=int(payload.get("pool_size", 16)),
            r_mode=str(payload["r_mode"]),
            get_aoc_gms=bool(payload.get("get_aoc_gms", False)),
            aoc_gms_n_points=int(payload.get("aoc_gms_n_points", 8192)),
            aoc_gms_n_angles=int(payload.get("aoc_gms_n_angles", 125)),
            aoc_gms_rel_tol=float(payload.get("aoc_gms_rel_tol", 0.05)),
            aoc_gms_cube_trick=bool(payload.get("aoc_gms_cube_trick", True)),
            aoc_gms_pc_cache_enable=bool(payload.get("aoc_gms_pc_cache_enable", False)),
            aoc_gms_upper_bound_tol_rt=int(
                payload.get("aoc_gms_upper_bound_tol_rt", 25)
            ),
            aoc_gms_autofix_sampling=bool(
                payload.get("aoc_gms_autofix_sampling", False)
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainSpec:
    loss_type: str
    importance_sampling_level: str
    scheduler: str
    reward: RewardSpec
    clip_cov: bool
    top_samples: int
    use_vllm: bool
    vllm_server_port: int | None
    bf16: bool
    gradient_checkpointing: bool
    remove_unused_columns: bool
    ddp_find_unused_parameters: bool
    beta: float
    weight_decay: float
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_completion_length: int
    log_completions: bool
    logging_steps: int
    num_generations: int
    generation_batch_size: int
    report_to: tuple[str, ...]
    run_name: str
    num_train_epochs: int
    save_strategy: str
    save_steps: int
    save_total_limit: int | None = None
    temperature: float = 1.0
    top_p: float = 0.99
    top_k: int = 50
    epsilon: float = 0.1
    num_iterations: int = 3
    scale_rewards: bool = False
    learning_rate: float = 3e-5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TrainSpec":
        payload = _strict_section(
            "train",
            data,
            required=(
                "loss_type",
                "importance_sampling_level",
                "scheduler",
                "reward",
                "clip_cov",
                "top_samples",
                "use_vllm",
                "vllm_server_port",
                "bf16",
                "gradient_checkpointing",
                "remove_unused_columns",
                "ddp_find_unused_parameters",
                "beta",
                "weight_decay",
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
                "temperature",
                "top_p",
                "top_k",
                "epsilon",
                "num_iterations",
                "scale_rewards",
                "learning_rate",
            ),
            optional=("save_total_limit",),
        )
        report_to = payload["report_to"]
        if isinstance(report_to, (str, bytes, bytearray)):
            report_to = (str(report_to),)
        else:
            report_to = _tuple(report_to)
        return cls(
            loss_type=str(payload["loss_type"]),
            importance_sampling_level=str(payload["importance_sampling_level"]),
            scheduler=str(payload["scheduler"]),
            reward=RewardSpec.from_mapping(payload["reward"]),
            clip_cov=bool(payload["clip_cov"]),
            top_samples=int(payload["top_samples"]),
            use_vllm=bool(payload["use_vllm"]),
            vllm_server_port=None
            if payload["vllm_server_port"] is None
            else int(payload["vllm_server_port"]),
            bf16=bool(payload["bf16"]),
            gradient_checkpointing=bool(payload["gradient_checkpointing"]),
            remove_unused_columns=bool(payload["remove_unused_columns"]),
            ddp_find_unused_parameters=bool(payload["ddp_find_unused_parameters"]),
            beta=float(payload["beta"]),
            weight_decay=float(payload["weight_decay"]),
            output_dir=str(payload["output_dir"]),
            per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
            max_completion_length=int(payload["max_completion_length"]),
            log_completions=bool(payload["log_completions"]),
            logging_steps=int(payload["logging_steps"]),
            num_generations=int(payload["num_generations"]),
            generation_batch_size=int(payload["generation_batch_size"]),
            report_to=tuple(str(item) for item in report_to),
            run_name=str(payload["run_name"]),
            num_train_epochs=int(payload["num_train_epochs"]),
            save_strategy=str(payload["save_strategy"]),
            save_steps=int(payload["save_steps"]),
            save_total_limit=None
            if payload.get("save_total_limit") is None
            else int(payload.get("save_total_limit")),
            temperature=float(payload["temperature"]),
            top_p=float(payload["top_p"]),
            top_k=int(payload["top_k"]),
            epsilon=float(payload["epsilon"]),
            num_iterations=int(payload["num_iterations"]),
            scale_rewards=bool(payload["scale_rewards"]),
            learning_rate=float(payload["learning_rate"]),
        )


@dataclass(frozen=True, slots=True)
class InferSpec:
    split: str
    output_path: str | None = None
    checkpoint: str | None = None
    checkpoint_kind: str = "specific"
    batch_size: int = 8
    num_workers: int = 0
    raw_recursive: bool = False
    max_samples: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "InferSpec":
        payload = _strict_section(
            "infer",
            data,
            required=("split", "checkpoint_kind", "batch_size", "num_workers"),
            optional=(
                "output_path",
                "checkpoint",
                "raw_recursive",
                "max_samples",
            ),
        )
        return cls(
            split=str(payload["split"]),
            output_path=None
            if payload.get("output_path") is None
            else str(payload.get("output_path")),
            checkpoint=None
            if payload.get("checkpoint") is None
            else str(payload.get("checkpoint")),
            checkpoint_kind=str(payload["checkpoint_kind"]),
            batch_size=int(payload["batch_size"]),
            num_workers=int(payload["num_workers"]),
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
        payload = _strict_section(
            "mesh",
            data,
            optional=("input_path", "output_dir"),
        )
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
        payload = _strict_section(
            "eval",
            data,
            optional=("input_path", "output_path", "split", "suite"),
        )
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
        payload = _strict_section(
            "compare",
            data,
            optional=("summaries", "output_path"),
        )
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
    debug: bool = False
    run_id: str | None = None
    git_revision: str | None = None
    comet_experiment_id: str | None = None
    dataset_fingerprint: str | None = None
    resume_path: str = ""
    scheduler_training_steps: int = 200000

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RuntimeSpec":
        payload = _strict_section(
            "runtime",
            data,
            required=("seed", "dataset_split"),
            optional=(
                "debug",
                "run_id",
                "git_revision",
                "comet_experiment_id",
                "dataset_fingerprint",
                "resume_path",
                "scheduler_training_steps",
            ),
        )
        return cls(
            seed=int(payload["seed"]),
            dataset_split=str(payload["dataset_split"]),
            debug=bool(payload.get("debug", False)),
            run_id=None
            if payload.get("run_id") is None
            else str(payload.get("run_id")),
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
            scheduler_training_steps=int(
                payload.get("scheduler_training_steps", 200000)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dataset_split": self.dataset_split,
            "debug": self.debug,
            "run_id": self.run_id,
            "git_revision": self.git_revision,
            "comet_experiment_id": self.comet_experiment_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "resume_path": self.resume_path,
            "scheduler_training_steps": self.scheduler_training_steps,
        }


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
        payload = _strict_section(
            "system",
            data,
            required=("profile_id", "run_root"),
            optional=(
                "cache_dir",
                "checkpoints_root",
                "artifact_root",
                "logs_root",
                "vllm_port",
                "world_size",
                "cpu_workers",
                "vllm_placement_policy",
                "comet_enabled",
                "environment",
            ),
        )
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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        payload = _strict_section(
            "root",
            data,
            required=(
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
            ),
            optional=("config_path",),
        )
        return cls(
            profile_id=str(payload["profile_id"]),
            experiment_id=str(payload["experiment_id"]),
            stage=str(payload["stage"]),
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
            system=SystemConfig.from_mapping(payload.get("system")),
        )


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
    return _load_mapping(_read_text(path))


def resolve_run_config(
    reference: str | Path,
    profiles_root: str | Path = "configs",
) -> RunConfig:
    root = Path(profiles_root)
    config_path = _resolve_path(reference, root)
    document = load_profile_document(config_path, profiles_root=root)

    return RunConfig(
        profile_id=str(document["profile_id"]),
        experiment_id=str(document["experiment_id"]),
        stage=str(document["stage"]),
        config_path=str(config_path),
        task=TaskSpec.from_mapping(document.get("task")),
        data=DataSpec.from_mapping(document.get("data")),
        model=ModelSpec.from_mapping(document.get("model")),
        prepare=PrepareSpec.from_mapping(document.get("prepare")),
        train=TrainSpec.from_mapping(document.get("train")),
        infer=InferSpec.from_mapping(document.get("infer")),
        mesh=MeshSpec.from_mapping(document.get("mesh")),
        eval=EvalSpec.from_mapping(document.get("eval")),
        compare=CompareSpec.from_mapping(document.get("compare")),
        runtime=RuntimeSpec.from_mapping(document.get("runtime")),
        system=SystemConfig.from_mapping(document.get("system")),
    )


class ProfileResolver:
    def __init__(self, profiles_root: str | Path = "configs") -> None:
        self.profiles_root = Path(profiles_root)

    def load(self, reference: str | Path) -> dict[str, Any]:
        return load_profile_document(reference, profiles_root=self.profiles_root)

    def resolve(self, reference: str | Path) -> RunConfig:
        return resolve_run_config(reference, profiles_root=self.profiles_root)

    def replace_root(self, profiles_root: str | Path) -> "ProfileResolver":
        return ProfileResolver(profiles_root)


__all__ = [
    "ConfigError",
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
    "RewardSpec",
    "RunConfig",
    "RuntimeSpec",
    "SystemConfig",
    "TaskSpec",
    "TrainSpec",
    "load_profile_document",
    "resolve_run_config",
    "to_serializable",
]
