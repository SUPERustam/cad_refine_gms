from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .schema import (
    AlgorithmSpec,
    MachineProfile,
    ModelSpec,
    ResolvedExperimentConfig,
    TaskSpec,
    TrainerSpec,
)


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


def _component_profile(
    component_name: str,
    value: Any,
    profiles_root: Path,
    base_dir: Path,
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        resolved = _resolve_path(value, profiles_root, relative_to=base_dir)
        return load_profile_document(
            resolved, profiles_root=profiles_root, _relative_to=resolved.parent
        )
    raise TypeError(
        f"Expected mapping or string reference for {component_name}, got {type(value)!r}"
    )


def resolve_experiment_profile(
    reference: str | Path,
    profiles_root: str | Path = "configs",
) -> ResolvedExperimentConfig:
    root = Path(profiles_root)
    experiment_path = _resolve_path(reference, root)
    document = load_profile_document(experiment_path, profiles_root=root)
    base_dir = experiment_path.parent
    overrides = document.pop("overrides", {})
    runtime = document.pop("runtime", {})
    profile_id = str(document.get("profile_id", experiment_path.stem))

    task_doc = _component_profile("task", document["task"], root, base_dir)
    model_doc = _component_profile("model", document["model"], root, base_dir)
    algorithm_doc = _component_profile(
        "algorithm", document["algorithm"], root, base_dir
    )
    machine_doc = _component_profile("machine", document["machine"], root, base_dir)
    trainer_doc = _component_profile(
        "trainer", document.get("trainer", {}), root, base_dir
    )

    if isinstance(overrides, Mapping):
        task_doc = _deep_merge(task_doc, overrides.get("task", {}))
        model_doc = _deep_merge(model_doc, overrides.get("model", {}))
        algorithm_doc = _deep_merge(algorithm_doc, overrides.get("algorithm", {}))
        machine_doc = _deep_merge(machine_doc, overrides.get("machine", {}))
        trainer_doc = _deep_merge(trainer_doc, overrides.get("trainer", {}))
        runtime = _deep_merge(runtime, overrides.get("runtime", {}))

    return ResolvedExperimentConfig(
        profile_id=profile_id,
        task=TaskSpec.from_mapping(task_doc),
        model=ModelSpec.from_mapping(model_doc),
        algorithm=AlgorithmSpec.from_mapping(algorithm_doc),
        machine=MachineProfile.from_mapping(machine_doc),
        trainer=TrainerSpec.from_mapping(trainer_doc),
        runtime=runtime if isinstance(runtime, Mapping) else {},
        extras={
            k: deepcopy(v)
            for k, v in document.items()
            if k
            not in {"profile_id", "task", "model", "algorithm", "machine", "trainer"}
        },
    )


class ProfileResolver:
    def __init__(self, profiles_root: str | Path = "configs") -> None:
        self.profiles_root = Path(profiles_root)

    def load(self, reference: str | Path) -> dict[str, Any]:
        return load_profile_document(reference, profiles_root=self.profiles_root)

    def resolve(self, reference: str | Path) -> ResolvedExperimentConfig:
        return resolve_experiment_profile(reference, profiles_root=self.profiles_root)

    def replace_root(self, profiles_root: str | Path) -> "ProfileResolver":
        return ProfileResolver(profiles_root)
