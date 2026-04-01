from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cad_rl.config import RunConfig, to_serializable

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value.items())
    raise TypeError(f"Expected mapping, got {type(value)!r}")


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_yaml(path: Path, payload: Any) -> None:
    if yaml is None:
        return
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),  # type: ignore[no-untyped-call]
        encoding="utf-8",
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
    trainer_id: str | None = None
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
            trainer_id=payload.get("trainer_id"),
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
            run_ids=tuple(str(item) for item in _tuple(payload.get("run_ids"))),
            checkpoint_refs=checkpoints,
            leaderboard=tuple(_dict(item) for item in payload.get("leaderboard", [])),
            checkpoint_history=tuple(
                _dict(item) for item in payload.get("checkpoint_history", [])
            ),
            summary=_dict(payload.get("summary")),
            metadata=_dict(payload.get("metadata")),
        )


class RunRegistry:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def resolved_config_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "resolved_config.json"

    def checkpoints_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoints"

    def latest_pointer_path(self, run_id: str) -> Path:
        return self.checkpoints_dir(run_id) / "latest.txt"

    def checkpoint_index_path(self, run_id: str) -> Path:
        return self.checkpoints_dir(run_id) / "index.jsonl"

    def ensure_run_dir(self, run_id: str) -> Path:
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir(run_id).mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_resolved_config(self, run_id: str, resolved: RunConfig) -> Path:
        self.ensure_run_dir(run_id)
        path = self.resolved_config_path(run_id)
        path.write_text(
            json.dumps(to_serializable(resolved), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_manifest(self, manifest: RunManifest) -> Path:
        self.ensure_run_dir(manifest.run_id)
        path = self.manifest_path(manifest.run_id)
        path.write_text(
            json.dumps(to_serializable(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load_manifest(self, run_id: str) -> RunManifest:
        data = json.loads(self.manifest_path(run_id).read_text(encoding="utf-8"))
        return RunManifest.from_mapping(data)

    def load_resolved_config(self, run_id: str) -> RunConfig:
        data = json.loads(self.resolved_config_path(run_id).read_text(encoding="utf-8"))
        return RunConfig.from_mapping(data)

    def append_checkpoint(
        self, ref: CheckpointRef, manifest_updates: Mapping[str, object] | None = None
    ) -> CheckpointRef:
        self.ensure_run_dir(ref.run_id)
        entry = CheckpointRef(
            run_id=ref.run_id,
            step=ref.step,
            path=ref.path,
            kind=ref.kind,
            parent_checkpoint=ref.parent_checkpoint,
            score=ref.score,
            created_at=ref.created_at or _now_iso(),
            metadata={**ref.metadata, **dict(manifest_updates or {})},
        )
        index_path = self.checkpoint_index_path(ref.run_id)
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_serializable(entry), sort_keys=True))
            handle.write("\n")
        if entry.kind == "latest" or self._is_newer_than_latest(ref.run_id, entry):
            self.latest_pointer_path(ref.run_id).write_text(
                entry.path, encoding="utf-8"
            )
        return entry

    def _is_newer_than_latest(self, run_id: str, candidate: CheckpointRef) -> bool:
        latest = self.read_latest_checkpoint(run_id)
        if latest is None:
            return True
        if candidate.step is None:
            return False
        if latest.step is None:
            return True
        return candidate.step >= latest.step

    def list_checkpoints(self, run_id: str) -> list[CheckpointRef]:
        path = self.checkpoint_index_path(run_id)
        if not path.exists():
            return []
        entries: list[CheckpointRef] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(CheckpointRef.from_mapping(json.loads(line)))
        return entries

    def read_latest_checkpoint(self, run_id: str) -> CheckpointRef | None:
        pointer = self.latest_pointer_path(run_id)
        if pointer.exists():
            path = pointer.read_text(encoding="utf-8").strip()
            candidates = [
                item for item in self.list_checkpoints(run_id) if item.path == path
            ]
            if candidates:
                return candidates[-1]
            return CheckpointRef(run_id=run_id, step=None, path=path, kind="latest")
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda item: -1 if item.step is None else item.step)

    def read_best_checkpoint(self, run_id: str) -> CheckpointRef | None:
        checkpoints = self.list_checkpoints(run_id)
        scored = [item for item in checkpoints if item.score is not None]
        if scored:
            return max(
                scored,
                key=lambda item: (
                    float(item.score) if item.score is not None else float("-inf"),
                    -1 if item.step is None else item.step,
                ),
            )
        if checkpoints:
            return max(
                checkpoints, key=lambda item: -1 if item.step is None else item.step
            )
        return None

    def write_manifest_from_resolved(
        self,
        resolved: RunConfig,
        run_id: str,
        *,
        git_revision: str | None = None,
        seed: int | None = None,
        dataset_fingerprint: str | None = None,
        comet_experiment_id: str | None = None,
        latest_checkpoint: str | None = None,
        checkpoint_inventory: Iterable[CheckpointRef] = (),
    ) -> RunManifest:
        manifest = RunManifest(
            run_id=run_id,
            resolved_config_path=str(self.resolved_config_path(run_id)),
            git_revision=git_revision,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
            machine_profile_id=resolved.system.profile_id,
            task_id=resolved.task.task_id,
            model_id=resolved.model.base_checkpoint,
            trainer_id=resolved.train.trainer_type,
            checkpoint_inventory=tuple(checkpoint_inventory),
            latest_checkpoint=latest_checkpoint,
            comet_experiment_id=comet_experiment_id,
            metadata={"created_at": _now_iso()},
        )
        self.write_manifest(manifest)
        if latest_checkpoint is not None:
            self.latest_pointer_path(run_id).write_text(
                latest_checkpoint, encoding="utf-8"
            )
        return manifest

    def materialize_run(
        self,
        run_id: str,
        resolved: RunConfig,
        *,
        git_revision: str | None = None,
        seed: int | None = None,
        dataset_fingerprint: str | None = None,
        comet_experiment_id: str | None = None,
        checkpoint_inventory: Iterable[CheckpointRef] = (),
        latest_checkpoint: str | None = None,
    ) -> RunManifest:
        self.ensure_run_dir(run_id)
        self.write_resolved_config(run_id, resolved)
        return self.write_manifest_from_resolved(
            resolved,
            run_id,
            git_revision=git_revision,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
            comet_experiment_id=comet_experiment_id,
            latest_checkpoint=latest_checkpoint,
            checkpoint_inventory=checkpoint_inventory,
        )


def materialize_resolved_run(
    run_dir: str | Path,
    resolved: RunConfig,
    manifest: RunManifest,
) -> dict[str, Path]:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (path / "artifacts").mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)

    resolved_config_path = path / "resolved_config.json"
    resolved_payload = to_serializable(resolved)
    resolved_config_path.write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_yaml(path / "resolved_config.yaml", resolved_payload)

    manifest_path = path / "manifest.json"
    manifest_payload = to_serializable(manifest)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_yaml(path / "manifest.yaml", manifest_payload)

    if manifest.latest_checkpoint:
        (checkpoints_dir / "latest.txt").write_text(
            manifest.latest_checkpoint, encoding="utf-8"
        )

    return {"resolved_config": resolved_config_path, "manifest": manifest_path}


def select_latest_checkpoint(run_id: str, registry: RunRegistry) -> CheckpointRef:
    checkpoint = registry.read_latest_checkpoint(run_id)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint registered for run {run_id!r}")
    return checkpoint


def select_best_checkpoint(run_id: str, registry: RunRegistry) -> CheckpointRef:
    checkpoint = registry.read_best_checkpoint(run_id)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint registered for run {run_id!r}")
    return checkpoint


def _match_specific(
    run_id: str, registry: RunRegistry, specific: str | Path | int
) -> CheckpointRef:
    if isinstance(specific, int):
        candidates = [
            item for item in registry.list_checkpoints(run_id) if item.step == specific
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for step {specific} in run {run_id!r}"
            )
        return candidates[-1]

    candidate_path = Path(specific)
    if candidate_path.exists():
        return CheckpointRef(
            run_id=run_id, step=None, path=str(candidate_path), kind="specific"
        )

    candidates = [
        item
        for item in registry.list_checkpoints(run_id)
        if item.path == str(specific)
        or (item.step is not None and str(item.step) == str(specific))
    ]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No checkpoint found for reference {specific!r} in run {run_id!r}"
    )


def select_specific_checkpoint(
    run_id: str, registry: RunRegistry, specific: str | Path | int
) -> CheckpointRef:
    return _match_specific(run_id, registry, specific)


def select_checkpoint(
    run_id: str,
    registry: RunRegistry,
    reference: str | Path | int = "latest",
) -> CheckpointRef:
    if reference == "latest":
        return select_latest_checkpoint(run_id, registry)
    if reference == "best":
        return select_best_checkpoint(run_id, registry)
    return select_specific_checkpoint(run_id, registry, reference)


__all__ = [
    "CheckpointRef",
    "ComparisonReport",
    "EvalRecord",
    "InferenceRecord",
    "MeshRecord",
    "RunManifest",
    "RunRegistry",
    "materialize_resolved_run",
    "select_best_checkpoint",
    "select_checkpoint",
    "select_latest_checkpoint",
    "select_specific_checkpoint",
]
