from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from ..config.schema import (
    CheckpointRef,
    RunManifest,
    ResolvedExperimentConfig,
    to_serializable,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def write_resolved_config(
        self, run_id: str, resolved: ResolvedExperimentConfig
    ) -> Path:
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

    def load_resolved_config(self, run_id: str) -> ResolvedExperimentConfig:
        data = json.loads(self.resolved_config_path(run_id).read_text(encoding="utf-8"))
        return ResolvedExperimentConfig.from_mapping(data)

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
            return CheckpointRef(
                run_id=run_id,
                step=None,
                path=path,
                kind="latest",
            )
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None
        return max(
            checkpoints, key=lambda item: (-1 if item.step is None else item.step)
        )

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
                checkpoints, key=lambda item: (-1 if item.step is None else item.step)
            )
        return None

    def write_manifest_from_resolved(
        self,
        resolved: ResolvedExperimentConfig,
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
            machine_profile_id=resolved.machine.profile_id,
            task_id=resolved.task.task_id,
            model_id=resolved.model.base_checkpoint,
            algorithm_id=resolved.algorithm.trainer_type,
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
        resolved: ResolvedExperimentConfig,
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
