from __future__ import annotations

from pathlib import Path

from ..config.schema import CheckpointRef
from .registry import RunRegistry


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
