from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.schema import ResolvedExperimentConfig, RunManifest, to_serializable

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _write_yaml(path: Path, payload: Any) -> None:
    if yaml is None:
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")  # type: ignore[no-untyped-call]


def materialize_resolved_run(
    run_dir: str | Path,
    resolved: ResolvedExperimentConfig,
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

    return {
        "resolved_config": resolved_config_path,
        "manifest": manifest_path,
    }
