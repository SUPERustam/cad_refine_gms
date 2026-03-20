from __future__ import annotations

import json
from pathlib import Path

from cad_rl.metrics.async_metrics import code_to_mesh_and_brep_less_safe


def build_mesh_records(
    records: list[dict], output_dir: str | Path, var_name: str = "result"
) -> Path:
    output_dir = Path(output_dir)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mesh_records.jsonl"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            mesh = code_to_mesh_and_brep_less_safe(
                record["raw_generation"], var_name=var_name
            )
            mesh_path = meshes_dir / f"sample_{index:06d}.stl"
            status = "success"
            error = None
            if mesh is None:
                status = "invalid_code"
                mesh_path = None
                error = "cadquery_execution_failed"
            else:
                mesh.export(str(mesh_path))
            row = {
                "mesh_path": str(mesh_path) if mesh_path else None,
                "source_mesh_path": record["mesh_path"],
                "task_id": record.get("task_id"),
                "status": status,
                "error": error,
            }
            handle.write(json.dumps(row) + "\n")
    return manifest_path
