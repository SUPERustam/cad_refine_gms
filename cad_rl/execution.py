from __future__ import annotations

import json
import os
import time
from multiprocessing import Process, get_context
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

from cad_rl.config import RunConfig, resolve_run_config, to_serializable
from cad_rl.inference import load_jsonl
from cad_rl.runtime import InferenceRecord, MeshRecord

try:  # pragma: no cover - optional dependency
    import trimesh
except Exception:  # pragma: no cover - optional dependency
    trimesh = None


class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super().Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc


def init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def compound_to_mesh(compound):
    if trimesh is None:  # pragma: no cover - import guard
        raise RuntimeError("trimesh is required for mesh execution")
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def execute_code_to_mesh(code_str, var_name="result"):
    import cadquery as cq

    ns = {"cq": cq}
    try:
        exec(code_str, ns)
        return compound_to_mesh(ns[var_name].val())
    except Exception:
        return None


def serialize_mesh(mesh: Any) -> dict[str, Any] | None:
    if mesh is None:
        return None
    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
    }


POOL = None


def init_pool(max_workers):
    ctx = get_context("forkserver")
    global POOL
    if POOL is None:
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx,
        )
    return POOL


def close_pool():
    global POOL
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None


def _execute_single_text(text, gt_file, var_name="result"):
    base_file = os.path.basename(gt_file).rsplit(".stl", 1)[0]
    try:
        pred_mesh = execute_code_to_mesh(text, var_name)
    except Exception:
        pred_mesh = None
    return {
        "file_name": base_file,
        "target_mesh_path": gt_file,
        "pred_mesh": serialize_mesh(pred_mesh),
        "status": "ok" if pred_mesh is not None else "invalid",
    }


def _run_child(conn, arg):
    try:
        conn.send(_execute_single_text(*arg))
    finally:
        conn.close()


def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_run_child, args=(child, arg))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        parent.close()
        return "__TIMEOUT__"
    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def execute_generated_codes(
    texts, meshes, max_workers=None, var_name="result"
):
    args = [(text, gt, var_name) for text, gt in zip(texts, meshes)]
    if POOL is None:
        return [_execute_single_text(*arg) for arg in args]

    async_results = [POOL.apply_async(timed_process_text, args=(arg,)) for arg in args]
    results = []
    for res in async_results:
        output = res.get()
        if output in {"__TIMEOUT__", "__CRASH__"}:
            results.append(
                {
                    "file_name": None,
                    "target_mesh_path": None,
                    "pred_mesh": None,
                    "status": "invalid",
                }
            )
        else:
            results.append(output)
    return results


def resolve_mesh_config(
    config_path: str | Path,
    *,
    system: str | Path | None = None,
) -> RunConfig:
    return resolve_run_config(
        config_path, profiles_root=Path(config_path).parents[1], system=system
    )


def export_mesh_contract(config: RunConfig) -> dict:
    return to_serializable(config)


def build_mesh_records(
    records: list[dict], output_dir: str | Path, var_name: str = "result"
) -> Path:
    output_dir = Path(output_dir)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mesh_records.jsonl"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(records):
            record = (
                item if isinstance(item, InferenceRecord) else InferenceRecord.from_mapping(item)
            )
            started = time.time()
            mesh = execute_code_to_mesh(record.raw_generation or "", var_name=var_name)
            mesh_path = meshes_dir / f"sample_{index:06d}.stl"
            status = "ok"
            error = None
            if mesh is None:
                status = "invalid"
                mesh_path = None
                error = "cadquery_execution_failed"
            else:
                mesh.export(str(mesh_path))
            row = MeshRecord(
                run_id=record.run_id,
                checkpoint=record.checkpoint,
                sample_id=record.sample_id,
                mesh_path=str(mesh_path) if mesh_path else None,
                status=status,
                failure_reason=error,
                duration_s=time.time() - started,
                metadata={
                    "task_id": record.metadata.get("task_id"),
                    "target_mesh_path": record.metadata.get("target_mesh_path"),
                    "source_record_status": record.status,
                },
            )
            handle.write(json.dumps(to_serializable(row)) + "\n")
    return manifest_path


def build_meshes_from_resolved(config: RunConfig) -> Path:
    if config.mesh.input_path is None or config.mesh.output_dir is None:
        raise ValueError("Mesh stage requires mesh.input_path and mesh.output_dir")
    records = load_jsonl(config.mesh.input_path)
    return build_mesh_records(
        records,
        config.mesh.output_dir,
        var_name=config.task.output_var_name,
    )
