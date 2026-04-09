import os
import time

os.environ["PYGLET_HEADLESS"] = "True"
from multiprocessing.pool import Pool
from multiprocessing import Process
from multiprocessing import get_context

import subprocess, sys
import traceback
import base64, pickle, json, signal, select

import numpy as np

import os
from logging_utils import get_logger, log_event, serialize_exception

_REMAP_RULES = [
    (
        "/home/jovyan/shares/SR008.nfs2/users/CAD/cadexp/datasets/",
        "/home/jovyan/shares/SR008.fs2/CAD/cadexp/datasets/",
    ),

    (
        "/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/MCB_A/",
        "/workspace-SR008.fs2/CAD/cadexp/datasets/MCB_A/",
    ),
    (
        "/home/jovyan/shares/SR008.nfs2/",
        "/home/jovyan/shares/SR008.fs2/",
    ),
    (
        "/workspace-SR008.nfs2/",
        "/workspace-SR008.fs2/",
    ),
]

def remap_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return p

    for old, new in _REMAP_RULES:
        if p.startswith(old):
            cand = new + p[len(old):]
            if os.path.exists(cand):
                return cand

    if os.path.exists(p):
        return p

    for old, new in _REMAP_RULES:
        if p.startswith(old):
            return new + p[len(old):]

    return p

class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc

# process initializer used in case of forkserver
def init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['cq'] = cq


def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area over the curve (AOC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    #tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol  / 100

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)
    # get the indices of the neighbors for each ground-truth point

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs] # candidates

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)  # (k,)
        best_idx = np.argmax(dots)  # index of the best aligned normal

        valid_pred_normals.append(pn_neighbors[best_idx])  # (3,)

        valid_gt_points.append(gt_points[i])  # (3,)
        valid_pred_points.append(pred_points[idxs[best_idx]])  # (3,)

    if len(valid_pred_normals) == 0:
        return None, None, None

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    #print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")

    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)

    # add invalid points to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    auc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible aoc (which is pi)

    #we want to maximize the AUC
    #aoc_normalized = 1 - auc_normalized
    # plot the aoc
    #if aoc_normalized > 0.3:
        #print(f"HIGH aoc: {aoc_normalized:.2f}")
        #plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)


    return auc_normalized, mean_cos_sim, per_invalid


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except Exception:
        raise


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd



def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(2.0 / max(mesh.extents))  # Normalize to [-1, 1]
    return mesh


def transform_mesh_0_1(mesh):
    # scale a mesh to be centered and inside [0,1]
    m = mesh.copy()
    if m.vertices.size == 0:
        return m
    center = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-center)
    extent = max(m.extents) if max(m.extents) > 1e-9 else 1.0
    m.apply_scale(1 / extent)
    m.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return m


def transform_gt_mesh_cad_recodev2(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(0.875 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh

def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh

def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep_less_safe(code_str, var_name="result"):
    safe_ns = {"cq": cq}
    ns=safe_ns.copy()
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns[var_name].val())
        # export files if needed
        # mesh.export(mesh_path)
        return mesh
    except Exception as e:
        raise RuntimeError(f"Error executing CadQuery code: {e}") from e


def get_metrics_from_single_text(text, gt_file, n_points, nc_params=None, var_name="result"):
    logger = get_logger()
    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]
    result = {
        "file_name": base_file,
        "cd": None,
        "iou": None,
        "auc": None,
        "auc_gms": None,
        "status": "ok",
        "phase": "init",
        "timings": {},
    }
    started_at = time.perf_counter()
    gt_mesh = None
    pred_mesh = None

    try:
        exec_started = time.perf_counter()
        pred_mesh = code_to_mesh_and_brep_less_safe(text, var_name)
        result["timings"]["cadquery_exec_ms"] = round((time.perf_counter() - exec_started) * 1000, 3)
        result["phase"] = "cadquery_exec"
    except Exception as e:
        result["status"] = "cadquery_error"
        result["error"] = serialize_exception(e)
        log_event(
            logger,
            "cadquery_execution_failed",
            status="error",
            level=40,
            mesh_path=gt_file,
            file_name=base_file,
            phase="cadquery_exec",
            exception=result["error"],
        )
        return result

    try: 
        load_started = time.perf_counter()
        gt_mesh = trimesh.load_mesh(gt_file)
        gt_mesh = transform_mesh_0_1(gt_mesh)
        pred_mesh = transform_mesh_0_1(pred_mesh)
        result["timings"]["mesh_load_ms"] = round((time.perf_counter() - load_started) * 1000, 3)
        result["phase"] = "metrics"

        cd_started = time.perf_counter()
        result["cd"] = compute_cd(gt_mesh, pred_mesh, n_points)
        result["timings"]["cd_ms"] = round((time.perf_counter() - cd_started) * 1000, 3)
        try:
            iou_started = time.perf_counter()
            result["iou"] = compute_iou(gt_mesh, pred_mesh)
            result["timings"]["iou_ms"] = round((time.perf_counter() - iou_started) * 1000, 3)
        except Exception as e:
            result["status"] = "iou_error"
            result["iou_error"] = serialize_exception(e)
            log_event(
                logger,
                "iou_failed",
                status="error",
                level=40,
                mesh_path=gt_file,
                file_name=base_file,
                phase="iou",
                exception=result["iou_error"],
            )
        if nc_params and nc_params.get("get_nc") is True:
            auc_started = time.perf_counter()
            result["auc"], _, _ = compute_normals_metrics(
                gt_mesh,
                pred_mesh,
                n_points=nc_params.get("n_points", n_points),
                tol=nc_params.get("tol", 5),
            )
            result["timings"]["auc_ms"] = round((time.perf_counter() - auc_started) * 1000, 3)
        if nc_params and nc_params.get("get_aoc_gms", False):
            try:
                from aoc_gms_metric import aoc_gms_from_meshes

                aoc_started = time.perf_counter()
                aoc_kwargs = {
                    "n_points": nc_params.get("aoc_gms_n_points", n_points),
                    "n_angles": nc_params.get("aoc_gms_n_angles", 125),
                    "rel_dist_tol": nc_params.get("aoc_gms_rel_tol", 0.05),
                    "cube_trick": nc_params.get("aoc_gms_cube_trick", True),
                    "pc_cache_enable": nc_params.get("aoc_gms_pc_cache_enable", False),
                    "upper_bound_tol_rt": nc_params.get("aoc_gms_upper_bound_tol_rt", 25),
                    "autofix_sampling": nc_params.get("aoc_gms_autofix_sampling", False),
                    "add_auc": True,
                }

                _, _, _, result["auc_gms"] = aoc_gms_from_meshes(
                    gt_mesh,
                    pred_mesh,
                    **aoc_kwargs,
                )
                result["timings"]["aoc_gms_ms"] = round((time.perf_counter() - aoc_started) * 1000, 3)
            except Exception as e:
                result["status"] = "aoc_gms_error"
                result["aoc_gms_error"] = serialize_exception(e)
                log_event(
                    logger,
                    "aoc_gms_failed",
                    status="error",
                    level=40,
                    mesh_path=gt_file,
                    file_name=base_file,
                    phase="aoc_gms",
                    exception=result["aoc_gms_error"],
                )

    except Exception as e:
        result["status"] = "metrics_error"
        result["phase"] = "metrics"
        result["error"] = serialize_exception(e)
        log_event(
            logger,
            "metrics_failed",
            status="error",
            level=40,
            mesh_path=gt_file,
            file_name=base_file,
            phase="metrics",
            exception=result["error"],
        )
    finally:
        result["timings"]["total_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        try:
            if gt_mesh is not None:
                del gt_mesh
            if pred_mesh is not None:
                del pred_mesh
        except Exception:
            log_event(
                logger,
                "metrics_cleanup_failed",
                status="error",
                level=40,
                mesh_path=gt_file,
                file_name=base_file,
                phase="cleanup",
            )
    return result




POOL = None
_POOL_PROCS = 0

def _metrics_timeout_sec() -> float:
    try:
        return float(os.environ.get("METRICS_TIMEOUT_SEC", "100"))
    except ValueError:
        return 100.0


class _MetricsTimeout(Exception):
    """Raised in pool workers when wall-clock timeout fires (best-effort; may not interrupt native code)."""


def _pool_worker_run(arg):
    """
    Run one metrics task inside a forkserver pool worker.

    Previously each task forked a second child + Pipe for timeout isolation, which doubled
    process count and IPC (semaphores/shm) and contributed to SIGBUS under load. Timeout is
    handled here with SIGALRM / setitimer in the worker only (no nested Process).
    """

    def _run():
        return get_metrics_from_single_text(*arg)

    timeout = _metrics_timeout_sec()
    if timeout <= 0 or os.name != "posix" or not hasattr(signal, "SIGALRM"):
        try:
            return _run()
        except Exception:
            print("[metrics] worker error (no-timeout path):", flush=True)
            traceback.print_exc()
            return dict(file_name=None, cd=None, iou=None, auc=None)

    def _on_alarm(signum, frame):
        raise _MetricsTimeout()

    old = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            return _run()
        except _MetricsTimeout:
            return "__TIMEOUT__"
        except Exception:
            print("[metrics] worker error:", flush=True)
            traceback.print_exc()
            return dict(file_name=None, cd=None, iou=None, auc=None)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
    finally:
        signal.signal(signal.SIGALRM, old)


def init_pool(max_workers):
    ctx = get_context("forkserver")
    global POOL, _POOL_PROCS
    if POOL is None:
        _POOL_PROCS = int(max_workers)
        print(
            f"[metrics] pool workers={max_workers} timeout_sec={_metrics_timeout_sec()} "
            f"(single-level pool; no nested Process per sample)",
            flush=True,
        )
        #ctx = get_context("spawn")
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx,
        )
        log_event(get_logger(), "metrics_pool_initialized", pool_size=max_workers, start_method="forkserver")
    return POOL

def close_pool():
    global POOL
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None
        log_event(get_logger(), "metrics_pool_closed")



def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    started_at = time.perf_counter()
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return {
            "file_name": os.path.basename(arg[1]).rsplit(".stl", 1)[0],
            "cd": None,
            "iou": None,
            "auc": None,
            "auc_gms": None,
            "status": "timeout",
            "phase": "worker_timeout",
            "timings": {"total_ms": round((time.perf_counter() - started_at) * 1000, 3)},
        }

    result = parent.recv() if parent.poll() else {
        "file_name": os.path.basename(arg[1]).rsplit(".stl", 1)[0],
        "cd": None,
        "iou": None,
        "auc": None,
        "auc_gms": None,
        "status": "crash",
        "phase": "worker_crash",
        "timings": {"total_ms": round((time.perf_counter() - started_at) * 1000, 3)},
    }
    parent.close()
    return result  


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    except Exception as e:
        conn.send(
            {
                "file_name": os.path.basename(arg[1]).rsplit(".stl", 1)[0],
                "cd": None,
                "iou": None,
                "auc": None,
                "auc_gms": None,
                "status": "crash",
                "phase": "worker_exception",
                "error": serialize_exception(e),
            }
        )
    finally:
        conn.close()


def get_metrics_from_texts(texts, meshes, nc_params=None, max_workers=None, var_name="result"):
    logger = get_logger()
    n_points = 8192
    args = [
        (text, gt, n_points, nc_params, var_name)
        for text, gt in zip(texts, meshes)
    ]
    if os.environ.get("METRICS_DEBUG", "").strip() in ("1", "true", "yes"):
        print(
            f"[metrics] batch completions={len(args)} pool_workers={_POOL_PROCS} pid={os.getpid()}",
            flush=True,
        )
    async_results = [POOL.apply_async(_pool_worker_run, args=(arg,)) for arg in args]
    results = []

    for idx, res in enumerate(async_results):
        output = res.get()
        results.append(output)
        if output.get("status") != "ok":
            log_event(
                logger,
                "metrics_sample_non_ok",
                status=output.get("status"),
                level=40,
                mesh_path=meshes[idx],
                generation_idx=idx,
                metrics=output,
            )

    return results
