import os
import time

os.environ["PYGLET_HEADLESS"] = "True"
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError, Process
from multiprocessing import get_context

import subprocess, sys
import base64, pickle, json, signal, select

import numpy as np

import os

from benchmark.visualization_iso import Plotter


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
            cand = new + p[len(old) :]
            if os.path.exists(cand):
                return cand

    if os.path.exists(p):
        return p

    for old, new in _REMAP_RULES:
        if p.startswith(old):
            return new + p[len(old) :]

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


# Pool initializer for spawn workers (CPU metrics only; CUDA hidden in worker).
def init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq

    # make them available to your metric code
    globals()["trimesh"] = trimesh
    globals()["cKDTree"] = cKDTree
    globals()["cq"] = cq
    # Optional: create Plotter once per pool worker when MAE is always used (reduces per-fork init).
    # Off by default — PyVista state after fork can be finicky; enable via env if needed.
    if os.environ.get("METRICS_PREINIT_MAE_PLOTTER", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        _get_mae_plotter()


def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area over the curve (AOC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    # tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol / 100

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
        pn_neighbors = pred_normals[idxs]  # candidates

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
    # print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")

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
    cdf = np.arange(1, N + 1) / N

    from numpy import trapz

    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0], cdf, [1.0]))
    auc_normalized = (
        trapz(y, x) / np.pi
    )  # Normalize by the maximum possible aoc (which is pi)

    # we want to maximize the AUC
    # aoc_normalized = 1 - auc_normalized
    # plot the aoc
    # if aoc_normalized > 0.3:
    # print(f"HIGH aoc: {aoc_normalized:.2f}")
    # plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)

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
    except:
        pass


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


def render_based_mae_similarity(gt_file, pred_mesh):
    plotter = Plotter()
    gt_img = pred_img = None
    try:
        gt_img = plotter._get_img(
            gt_file, plotter.cmap_gt, apply_augs=False, color=(0, 255, 0), scale=True
        )
        pred_img = plotter._get_img(
            pred_mesh,
            plotter.cmap_pred,
            apply_augs=False,
            color=(0, 255, 0),
            scale=True,
        )
        if gt_img is None or pred_img is None:
            raise ValueError("GT or pred image is None")
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred_img)
        if gt_arr.shape != pred_arr.shape:
            raise ValueError("Matrices must have the same shape")
        try:
            gt_arr = gt_arr[
                :714, :, 1
            ]  # remove isometric views and leave only green channel
            pred_arr = pred_arr[:714, :, 1]
        except Exception:
            raise ValueError(
                "GT or pred array is not valid for removing isometric views/green channel"
            )
        if gt_arr.shape != pred_arr.shape:
            raise ValueError(
                "GT or pred array is not valid for removing isometric views/green channel"
            )
        return float(1.0 - np.mean(np.abs(gt_arr - pred_arr)) / 255.0)
    finally:
        for im in (gt_img, pred_img):
            if im is not None:
                try:
                    im.close()
                except Exception:
                    pass
        try:
            plotter.close()
        except Exception:
            pass
    return None


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep_less_safe(code_str, var_name="result"):
    safe_ns = {"cq": cq}
    ns = safe_ns.copy()
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns[var_name].val())
        # export files if needed
        # mesh.export(mesh_path)
        return mesh
    except Exception as e:
        print(f"Error executing CadQuery code : {e}")
        return None


def get_metrics_from_single_text(
    text, gt_file, n_points, nc_params=None, var_name="result"
):
    # ME: comment this
    # gt_file = os.path.abspath(gt_file)
    # gt_file = remap_path(gt_file)
    base_file = os.path.basename(gt_file).rsplit(".stl", 1)[0]
    try:
        pred_mesh = code_to_mesh_and_brep_less_safe(text, var_name)
    except Exception as e:
        return dict(
            file_name=base_file,
            cd=None,
            iou=None,
            auc=None,
            auc_gms=None,
            mae_similarity=None,
        )

    if pred_mesh is None:
        return dict(
            file_name=base_file,
            cd=None,
            iou=None,
            auc=None,
            auc_gms=None,
            mae_similarity=None,
        )
    cd, iou, auc, auc_gms, mae_similarity_val = None, None, None, None, None
    metric_cfg = nc_params or {}
    need_cd = bool(metric_cfg.get("get_cd", True))
    need_iou = bool(metric_cfg.get("get_iou", True))
    need_nc = bool(metric_cfg.get("get_nc", False))
    need_aoc_gms = bool(metric_cfg.get("get_aoc_gms", False))
    need_mae_render = bool(metric_cfg.get("get_mae_render", False))
    need_gt_mesh = need_cd or need_iou or need_nc or need_aoc_gms
    gt_mesh = None
    try:
        pred_mesh = transform_mesh_0_1(pred_mesh)
        if need_gt_mesh:
            gt_mesh = trimesh.load_mesh(gt_file)
            gt_mesh = transform_mesh_0_1(gt_mesh)

        if need_cd:
            try:
                cd = compute_cd(gt_mesh, pred_mesh, n_points)
            except Exception as e:
                print(f"CD error for {base_file}: {e}", flush=True)
                cd = None

        if need_iou:
            try:
                iou = compute_iou(gt_mesh, pred_mesh)
            except Exception as e:
                print(f"IoU error for {base_file}: {e}", flush=True)
                iou = None

        if need_nc:
            try:
                auc, _, _ = compute_normals_metrics(
                    gt_mesh,
                    pred_mesh,
                    n_points=metric_cfg.get("n_points", n_points),
                    tol=metric_cfg.get("tol", 5),
                )
            except Exception as e:
                print(f"Normals error for {base_file}: {e}", flush=True)
                auc = None

        if need_aoc_gms:
            try:
                from aoc_gms_metric import aoc_gms_from_meshes

                aoc_kwargs = {
                    "n_points": metric_cfg.get("aoc_gms_n_points", n_points),
                    "n_angles": metric_cfg.get("aoc_gms_n_angles", 125),
                    "rel_dist_tol": metric_cfg.get("aoc_gms_rel_tol", 0.05),
                    "cube_trick": metric_cfg.get("aoc_gms_cube_trick", True),
                    "pc_cache_enable": metric_cfg.get("aoc_gms_pc_cache_enable", False),
                    "upper_bound_tol_rt": metric_cfg.get(
                        "aoc_gms_upper_bound_tol_rt", 25
                    ),
                    "autofix_sampling": metric_cfg.get(
                        "aoc_gms_autofix_sampling", False
                    ),
                    "add_auc": True,
                }

                _, _, _, auc_gms = aoc_gms_from_meshes(
                    gt_mesh,
                    pred_mesh,
                    **aoc_kwargs,
                )
            except Exception as e:
                print(f"AOC-GMS error for {base_file}: {e}", flush=True)

        if need_mae_render:
            try:
                mae_similarity_val = render_based_mae_similarity(gt_file, pred_mesh)
            except Exception as e:
                print(f"MAE render error for {base_file}: {e}", flush=True)

    except Exception as e:
        print(f"error for {base_file}: {e}", flush=True)
        pass
    finally:
        try:
            if gt_mesh is not None:
                del gt_mesh
            if pred_mesh is not None:
                del pred_mesh
        except:
            pass
    return dict(
        file_name=base_file,
        cd=cd,
        iou=iou,
        auc=auc,
        auc_gms=auc_gms,
        mae_similarity=mae_similarity_val,
    )


POOL = None


def init_pool(max_workers):
    ctx = get_context("forkserver")
    global POOL
    if POOL is None:
        # ctx = get_context("spawn")
        # ctx = get_context("spawn")
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


def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    finally:
        conn.close()


def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    finally:
        conn.close()


def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    finally:
        conn.close()


def timed_process_text(arg, timeout=100):
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    finally:
        conn.close()


def get_metrics_from_texts(
    texts, meshes, nc_params=None, max_workers=None, var_name="result"
):
    n_points = 8192
    args = [
        (text, gt, n_points, nc_params, var_name) for text, gt in zip(texts, meshes)
    ]
    async_results = [POOL.apply_async(timed_process_text, args=(arg,)) for arg in args]
    results = []

    for res in async_results:
        output = res.get()
        if output == "__TIMEOUT__" or output == "__CRASH__":
            print(f"[{output}] metrics task computation ERROR, skipping", flush=True)
            results.append(
                dict(
                    file_name=None,
                    cd=None,
                    iou=None,
                    auc=None,
                    auc_gms=None,
                    mae_similarity=None,
                )
            )
        else:
            results.append(output)

    return results
