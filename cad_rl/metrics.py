from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import trimesh
except Exception:  # pragma: no cover - optional dependency
    trimesh = None


def _require_trimesh():
    if trimesh is None:  # pragma: no cover - import guard
        raise RuntimeError("trimesh is required for geometry metrics")
    return trimesh


def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    trimesh_mod = _require_trimesh()
    from scipy.spatial import cKDTree

    tol = pred_mesh.extents.max() * tol / 100
    gt_points, gt_face_indexes = trimesh_mod.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh_mod.sample.sample_surface(
        pred_mesh, n_points
    )
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)

    valid_pred_normals = []
    valid_gt_normals = []
    for i, idxs in enumerate(neighbors):
        if not idxs:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs]
        dots = (pn_neighbors * gn).sum(axis=1)
        best_idx = np.argmax(dots)
        valid_gt_normals.append(gn)
        valid_pred_normals.append(pn_neighbors[best_idx])

    if not valid_pred_normals:
        return None, None, None

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    cos_sim = np.clip((valid_pred_normals * valid_gt_normals).sum(axis=1), -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    angles = np.sort(np.arccos(cos_sim))
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))
    cdf = np.arange(1, len(angles) + 1) / len(angles)
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0], cdf, [1.0]))
    auc_normalized = np.trapezoid(y, x) / np.pi
    return auc_normalized, mean_cos_sim, per_invalid


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                intersection_volume += (
                    intersection.volume if intersection is not None else 0
                )
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        if union_volume <= 0:
            return None
        return intersection_volume / union_volume
    except Exception:
        return None


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    trimesh_mod = _require_trimesh()
    from scipy.spatial import cKDTree

    gt_points, _ = trimesh_mod.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh_mod.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def transform_mesh_0_1(mesh):
    trimesh_mod = _require_trimesh()
    m = mesh.copy()
    if m.vertices.size == 0:
        return m
    center = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-center)
    extent = max(m.extents) if max(m.extents) > 1e-9 else 1.0
    m.apply_scale(1 / extent)
    m.apply_transform(trimesh_mod.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return m


def transform_gt_mesh_cad_0875(mesh):
    trimesh_mod = _require_trimesh()
    if mesh is None or mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(0.875 / extent)
    mesh.apply_transform(
        trimesh_mod.transformations.translation_matrix([0.5, 0.5, 0.5])
    )
    return mesh


def compute_metrics_from_meshes(
    pred_mesh,
    target_mesh,
    *,
    n_points: int = 8192,
    normal_params: dict | None = None,
) -> dict[str, float | None]:
    if pred_mesh is None or target_mesh is None:
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}

    pred_mesh = transform_mesh_0_1(pred_mesh)
    target_mesh = transform_mesh_0_1(target_mesh)
    auc = None
    auc_gms = None
    if normal_params and normal_params.get("get_nc"):
        auc, _, _ = compute_normals_metrics(
            target_mesh,
            pred_mesh,
            n_points=normal_params.get("n_points", n_points),
            tol=normal_params.get("tol", 5),
        )
    return {
        "cd": compute_cd(target_mesh, pred_mesh, n_points),
        "iou": compute_iou(target_mesh, pred_mesh),
        "auc": auc,
        "auc_gms": auc_gms,
    }


def compute_metrics_from_mesh_paths(
    pred_mesh_path: str | None, target_mesh_path: str | None
) -> dict[str, float | None]:
    trimesh_mod = _require_trimesh()
    if not pred_mesh_path or not target_mesh_path:
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}
    pred_path = Path(pred_mesh_path)
    target_path = Path(target_mesh_path)
    if not pred_path.exists() or not target_path.exists():
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}
    try:
        pred_mesh = trimesh_mod.load_mesh(pred_path)
        target_mesh = trimesh_mod.load_mesh(target_path)
        return compute_metrics_from_meshes(pred_mesh, target_mesh)
    except Exception:
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}


def compute_metrics_from_execution(
    execution_result: dict[str, object],
    *,
    n_points: int = 8192,
    normal_params: dict | None = None,
) -> dict[str, float | None]:
    trimesh_mod = _require_trimesh()
    pred_mesh_data = execution_result.get("pred_mesh")
    target_mesh_path = execution_result.get("target_mesh_path")
    if not pred_mesh_data or not target_mesh_path:
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}
    try:
        pred_mesh = trimesh_mod.Trimesh(
            vertices=pred_mesh_data["vertices"],
            faces=pred_mesh_data["faces"],
        )
        target_mesh = trimesh_mod.load_mesh(Path(str(target_mesh_path)))
        return compute_metrics_from_meshes(
            pred_mesh,
            target_mesh,
            n_points=n_points,
            normal_params=normal_params,
        )
    except Exception:
        return {"cd": None, "iou": None, "auc": None, "auc_gms": None}


def reward_from_metrics(
    cd: float | None,
    iou: float | None,
    auc: float | None = 0,
    auc_gms: float | None = 0,
    mode: str = "default",
) -> float:
    if cd is None or (isinstance(cd, float) and (math.isnan(cd) or cd <= 0)):
        cd = 1.0
    if iou is None or (isinstance(iou, float) and math.isnan(iou)):
        iou = 0.0
    if auc is None or (isinstance(auc, float) and math.isnan(auc)):
        auc = 0.0
    if auc_gms is None or (isinstance(auc_gms, float) and math.isnan(auc_gms)):
        auc_gms = 0.0
    if mode == "10_iou":
        reward = 10.0 * float(iou)
    elif mode == "cd_to_reward":
        ln = math.log(max(cd, 1e-8))
        denom = ln - 1.0
        if abs(denom) < 1e-4:
            denom = 1e-4 if denom >= 0 else -1e-4
        reward = 10.0 * (1.0 + 1.0 / denom)
    elif mode == "iou":
        reward = float(iou)
    elif mode == "10_normal_auc":
        reward = 10.0 * float(auc)
    elif mode == "10_auc_gms":
        reward = 10.0 * float(auc_gms)
    elif mode == "5nc_5iou":
        reward = 5 * reward_from_auc_blend(auc) + 5 * float(iou)
    else:
        reward = 10.0 * float(iou)
    return float(np.clip(reward, -10.0, 10.0))


def reward_from_auc_blend(auc, pivot=0.98, L=0.60, beta=3.0, q=3.0, lam=0.5):
    a = np.asarray(auc, dtype=float)
    alpha = L / pivot
    left = alpha * a
    t = np.clip((a - pivot) / (1.0 - pivot), 0.0, 1.0)
    s_power = t**beta
    s_soft = 1.0 - (1.0 - t) ** q
    s_mix = (1 - lam) * s_power + lam * s_soft
    right = L + (1.0 - L) * s_mix
    return np.where(a < pivot, left, right)


__all__ = [
    "compute_cd",
    "compute_iou",
    "compute_metrics_from_mesh_paths",
    "compute_metrics_from_execution",
    "compute_metrics_from_meshes",
    "compute_normals_metrics",
    "reward_from_auc_blend",
    "reward_from_metrics",
    "transform_gt_mesh_cad_0875",
    "transform_mesh_0_1",
]
