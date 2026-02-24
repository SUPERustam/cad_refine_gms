"""Thin wrapper around `_aoc_gms` for in-memory mesh inputs."""

import logging
from typing import Optional

from one_file_metric import LOGGER, _aoc_gms, _coerce_mesh, make_same_scale, MeshInput


def aoc_gms_from_meshes(
    gt_mesh: MeshInput,
    pred_mesh: MeshInput,
    n_points: int = 8192,
    n_angles: int = 125,
    rel_dist_tol: float = 0.05,
    cube_trick: bool = True,
    pc_cache_enable: bool = False,
    upper_bound_tol_rt: float = 25,
    autofix_sampling: bool = False,
    add_auc: bool = True,
    logger: Optional["logging.Logger"] = None,
) -> tuple[float, list[float], list[float], float]:
    if logger is None:
        logger = LOGGER

    gt_mesh_obj = _coerce_mesh(gt_mesh)
    pred_mesh_obj = _coerce_mesh(pred_mesh)
    gt_mesh_scaled, pred_mesh_scaled = make_same_scale(gt_mesh_obj, pred_mesh_obj)

    return _aoc_gms(
        gt_mesh=gt_mesh_scaled,
        pred_mesh=pred_mesh_scaled,
        n_points=n_points,
        n_angles=n_angles,
        rel_dist_tol=rel_dist_tol,
        cube_trick=cube_trick,
        pc_cache_enable=pc_cache_enable,
        upper_bound_tol_rt=upper_bound_tol_rt,
        autofix_sampling=autofix_sampling,
        add_auc=add_auc,
        logger=logger,
    )
