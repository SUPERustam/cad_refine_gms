import math
from metrics_async import get_metrics_from_texts
from utils import _maybe_print_sample
import numpy as np
import os
from logging_utils import log_event, log_failure_payload, truncate_text

_DEFAULT_VAR_NAME = os.getenv("METRICS_VAR_NAME", "result")
_FALLBACK_VAR_NAME = os.getenv("METRICS_VAR_FALLBACK", "")

def reward_from_metrics(cd: float, iou: float, auc: float = 0, auc_gms: float = 0, mode: str = "default") -> float:
    if cd is None or math.isnan(cd) or cd <= 0: cd = 1.0
    if iou is None or (isinstance(iou, float) and math.isnan(iou)):
        iou = 0.0
    if auc is None or (isinstance(auc, float) and math.isnan(auc)):
        auc = 0.0
    if auc_gms is None or (isinstance(auc_gms, float) and math.isnan(auc_gms)):
        auc_gms = 0.0
    if mode == "10_iou":
        r = 10.0 * float(iou)
    elif mode == "cd_to_reward":
        ln = math.log(max(cd, 1e-8))
        denom = (ln - 1.0)
        if abs(denom) < 1e-4: denom = 1e-4 if denom >= 0 else -1e-4
        r = 10.0 * (1.0 + 1.0 / denom)
    elif mode == "iou":
        r = float(iou)
    elif mode == "10_normal_auc":
        r  = 10.0 * auc
    elif mode == "10_auc_gms":
        r = 10.0 * auc_gms
    elif mode == "5nc_5iou":
        r  = 5 * reward_from_auc_blend(auc) + 5 *iou
    else:
        r = 10.0 * float(iou)
    return float(np.clip(r, -10.0, 10.0))


def get_reward_function(
    failure_reward,
    iou_coef=10,
    cd_coef=0,
    auc_coef=0,
    aoc_gms_coef=0,
    nc_params=None,
    mode="10_iou",
    print_every=50,
    var_name=None,
    logger=None,
    max_logged_completion_chars=4000,
):
    def combined_reward(completions, mesh_path, trainer_state=None, **kwargs):
        vn = var_name or _DEFAULT_VAR_NAME
        global_step = getattr(trainer_state, "global_step", None)
        # Get individual rewards
        rewards = []
        """
        if nc_params.get("get_nc") == True:
            updt_tol = update_step_tol(step=getattr(trainer_state, "global_step", 0))
            nc_params["tol"] = updt_tol"""

        pred_metrics = get_metrics_from_texts(
            completions, mesh_path, nc_params, var_name=vn)
        failure_count = 0
        timeout_count = 0
        crash_count = 0
        for idx, m in enumerate(pred_metrics):
            reward = 0
            iou = m["iou"] if m is not None else None
            cd =  m["cd"] if m is not None else None
            auc =  m["auc"] if m is not None else None
            status = m.get("status", "ok") if m is not None else "missing"
            if iou is None:
                reward = failure_reward
            else:
                use_aoc_gms = (
                    nc_params
                    and nc_params.get("get_aoc_gms")
                    and aoc_gms_coef > 0
                )
                if use_aoc_gms:
                    auc_gms_value = m.get("auc_gms")
                    if auc_gms_value is None:
                        reward = failure_reward
                    else:
                        gms_reward = reward_from_metrics(
                            cd,
                            iou,
                            auc_gms=auc_gms_value,
                            mode="10_auc_gms",
                        )
                        reward = float(
                            np.clip((aoc_gms_coef / 10.0) * gms_reward, -10.0, 10.0)
                        )
                else:
                    reward = reward_from_metrics(cd, iou, auc=auc, mode=None)
            if not math.isfinite(reward): reward = failure_reward
            rewards.append(float(reward))
            if status != "ok":
                failure_count += 1
            if status == "timeout":
                timeout_count += 1
            if status == "crash":
                crash_count += 1
            if status != "ok" and logger is not None:
                log_failure_payload(
                    "reward_sample_failure",
                    status=status,
                    global_step=global_step,
                    sample_idx=idx,
                    generation_idx=idx,
                    mesh_path=mesh_path[idx],
                    completion=truncate_text(completions[idx], max_logged_completion_chars),
                    metrics=m,
                    reward=reward,
                    var_name=vn,
                )

        # ---- print one sample every 50 steps ----
        top_idx = rewards.index(max(rewards))
        top_generation = completions[top_idx]
        top_mesh_path = mesh_path[top_idx]
        _maybe_print_sample(
            top_generation,
            top_mesh_path,
            step=global_step,
            every=print_every,
            logger=logger,
            reward=rewards[top_idx],
            generation_idx=top_idx,
            max_logged_completion_chars=max_logged_completion_chars,
        )
        if logger is not None:
            log_event(
                logger,
                "reward_batch_complete",
                global_step=global_step,
                completion_count=len(completions),
                failure_count=failure_count,
                timeout_count=timeout_count,
                crash_count=crash_count,
                reward_max=max(rewards) if rewards else None,
                reward_min=min(rewards) if rewards else None,
                reward_mean=float(np.mean(rewards)) if rewards else None,
            )
        return rewards
    return combined_reward

def reward_from_auc_blend(auc, pivot=0.98, L=0.60, beta=3.0, q=3.0, lam=0.5):
    """
    exponentially increase reward for increases larger than 0.98
    """
    a = np.asarray(auc, dtype=float)
    alpha = L / pivot
    left = alpha * a
    t = np.clip((a - pivot) / (1.0 - pivot), 0.0, 1.0)
    S_power = t ** beta
    S_soft  = 1.0 - (1.0 - t) ** q
    S_mix   = (1 - lam) * S_power + lam * S_soft
    right = L + (1.0 - L) * S_mix
    return np.where(a < pivot, left, right)
