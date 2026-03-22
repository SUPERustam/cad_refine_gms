---
name: Add MAE RL metric
overview: Add a render-based `mae_similarity` metric to the shared async metric pipeline, wire it into RL reward/config selection, and add a new config plus a thin single-run launch script based on the existing GMS launcher.
todos:
  - id: add-mae-metric
    content: Extend `metrics_async.py` with render generation helpers and `mae_similarity`, returning the new metric in the per-sample result dict.
    status: completed
  - id: wire-reward-config
    content: Add MAE reward/config plumbing in `rewards.py`, `rl_train_cos_sched.py`, and likely `rl_train.py`, while cleaning up metric-specific reward signatures.
    status: completed
  - id: update-callers
    content: Adjust evaluation/benchmark callers to tolerate or expose the new metric result field without breaking existing workflows.
    status: completed
  - id: new-config-script
    content: Create a new MAE-focused config under `configs/` and a single-run launcher script derived from `train_loop_dp_gms.sh` with no restart loop and no MAE params hardcoded in shell.
    status: completed
  - id: todo-1773757797830-r8h8f7y4n
    content: ""
    status: pending
isProject: false
---

# Add MAE RL Metric

## Scope

Implement a new render-based metric equivalent to:

```python
def mae_similarity(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")
    mae = np.mean(np.abs(A - B))
    return float(1.0 - mae / 255.0)
```

and make it selectable as an RL reward metric through config, without hardcoding the new metric settings in the launch shell script.

## Planned Changes

- Extend `[/scratch/hsegrif/cad_refine_m/metrics_async.py](/scratch/hsegrif/cad_refine_m/metrics_async.py)` to compute a render-based similarity score for each `(generated_code, gt_mesh_path)` sample.
  - Reuse the existing render stack rather than inventing a new one, ideally via `[/scratch/hsegrif/cad_refine_m/helper_visu.py](/scratch/hsegrif/cad_refine_m/helper_visu.py)` / `[/scratch/hsegrif/cad_refine_m/multiview_dataset.py](/scratch/hsegrif/cad_refine_m/multiview_dataset.py)`, so RL reward-time rendering matches dataset/eval rendering as closely as possible.
  - Add a small helper that converts GT and predicted meshes into same-shape RGB arrays and computes `mae_similarity`.
  - Return the new value in the metrics payload alongside existing fields (`cd`, `iou`, `auc`, `auc_gms`) so downstream consumers can opt in without breaking existing metrics.
- Update reward plumbing in `[/scratch/hsegrif/cad_refine_m/rewards.py](/scratch/hsegrif/cad_refine_m/rewards.py)` so MAE similarity can be used as the active RL reward.
  - Add a reward mode for MAE similarity, keeping the current `r_mode` selection pattern already used by `10_iou` and `10_normal_auc`.
  - Add a dedicated coefficient/config field for MAE if needed, so the config controls reward scaling instead of the shell script.
  - Clean up the reward helper signature at the same time so metric-specific branches are explicit; this also avoids the current mismatch where `reward_from_metrics()` is called with `auc_gms=...` even though that parameter is not in the function signature.
- Add MAE config fields to the RL entrypoints so YAML fully describes the new reward path.
  - Update `[/scratch/hsegrif/cad_refine_m/rl_train_cos_sched.py](/scratch/hsegrif/cad_refine_m/rl_train_cos_sched.py)` `RewardArgs` and the `nc_params`/metric-settings dict to include MAE-render options.
  - Mirror the same additions in `[/scratch/hsegrif/cad_refine_m/rl_train.py](/scratch/hsegrif/cad_refine_m/rl_train.py)` if you want both training entrypoints to stay feature-compatible.
  - Keep the public interface config-driven, similar to existing fields such as `get_aoc_gms`, `aoc_gms_coef`, and `r_mode`.
- Update non-training callers that consume `metrics_async.py` so the new metric is available consistently.
  - Check `[/scratch/hsegrif/cad_refine_m/evaluate_model.py](/scratch/hsegrif/cad_refine_m/evaluate_model.py)` and `[/scratch/hsegrif/cad_refine_m/benchmark/eval_cadevolve.py](/scratch/hsegrif/cad_refine_m/benchmark/eval_cadevolve.py)` for any assumptions about the returned metric dict.
  - At minimum, make sure they do not break when `mae_similarity` is added; optionally expose/log it if that is useful for offline evaluation.
- Add a new YAML config in `[/scratch/hsegrif/cad_refine_m/configs](/scratch/hsegrif/cad_refine_m/configs)` for the MAE-based run.
  - Base it on the current GMS launch intent from `[/scratch/hsegrif/cad_refine_m/train_loop_dp_gms.sh](/scratch/hsegrif/cad_refine_m/train_loop_dp_gms.sh)`, but move the training/reward knobs into YAML instead of passing them inline on the `accelerate launch` command.
  - Include `r_mode` and any MAE-specific toggles/coefs in the YAML so the shell script only points to this config.
  - Likely outcome: a new file such as `configs/mae_config.yaml` (exact name can be chosen during implementation).
- Create a new single-run shell launcher based on `[/scratch/hsegrif/cad_refine_m/train_loop_dp_gms.sh](/scratch/hsegrif/cad_refine_m/train_loop_dp_gms.sh)`.
  - Remove the `while true` restart loop and checkpoint-cycling logic, per your request.
  - Keep only launch concerns in shell: environment loading, optional `METRICS_VAR_NAME`, vLLM startup, trap/cleanup, and a single `accelerate launch ... --config <new_yaml>` call.
  - Avoid hardcoding the newly added MAE parameters in shell; they should live only in the config.
  - Improve script hygiene with common bash best practices already missing in the existing launcher: `set -euo pipefail`, repo-relative path resolution, quoted vars, and cleanup on exit.

## Key Existing References

Current GMS wrapper hardcodes many training flags inline instead of relying on config:

```25:31:/scratch/hsegrif/cad_refine_m/train_loop_dp_gms.sh
CMD='script --flush ${LOG_FILE} \
--command "COMET_API_KEY=${COMET_API_KEY} COMET_PROJECT_NAME=${COMET_PROJECT_NAME} COMET_WORKSPACE=${COMET_WORKSPACE} CUDA_VISIBLE_DEVICES=2,3,4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch ${LAUNCH_SCRIPT} --config ${CONFIG_FILE} --importance_sampling_level token --output_dir ${BASE_DIR} \
--run_name ${RUN_NAME} --use_vllm true --max_completion_length 3500 --temperature 1 --top_p 1 --top_k 100 \
--pool_size 40 --num_generations 16 --top_samples 4 --generation_batch_size 384 --per_device_train_batch_size 1 --max_prompt_length 600 \
--sft_path ${CHECKPOINT} \
--learning_rate 1e-5 --failure_reward 0 --gradient_accumulation_steps 2  --resume_ckpt_path ${RESUME} --save_steps 400"'
```

Current metric payload is centralized here and is the right insertion point for MAE:

```265:330:/scratch/hsegrif/cad_refine_m/metrics_async.py
def get_metrics_from_single_text(text, gt_file, n_points, nc_params=None, var_name="result"):
    # ...
    cd, iou, auc, auc_gms = None, None, None, None
    try: 
        gt_mesh = trimesh.load_mesh(gt_file)
        gt_mesh = transform_mesh_0_1(gt_mesh)
        pred_mesh = transform_mesh_0_1(pred_mesh)

        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        try:
            iou = compute_iou(gt_mesh, pred_mesh)
        except Exception as e:
            print(f"IoU error for {base_file}: {e}", flush=True)
            iou = None
            # ... existing optional metric branches ...
    # ...
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc, auc_gms=auc_gms)
```

Current reward/config plumbing already supports metric selection through `r_mode` and reward args:

```26:49:/scratch/hsegrif/cad_refine_m/rl_train_cos_sched.py
@dataclass
class RewardArgs:
    failure_reward: float = -10.0
    iou_coef: float = 10.0
    cd_coef: float = 0.0
    auc_coef: float = 0.0
    aoc_gms_coef: float = 0.0
    get_nc: bool = False
    nc_n_points: int = 16384
    nc_tol: int = 5
    print_sample_steps: int = 25
    pool_size: int = 16
    r_mode: str = "10_iou"
    get_aoc_gms: bool = False
    aoc_gms_n_points: int = 8192
    aoc_gms_n_angles: int = 125
    aoc_gms_rel_tol: float = 0.05
```

## Implementation Notes

- Because the reward function currently receives `completions` plus `mesh_path`, the most robust MAE design is to render GT and prediction inside the metric worker from mesh/code, rather than depend on trainer-only image tensors.
- If rendering is expensive, I will keep the MAE branch gated by config so existing IoU/AOC runs do not pay that cost.
- I will preserve backward compatibility for existing configs/scripts as much as possible, only adding optional fields and the new launcher/config.

