# File and directory access (training on Slurm)

## Which train loop?

| Entry | Script | Config |
|--------|--------|--------|
| Slurm in this repo | [`slurm_runner.sh`](../slurm_runner.sh) runs [`train_loop_dp_gms_resume_4_tmp.sh`](../train_loop_dp_gms_resume_4_tmp.sh) | [`configs/train_loop_dp_gms_resume_4_tmp.env`](../configs/train_loop_dp_gms_resume_4_tmp.env) (override with `TRAIN_LOOP_CONFIG`) |
| Simple local loop (no `/tmp` staging) | [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | Paths inlined in the shell script |

Chain: train loop → `trl vllm-serve` + `accelerate launch` [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) + [`configs/gms_config.yaml`](../configs/gms_config.yaml). Other `train_loop_dp_*.sh` scripts follow the same idea with different paths.

---

## Optional `/tmp` staging (`train_loop_dp_gms_resume_4_tmp.sh`)

When **`ENABLE_TMP_STAGING=1`** in [`configs/train_loop_dp_gms_resume_4_tmp.env`](../configs/train_loop_dp_gms_resume_4_tmp.env), the tmp train loop copies selected read-heavy trees under **`TMP_SESSION`** (under `TMP_ROOT`, name `session_${RUN_NAME}_$$`), points runtime `--output_dir` / `--sft_path` / `--resume_ckpt_path` at `/tmp`, and **rsyncs** writable outputs back to scratch.

| Phase | Location | Note |
|-------|-----------|------|
| Canonical scratch paths | `BASE_DIR_SCRATCH`, `CHECKPOINT_SCRATCH`, `RESUME_SCRATCH`, dataset roots | Set in `configs/train_loop_dp_gms_resume_4_tmp.env` |
| Runtime training I/O | `BASE_DIR`, `--sft_path`, `--resume_ckpt_path` | Under `$TMP_SESSION/…` when staging is on |
| Periodic sync | `BASE_DIR_TMP` → `BASE_DIR_SCRATCH` | After each new `checkpoint-*` once `trainer_state.json` exists ([`scripts/tmp_staging_lib.sh`](../scripts/tmp_staging_lib.sh)) |
| Final sync | Same + optional logs / HF cache | `EXIT` trap in the train loop |

Toggles in the env file include: `STAGE_*`, `SYNC_BACK_*`, `STAGE_HF_CACHE`, `CKPT_WATCH_POLL_SEC`, `HF_CACHE_SCRATCH`, `STAGE_LOGS_TO_TMP`, etc.

- **Dataset on `/tmp`:** when `STAGE_DATASET=1`, the loop sets **`HF_DATASET_OVERRIDE`**; [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) loads that path instead of the default scratch dataset constants.
- **HF cache on `/tmp`:** when `STAGE_HF_CACHE=1`, the loop sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` under the session tree.

**Inspect sync health:** [`scripts/tmp_staging_status.sh`](../scripts/tmp_staging_status.sh) (same `TRAIN_LOOP_CONFIG`, compares latest `checkpoint-*` and sizes under `/tmp` vs scratch). Options: `--dry-run-sync`, `--grep-logs`.

**Slurm smoke copy** (manual round-trip, not the live train loop): [`slurm_scripts/copy_paster.sh`](../slurm_scripts/copy_paster.sh) sources the **simple** [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) with `TRAIN_LOOP_PATHS_ONLY=1` for path variables only. To align copy tests with the tmp train loop, use `TRAIN_LOOP_PATHS_ONLY=1 source train_loop_dp_gms_resume_4_tmp.sh` instead (after pointing `TRAIN_LOOP_CONFIG` at the same env file).

---

## Access table (tmp train loop + Python)

| Source | Path or pattern | R/W | Note |
|--------|------------------|-----|------|
| [`slurm_runner.sh`](../slurm_runner.sh) | [`train_loop_dp_gms_resume_4_tmp.sh`](../train_loop_dp_gms_resume_4_tmp.sh) | R | `srun bash …` |
| Slurm | `logs/slurm_*.{out,err}` (relative to submit cwd) | W | e.g. repo [`logs/`](../logs/) |
| [`configs/train_loop_dp_gms_resume_4_tmp.env`](../configs/train_loop_dp_gms_resume_4_tmp.env) | `BASE_DIR_SCRATCH`, checkpoints, toggles | R | Sourced by train loop; edit for config-driven runs |
| Train loop (staging on) | `$TMP_SESSION/train_output` | R/W | Mirror of scratch output during run |
| Train loop (staging on) | `$TMP_SESSION/sft_checkpoint`, `resume_ckpt`, optional `hf_dataset`, `hf_home` | R/W | Staged inputs / cache |
| [`train_loop_dp_gms_resume_4_tmp.sh`](../train_loop_dp_gms_resume_4_tmp.sh) | [`rl_train_cos_sched.py`](../rl_train_cos_sched.py), [`configs/gms_config.yaml`](../configs/gms_config.yaml) | R | Paths from env `LAUNCH_SCRIPT`, `CONFIG_FILE` |
| Train loop | `LOG_FILE_SCRATCH` → `logs/<RUN_NAME>.log` (via `REPO_ROOT`) | W | `script(1)`; optional `STAGE_LOGS_TO_TMP` |
| Train loop | `VLLM_LOG_SCRATCH` | W | `trl vllm-serve` |
| Train loop | `.env` in repo root | R | `SCRIPT_DIR/.env` next to the script |
| [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) | default dataset under `/scratch/498rustam/datasets/...` | R | Unless `HF_DATASET_OVERRIDE` is set |
| [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) | `$HF_DATASET_OVERRIDE` | R | Staged dataset path when `STAGE_DATASET=1` |
| Hugging Face / Accelerate | `$HF_HOME`, hub/transformers cache | R/W | Can be redirected to `/tmp` when `STAGE_HF_CACHE=1` |
| Comet ML | keys in `.env`, local dirs under `$HOME` | R/W | `report_to` in `gms_config` |
| `TMPDIR` / `XDG_CACHE_HOME` | cluster defaults | R/W | PyTorch, vLLM, other temps |
| Dataset rows (`mesh_path`, etc.) | inside HF dataset on disk | R | See [`rewards.py`](../rewards.py), [`metrics_async.py`](../metrics_async.py) |

### Simple train loop (`train_loop_dp_gms_resume_4.sh`)

Inline `BASE_DIR`, `CHECKPOINT`, `RESUME`, `logs/…`; no `BASE_DIR_SCRATCH` names; no `/tmp` staging. `.env` is loaded from **current working directory** (`source .env`), not necessarily repo root.

---

## Notes

- [`rl_train.py`](../rl_train.py) is not invoked by these Slurm chains.
- With **staging off** (or the simple train loop), the shell does not set `HF_HOME` / `TMPDIR`; behavior depends on the compute environment.
- Inspect env: `python -c "import os; print('HOME', os.path.expanduser('~')); print('TMPDIR', os.environ.get('TMPDIR')); print('XDG_CACHE_HOME', os.environ.get('XDG_CACHE_HOME')); print('HF_HOME', os.environ.get('HF_HOME'))"`.
- More workflow context: [AGENTS.md](../AGENTS.md), [docs/Setup.md](Setup.md), [docs/RL_Practical.md](RL_Practical.md).
