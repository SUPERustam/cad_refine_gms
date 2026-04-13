# File and directory access (training on Slurm)

Summary of disk access for the chain [`slurm_runner_1.sh`](../slurm_runner_1.sh) → [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) → `trl vllm-serve` + `accelerate launch` [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) + [`configs/gms_config.yaml`](../configs/gms_config.yaml). Other `train_loop_dp_*.sh` scripts follow the same idea with different hard-coded paths.

---

## Access table

| Source | Path or pattern | R/W | Note |
|--------|------------------|-----|------|
| [`slurm_runner_1.sh`](../slurm_runner_1.sh) | `/scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_4.sh` | R | `srun bash …` |
| [`slurm_runner_1.sh`](../slurm_runner_1.sh) | `logs/slurm_rl_gms_train.{out,err}` (relative to submit cwd) | W | Under `/scratch/.../cad_refine_m/logs/` if `sbatch` was run from repo on scratch |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_68000/` (`BASE_DIR` / `--output_dir`) | W | Trainer checkpoints and artifacts |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/checkpoints/sft-30682/` (`--sft_path`) | R | `from_pretrained` |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/rl_gms_train_sft_30682_resume_54000/checkpoint-68000` (`--resume_ckpt_path`) | R | Resume state |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/rl_train_cos_sched.py` | R | Entry script |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/configs/gms_config.yaml` | R | No literal `/scratch` inside YAML |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/logs_rl/<RUN_NAME>.log` | W | `script(1)` log |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `/scratch/498rustam/cad_refine_m/logs_rl/vllm_server.log` | W | `trl vllm-serve` stdout/stderr |
| [`train_loop_dp_gms_resume_4.sh`](../train_loop_dp_gms_resume_4.sh) | `.env` (relative path) | R | Resolved from job **initial cwd**, not script directory |
| [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) | `/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_fixed` | R | Default `HF_DATASET` when `output_dir` lacks `dp_f360` |
| [`rl_train_cos_sched.py`](../rl_train_cos_sched.py) | `/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_deepcadf360` | R | Only if `dp_f360` ∈ `output_dir` |
| Hugging Face Hub / Transformers | `$HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, or `~/.cache/huggingface/…` | R/W | Model `Qwen/Qwen2-VL-2B-Instruct`; `~` → `/scratch/...` if `$HOME` is on scratch |
| Accelerate | under HF cache, e.g. `~/.cache/huggingface/accelerate` | R/W | `accelerate launch` |
| Comet ML | Comet local/offline dirs (often under `$HOME`) | R/W | `report_to: ["comet_ml"]` in `gms_config`; keys in [`.env`](../.env) |
| `TMPDIR` / `TMP` / `TEMP` | site- or Slurm-defined | R/W | PyTorch, vLLM, other temps; may be scratch on cluster |
| `$XDG_CACHE_HOME` | if set | R/W | Replaces default `~/.cache` for many tools |
| Dataset rows (e.g. `mesh_path`) | values inside HF dataset on disk | R | Not literals in [`rewards.py`](../rewards.py) / [`metrics_async.py`](../metrics_async.py); paths depend on dataset build |
| Other scripts (not this chain) | various `/scratch/...` in `train_loop_dp_*.sh`, [`benchmark/eval_cadevolve.py`](../benchmark/eval_cadevolve.py), [`slurm_scripts/`](../slurm_scripts/) | — | See `rg /scratch` in repo if you run them |

---

## Notes

- [`rl_train.py`](../rl_train.py) repeats the same dataset `/scratch` constants but is **not** invoked by `slurm_runner_1.sh` / `train_loop_dp_gms_resume_4.sh`.
- Repo does **not** set `HF_HOME`, `TMPDIR`, or `HOME` in the resume-4 script; indirect scratch I/O depends on the **login/compute environment**.
- Inspect env: `python -c "import os; print('HOME', os.path.expanduser('~')); print('TMPDIR', os.environ.get('TMPDIR')); print('XDG_CACHE_HOME', os.environ.get('XDG_CACHE_HOME')); print('HF_HOME', os.environ.get('HF_HOME'))"`.
- More workflow context: [AGENTS.md](../AGENTS.md), [docs/Setup.md](Setup.md).
