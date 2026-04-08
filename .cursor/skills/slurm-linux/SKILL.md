---
name: slurm-linux
description: Submits, monitors, and debugs Slurm jobs on Linux (sbatch, squeue, srun, logs). Use when scheduling GPU/CPU work, writing #SBATCH scripts, checking job status, canceling jobs, or when the user mentions Slurm, sbatch, partitions, or cluster jobs.
---

# Slurm on Linux (this project)

## Quick reference

| Goal | Command |
|------|---------|
| Submit batch script | `sbatch path/to/job.sh` |
| Your jobs | `squeue -u $USER` |
| Cluster overview | `sinfo` or `sinfo -Nel` |
| Cancel | `scancel <JOBID>` |
| Job accounting | `sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES%30` |
| Hold details | `scontrol show job <JOBID>` |

## Submitting jobs (repo pattern)

1. **Batch script** uses `#SBATCH` directives, then usually `srun` to launch the real shell script so resource binding matches the allocation.
2. **Logs**: this repo writes under `logs/` (create it if missing: `mkdir -p logs`).
3. **Paths**: prefer absolute paths inside scripts (e.g. `/scratch/498rustam/cad_refine_m/...`) so the job cwd does not matter.

Example shape (matches `slurm_runner_1.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=my_run
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus=4
#SBATCH --output=logs/my_run.out
#SBATCH --error=logs/my_run.err

srun bash /scratch/498rustam/cad_refine_m/path/to/train_or_infer.sh
```

Submit: `cd /scratch/498rustam/cad_refine_m && sbatch slurm_runner_1.sh`

**Note**: exact flags (`--gpus`, partitions, accounts, time limits) depend on the site’s Slurm config; if a job is rejected, check `sinfo` / cluster docs or ask the admin for required `#SBATCH` lines.

## Interactive debugging

- One-off shell on a compute node: `srun --pty bash` (add `-p <partition>`, `--gres=gpu:1`, etc. as required).
- Short test allocation: `salloc --nodes=1 --cpus-per-task=4 ...` then run commands inside.

## When jobs fail

1. Read `logs/*.err` and `.out` for the jobid you care about.
2. `sacct -j <JOBID> -o JobID,State,ExitCode,Elapsed` — distinguish `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, node issues.
3. If the job never starts: `squeue -j <JOBID>` (pending reason), or `scontrol show job <JOBID>`.

## Environment for training in this repo

Per `AGENTS.md`, local interactive work uses `mamba activate cadtrl`. Batch scripts typically `source` project `.env` and call `accelerate` / `trl` as in `train_loop_*.sh`; keep that pattern unless you intentionally change it.

## Anti-patterns

- Submitting from a directory that breaks relative paths — use absolute paths for entrypoints and critical data.
- Forgetting `mkdir -p logs` before first submission.
- Running multi-GPU training without `srun` when the batch file expects an allocation (can lead to wrong binding on some clusters).
