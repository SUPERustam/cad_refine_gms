#!/bin/bash
#SBATCH --job-name=cadrille_rl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus=4
#SBATCH --output=slurm_logs/slurm_log-%j.out

srun bash /scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_2.sh