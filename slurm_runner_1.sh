#!/bin/bash
#SBATCH --job-name=rl_gms
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --gpus=8
#SBATCH --exclude=cn44,cn45
#SBATCH --output=slurm_logs/rl_gms_train.out
#SBATCH --error=slurm_logs/rl_gms_train.err

srun bash /scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_4.sh