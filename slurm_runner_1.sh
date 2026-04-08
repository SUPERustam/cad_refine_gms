#!/bin/bash
#SBATCH --job-name=rl_mae_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --gpus=8
#SBATCH --exclude=cn44,cn45
#SBATCH --output=slurm_logs/train_loop_dp_mae_2.out
#SBATCH --error=slurm_logs/train_loop_dp_mae_2.err

srun bash $HOME/cad_refine_m/train_loop_dp_mae.sh