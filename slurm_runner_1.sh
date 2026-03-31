#!/bin/bash
#SBATCH --job-name=cadrille
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=4
#SBATCH --output=slurm_logs/train_loop_dp_mae.out
#SBATCH --error=slurm_logs/train_loop_dp_mae.err

srun bash $HOME/cad_refine_m/train_loop_dp_mae.sh