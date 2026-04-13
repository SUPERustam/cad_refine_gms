#!/bin/bash
#SBATCH --job-name=rl_gms
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --output=logs/slurm_rl_gms_train.out
#SBATCH --error=logs/slurm_rl_gms_train.err

srun bash /scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_4.sh

# cn8 is okey