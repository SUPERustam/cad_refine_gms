#!/bin/bash
#SBATCH --job-name=rl_gms
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus=4
#SBATCH --exclude=cn24,cn15,cn43,cn44,cn45
#SBATCH --output=logs/rl_gms_train_2.1.out
#SBATCH --error=logs/rl_gms_train_2.1.err

srun bash /scratch/498rustam/cad_refine_m/train_loop_dp_gms_resume_2.1.sh

# cn8 is okey