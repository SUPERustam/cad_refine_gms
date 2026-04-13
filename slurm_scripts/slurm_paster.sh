#!/bin/bash
#SBATCH --job-name=paster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=logs/slurm_paster.out
#SBATCH --error=logs/slurm_paster.err

srun bash /scratch/498rustam/cad_refine_m/slurm_scripts/copy_paster.sh
# cn8 is okey