#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=2
#SBATCH --output=logs/inference_new_model.out
#SBATCH --error=logs/inference_new_model.err

srun bash /scratch/498rustam/cad_refine_m/slurm_scripts/inference_new_model_vllm.sh