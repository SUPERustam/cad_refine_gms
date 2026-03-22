#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=2
#SBATCH --output=slurm_logs/inference_new_model.out
#SBATCH --error=slurm_logs/inference_new_model.err

srun bash $HOME/cad_refine_m/slurm_scripts/inference_new_model_vllm.sh