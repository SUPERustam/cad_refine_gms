#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --output=slurm/inference.out
#SBATCH --error=slurm/inference.err

srun python cli.py infer \
  --config configs/demo/infer.yaml
