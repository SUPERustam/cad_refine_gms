#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --output=slurm/inference.out.log
#SBATCH --error=slurm/inference.err.log

# Add e.g. --backend vllm (after mamba env has vllm) for faster inference on GPU.
srun python infer.py \
  --task-profile configs/task.fusion360_test_mesh_1000_infer.yaml \
  --model-profile configs/model.qwen2_vl.yaml \
  --checkpoint /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
  --split test \
  --output /scratch/498rustam/cad_refine_m/outputs/inference_fusion360_test_mesh_1000.jsonl

# After inference, materialize STL meshes under predictions/fusion360_test_mesh_1000/meshes/:
#   sbatch slurm/build_meshes_fusion360.sh
