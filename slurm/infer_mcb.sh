#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=2
#SBATCH --output=slurm/inference.out.log
#SBATCH --error=slurm/inference.err.log

# Add e.g. --backend vllm (after mamba env has vllm) for faster inference on GPU.
srun python infer.py \
  --task-profile configs/task.mcb_a_batch_groundtruth_1000_infer.yaml \
  --model-profile configs/model.qwen2_vl.yaml \
  --checkpoint /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
  --split test \
  --output /scratch/498rustam/cad_refine_m/outputs/inference_mcb_a_batch_groundtruth_1000.jsonl \
  --backend vllm

# After inference, run build_meshes with paths appropriate for this run (see build_meshes.py / slurm helpers).
