#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=slurm/inference_legacy_vllm.out.log
#SBATCH --error=slurm/inference_legacy_vllm.err.log

export VLLM_WORKER_MULTIPROC_METHOD=spawn
srun python examples/inference_vllm.py \
      --stl_dir /scratch/498rustam/datasets/MCB_A_batch/groudtruth/ \
      --out_dir predictions/mcb_a_batch_groundtruth_vllm \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 200 --max_new_tokens 2048 --do_sample --temperature 0.7 \

srun python examples/inference_vllm.py \
      --stl_dir /scratch/498rustam/datasets/fusion360_test_mesh \
      --out_dir predictions/fusion360_test_mesh_vllm \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 200 --max_new_tokens 2048 --do_sample --temperature 0.7 \

srun python examples/inference_vllm.py \
      --stl_dir /scratch/498rustam/datasets/deepcad_test_mesh \
      --out_dir predictions/deepcad_test_mesh_vllm \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 200 --max_new_tokens 2048 --do_sample --temperature 0.7 \