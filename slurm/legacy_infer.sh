#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=slurm/inference_legacy_transformers.out.log
#SBATCH --error=slurm/inference_legacy_transformers.err.log

srun python examples/inference_cad_model.py \
      --stl_dir /scratch/498rustam/datasets/mcb_a_batch_groundtruth_1000/ \
      --out_dir predictions/mcb_a_batch_groundtruth_1000 \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7

srun python examples/inference_cad_model.py \
      --stl_dir /scratch/498rustam/datasets/mcb_a_batch_groundtruth_1000/ \
      --out_dir predictions/mcb_a_batch_groundtruth_1000 \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7

srun python examples/inference_cad_model.py \
      --stl_dir /scratch/498rustam/datasets/fusion360_test_mesh_1000/ \
      --out_dir predictions/fusion360_test_mesh_1000 \
      --model_path /scratch/498rustam/cad_refine_m/rl_checkpoints_resume_32800/checkpoint-49600 \
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7

