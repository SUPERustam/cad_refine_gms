#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=slurm/cad_legacy_transformers_2.out.log
#SBATCH --error=slurm/cad_legacy_transformers_2.err.log

srun python examples/build_meshes.py \
      --dataset mcb_1000 \
      --pred_py_path /scratch/498rustam/cad_refine_m/predictions/mcb_a_batch_groundtruth_1000_new/ \
      --workers 16 \
      --timeout 80

# srun python examples/build_meshes.py \
#       --dataset deepcad_1000 \
#       --pred_py_path /scratch/498rustam/cad_refine_m/predictions/deepcad_test_mesh_1000/ \
#       --workers 16 \
#       --timeout 80

# srun python examples/build_meshes.py \
#       --dataset fusion360_1000 \
#       --pred_py_path /scratch/498rustam/cad_refine_m/predictions/fusion360_test_mesh_1000/ \
#       --workers 16 \
#       --timeout 80
