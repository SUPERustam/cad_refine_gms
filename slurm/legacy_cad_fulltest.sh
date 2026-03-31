#!/bin/bash
#SBATCH --job-name=cad
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=1
#SBATCH --output=slurm/cad_legacy_fulltest.out.log
#SBATCH --error=slurm/cad_legacy_fulltest.err.log

srun python examples/build_meshes.py \
      --dataset mcb \
      --pred_py_path /scratch/498rustam/cad_refine_m/predictions/mcb_a_batch_groundtruth/ \
      --workers 36 \
      --timeout 80

srun python examples/build_meshes.py \
      --dataset deepcad \
      --pred_py_path /scratch/498rustam/cad_refine_m/predictions/deepcad_test_mesh/ \
      --workers 36 \
      --timeout 80

srun python examples/build_meshes.py \
      --dataset fusion360 \
      --pred_py_path /scratch/498rustam/cad_refine_m/predictions/fusion360_test_mesh/ \
      --workers 36 \
      --timeout 80
