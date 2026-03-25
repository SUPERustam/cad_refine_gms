#!/bin/bash
#SBATCH --job-name=build_meshes_f360
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/build_meshes_fusion360.out.log
#SBATCH --error=slurm/build_meshes_fusion360.err.log

# Run after inference completes and outputs/inference_fusion360_test_mesh_1000.jsonl is populated.
srun python build_meshes.py \
  --input /scratch/498rustam/cad_refine_m/outputs/inference_fusion360_test_mesh_1000.jsonl \
  --output-dir /scratch/498rustam/cad_refine_m/predictions/fusion360_test_mesh_1000 \
  --var-name r
