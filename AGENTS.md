# AGENTS

- To run the code: 
    - `mamba activate cadtrl_restored`
    - Edit `slurm/infer.sh` to point to the correct checkpoint and task profile
    - Run `sbatch slurm/infer.sh`
    - Wait for the job to finish and check the output in `slurm/inference.out.log` and `slurm/inference.err.log`

- Root Python files are public CLIs only: `prepare_dataset.py`, `train.py`, `resume_train.py`, `infer.py`, `build_meshes.py`, `evaluate.py`, `compare_runs.py`.
- Reusable code lives under `cad_rl/`. Do not add new root helper modules.
- Use absolute package imports inside `cad_rl/`; never rely on the current working directory.
- Dataset-prep-only rendering code stays under `cad_rl.data`.
- Keep config-driven workflows intact. Prefer updating profiles in `configs/` and package code over adding ad hoc scripts.
- Focused verification: `pytest tests/test_config_runtime.py tests/test_specs_and_artifacts.py tests/test_training_contracts.py tests/test_dataset_boundary.py`.
