# Setup

1. Create `.env` file in the root directory of the project.

```bash
COMET_API_KEY= # your Comet API key
COMET_PROJECT_NAME= # your Comet project name
COMET_WORKSPACE= # your Comet workspace name
HF_TOKEN= # your Hugging Face token
```

2. Create some dirs

```bash
mkdir -p rl_checkpoints/ # for saving RL checkpoints
mkdir -p checkpoints/ # for saving SFT checkpoints
mkdir -p logs/ # for saving logs from the RL training
mkdir -p slurm_logs/ # for saving logs from the SLURM training
```

3. Prepare the Dataset

You need a pre-rendered HuggingFace dataset. This is created using `create_hf_dataset.py`.

- Set `STLS_ROOT` to the folder containing your STLs.
- Set `SPLIT` to the path of the txt file containing validation split folder names.
- Run `python create_hf_dataset.py`.

4. Python environment

Use the project conda/mamba environment (see [AGENTS.md](../AGENTS.md)), for example:

```bash
mamba activate cadtrl_restored
```

5. Train loop configuration

- **`train_loop_dp_gms_resume_4_tmp.sh`** (used by [`slurm_runner.sh`](../slurm_runner.sh)) loads paths and toggles from **`configs/train_loop_dp_gms_resume_4_tmp.env`**. Override the file path with `export TRAIN_LOOP_CONFIG=/path/to/custom.env` before launching.
- **`train_loop_dp_gms_resume_4.sh`** keeps a minimal inline loop (paths edited directly in that script); it does not use the `.env` file next to the script unless you run it from repo root with `source .env`.

6. After a run with `/tmp` staging, you can compare scratch vs `/tmp` session dirs with **`scripts/tmp_staging_status.sh`** (see [docs/file_access.md](file_access.md) and [docs/RL_Practical.md](RL_Practical.md)).

## Reinforcement Learning (RL)

For detailed information on the RL implementation and how to run it, see:

- [RL Theory](RL_Theory.md): Explains the GRPO/CPPO algorithm and reward logic.
- [RL Practical Guide](RL_Practical.md): Instructions for launching training.
- [Troubleshooting](Troubleshooting.md): Troubleshooting guide.

`train_loop_dp_*.sh` scripts launch the vLLM server and the training script. Typical variables (names differ between the simple loop and the tmp + env config) include output directory, SFT checkpoint path, and run name for logging.
