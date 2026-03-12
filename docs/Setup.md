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
mkdir -p logs_rl/ # for saving logs from the RL training
mkdir -p slurm_logs/ # for saving logs from the SLURM training
```

3. Prepare the Dataset
You need a pre-rendered HuggingFace dataset. This is created using `create_hf_dataset.py`.
- Set `STLS_ROOT` to the folder containing your STLs.
- Set `SPLIT` to the path of the txt file containing validation split folder names.
- Run `python create_hf_dataset.py`.

4. Setup `con

## Reinforcement Learning (RL)

For detailed information on the RL implementation and how to run it, see:
- [RL Theory](RL_Theory.md): Explains the GRPO/CPPO algorithm and reward logic.
- [RL Practical Guide](RL_Practical.md): Instructions for launching training 
- [Troubleshooting](Troubleshooting.md): Troubleshooting guide.

4. `train_loop_dp_*.sh` scripts are used to launch training in a loop. They are responsible for launching the vllm server and the training script.

```bash
BASE_DIR= # path for saving RL checkpoints
CHECKPOINT= # path for saving SFT checkpoint
RUN_NAME= # name for the run, for the experiment tracking
```
