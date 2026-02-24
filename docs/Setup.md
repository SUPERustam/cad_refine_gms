1. Create `.env` file in the root directory of the project.
```bash
COMET_API_KEY= # your Comet API key
COMET_PROJECT_NAME= # your Comet project name
COMET_WORKSPACE= # your Comet workspace name
HF_TOKEN= # your Hugging Face token

2. Create some dirs
```bash
mkdir -p rl_checkpoints/ # for saving RL checkpoints
mkdir -p checkpoints/ # for saving SFT checkpoints
mkdir -p logs_rl/ # for saving logs from the RL training
mkdir -p slurm_logs/ # for saving logs from the SLURM training
```
3. Setup `con

2. `train_loop_dp_*.sh` scripts are used to launch training in a loop. They are responsible for launching the vllm server and the training script.

```bash
BASE_DIR= # path for saving RL checkpoints
CHECKPOINT= # path for saving SFT checkpoint
RUN_NAME= # name for the run, for the experiment tracking
```
