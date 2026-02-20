#!/usr/bin/env bash
set -uo pipefail

DELAY=5
BASE_DIR="/workspace-SR008.fs2/users/elistratov/rl/_only_rotated_rl_mcb_2_ckpt_21600"
CHECKPOINT="/workspace-SR008.fs2/users/elistratov/rl/_only_rotated_rl_mcb_1/checkpoint-21600"
RESUME="False"

CMD='script --flush /workspace-SR008.fs2/users/elistratov/rl/logs/_only_rotated_rl_mcb_2_ckpt_21600.log \
--command "COMET_API_KEY=SX2x3VnbCdXfDzITcfo1gSpn1 COMET_PROJECT_NAME=cadevolve COMET_WORKSPACE=elistratovm CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch /workspace-SR008.fs2/users/barannikov/cad_refine_rl/rl_train_cos_sched.py --config /workspace-SR008.fs2/users/elistratov/rl/config.yaml --importance_sampling_level token --output_dir $BASE_DIR \
--run_name _only_rotated_rl_mcb_2_ckpt_21600 --use_vllm true --max_completion_length 3500 --temperature 1 --top_p 1 --top_k 100 \
--pool_size 40 --num_generations 16 --top_samples 4 --generation_batch_size 384 --per_device_train_batch_size 2 --max_prompt_length 600 \
--sft_path $CHECKPOINT \
--learning_rate 1e-5 --failure_reward 0 --gradient_accumulation_steps 2  --resume_ckpt_path $RESUME --save_steps 400"'

while true; do
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 >/dev/null 2>&1 &
  sleep 70
  eval "$CMD"

  pkill -9 -f 'cadtrl|VLLM|vllm'  || true

  number=$(ls -d "$BASE_DIR"/checkpoint-* 2>/dev/null \
    | sed 's/.*checkpoint-//' \
    | sort -n \
    | tail -n 1|| true)

  if [[ -n "$number" ]]; then
    CHECKPOINT="$BASE_DIR/checkpoint-$number"
    echo "[$(date)] New CHECKPOINT: $CHECKPOINT"
  fi
  RESUME=$CHECKPOINT

  echo "[$(date)] exited; restarting in ${DELAY}s..."
  sleep "$DELAY"
done