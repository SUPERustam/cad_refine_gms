#!/usr/bin/env bash
set -uo pipefail


DELAY=5
BASE_DIR="/workspace-SR008.fs2/users/barannikov/cad_refine_rl/qwen_cadevolve_mix_lrlr5e_6_temp12"
CHECKPOINT="/workspace-SR008.fs2/users/barannikov/cad_refine_rl/qwen_cadevolve_mix_lrlr5e_6_temp1/checkpoint-6400"
RESUME="False"
#RESUME="/workspace-SR008.nfs2/users/barannikov/cad_refine_rl/qwen_cadevolve_mix_lr2e5_temp1/checkpoint-200"

CMD='script --flush logs/qwen_cadevolve_mix_lrlr5e_6_temp12.txt \
--command "COMET_API_KEY=CfQGtyWGF13CZEsUvXBeuPaSf COMET_PROJECT_NAME=cad COMET_WORKSPACE=marinabar  CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch rl_train.py --config config.yaml --importance_sampling_level token --output_dir $BASE_DIR \
--run_name qwen_cadevolve_mix_lrlr5e_6_temp12 --use_vllm true --max_completion_length 3000 --temperature 1.2 --top_p 1 --top_k 50 \
--pool_size 40 --num_generations 16 --top_samples 4 --generation_batch_size 384 --per_device_train_batch_size 2 --max_prompt_length 600 \
--sft_path $CHECKPOINT \
--learning_rate 5e-6 --failure_reward -1 --gradient_accumulation_steps 4 --resume_ckpt_path $RESUME --save_steps 400"'

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