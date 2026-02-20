
## Train with HuggingFace TRL and vLLM acceleration

### -> Create mesh dataset by pre-rendering the stls and serializing the images
in `create_hf_dataset.py` 

change `STLS_ROOT` - main folder containing STLs in subfolders, `SPLIT` - path to the txt file that contains the folder names of the validation split to run training on

The dataset is created in the format of STLImageToCode dataset, with the 7 angles isotopic rendering. To change the original dataset format simply add a format to multiview_dataset.py
run `python create_hf_dataset.py`

This will take a bit of time.
the stls are being rendered in PIL image format, which is then serialized into Arrow dataset. There is also additional pre-processing to match the image padding being done by qwen in process_vision_info, with the fetch_image function

### -> Launch training with vLLM generator worker
in `rl_train.py`

set `HF_DATASET` for the path to the generated HF dataset

Training can be launched on multiple GPU, based on the specified config : 
`CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch rl_train.py --config config.yaml`

Training arguments to modify the defaults and the config can be passed via the command line : 

`CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch rl_train.py --config config.yaml --importance_sampling_level sequence --output_dir test --run_name test --use_vllm false  --temperature 1 --learning_rate 3e-5 --save_steps 50 --failure_reward 0.0 --top_k 50 --max_completion_length 700 --num_generations 16 --top_samples 4 --max_prompt_length 200 --per_device_train_batch_size 4 --generation_batch_size 192"`

Additional explanations on GRPO config arguments are here 
https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig 

I recommend setting `generation_batch_size` to the maximum number of samples you'll have per batch, as vLLM uses kv caches to reduce memory and can handle many prompts in parallel. That is set `generation_batch_size` to **per_device_train_batch_size * num_generations * num_gpus** (4 * 16 * 3 = 192 in this example)

Some additional arguments to consider : 
- print_sample_steps : how often to print ou top generated samples, set -1 for never
- report_to : which logger to use, defaults to comet_ml, can use wandb instead. To use either, corresponding env variables must be set, for example 

`export COMET_WORKSPACE=marinabar && export COMET_API_KEY=xxxx && export COMET_PROJECT_NAME=cad`
- logging_steps : how often to log metrics to terminal and to external logger

To accelerate generation of training samples, use an external vLLM generator server 
1. first launch via `CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 1000`
where max_model_len is the prompt length + max_completion_length 
2. launch training without --use_vllm false (removing is equivalent to  --use_vllm true) 


__Note : during training, to fit a bigger batch size if OOM arises, it is possible to try setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`__