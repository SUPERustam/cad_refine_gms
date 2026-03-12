1. Error with `bad interpreter: No such file or directory`
Error looks like this (often with `trl` or `accelerate` libraries)
```sh
/scratch/498rustam/miniforge3/envs/cadtrl_restored/bin/trl: /home/jovyan/users/zhemchuzhnikov/miniconda3/envs/cadtrl/bin/python3.10: bad interpreter: No such file or directory
```

To fix it, you should edit file inside library folder. For example, for `trl` library it would be:
```sh
sed -i '1s|.*|#!/scratch/498rustam/miniforge3/envs/cadtrl_restored/bin/python3.10|' /scratch/498rustam/miniforge3/envs/cadtrl_restored/bin/trl && head -1 /scratch/498rustam/miniforge3/envs/cadtrl_restored/bin/trl
```

2. Error `requests.exceptions.ConnectionError` with `vllm` library

Error looks like this:
```sh
[rank0]: requests.exceptions.ConnectionError: HTTPConnectionPool(host='0.0.0.0', port=8000): Max retries exceeded with url: /health/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fb07c23feb0>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

To fix try to increase timeout in `train_loop_dp_*.sh` or other running scripts.
```sh 
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2-VL-2B-Instruct --max_model_len 3600 >"$VLLM_LOG" 2>&1 &
sleep 80 # <- this is the timeout
```

3. Errors with `CadQuery` execution with zero loss

Errors looks like this:
```sh
...
Error executing CadQuery code : 'result'
Error executing CadQuery code : 'result'
...

{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 9.999999879097347e-06, 'entropy': 0.1934378132224083, 'clip_ratio/low_mean': 0.0, ...
```
Edit format of cadquery code using `METRICS_VAR_NAME` environment variable. Default is `result` for CadEvolve format.

For example, for Cadrille format it would be:
```sh
export METRICS_VAR_NAME='r'
```

4. Error `-7` or Out of Memory (OOM)
    -   Decrease `per_device_train_batch_size`.
    -   Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
    -   Ensure `gradient_checkpointing: true` is in the config.

5. Zero Loss / Zero Grad Norm
    - Check `failure_reward` and ensure the model is initialized from a decent SFT checkpoint.
    - Check if hf_dataset have proper paths to stls.