---
name: inference-speed-experiments
overview: Run a small set of controlled inference experiments to identify whether throughput is limited by tensor parallelism, serial rendering, or overly conservative decoding/runtime settings.
todos:
  - id: exp-single-gpu
    content: Benchmark TP=1 with larger batch sizes on the same dataset and checkpoint.
    status: pending
  - id: exp-no-render-save
    content: Benchmark the same run without writing render PNGs to disk.
    status: pending
  - id: exp-lower-max-tokens
    content: Test smaller max_new_tokens values and check for output truncation.
    status: pending
  - id: exp-two-shards
    content: Compare one TP=2 job against two independent TP=1 shard jobs.
    status: pending
isProject: false
---

# Inference Speed Experiments

## Goal

Identify the fastest low-risk path to improve end-to-end inference throughput for the current vLLM STL pipeline launched from [slurm/legacy_infer.sh](/scratch/498rustam/cad_refine_m/slurm/legacy_infer.sh) and implemented in [examples/inference_vllm.py](/scratch/498rustam/cad_refine_m/examples/inference_vllm.py).

## What The Logs Suggest

- vLLM startup has a one-time cost of about 42s, but steady-state decode looks healthy for batches of 4.
- The current run uses `--vllm_tensor_parallel_size 2` even though each worker only loads about 2.1 GiB of model weights, so TP may be adding communication overhead without helping much.
- The script renders STL images serially with `Plotter.get_img(...)` and also writes every render PNG to disk before/alongside generation, which is a strong candidate bottleneck.

Essential code path:

```228:296:/scratch/498rustam/cad_refine_m/examples/inference_vllm.py
    def flush_batch() -> None:
        nonlocal batch_imgs, batch_paths
        if not batch_imgs:
            return

        vllm_inputs = build_vllm_inputs_for_images(batch_imgs, processor)
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
...
    for stl in stls:
        try:
            img = plotter.get_img(stl, None, apply_augs=args.apply_augs)
            save_img_any(...)
            batch_imgs.append(img)
            batch_paths.append(stl)
```

## Experiments

1. Single-GPU vLLM baseline

Run the same workload with `--vllm_tensor_parallel_size 1` and increase `--batch_size` from `4` to `16`, then `32` if memory allows.
Success signal: better total job time and better prompts/sec without OOM.
Why: avoids TP/NCCL overhead and exploits the large memory headroom seen in the logs.

1. Remove render write overhead

Repeat experiment 1 but disable saving `renders/*.png` so the job only renders in memory and writes predictions.
Success signal: noticeable drop in wall-clock time with similar model output quality.
Why: the current loop performs 1000 PNG writes that are not needed for pure inference benchmarking.

1. Reduce generation budget

Starting from the best result above, lower `--max_new_tokens` from `2048` to `1024`, and optionally `512` if outputs remain complete.
Success signal: lower long-tail batch latency without truncating valid outputs.
Why: current decode budget is likely much larger than necessary for many samples.

1. Two independent single-GPU shards

Instead of one 2-GPU TP job, split the STL list into two shards and run two independent single-GPU inference jobs in parallel.
Success signal: higher aggregate throughput than TP=2 with similar output quality.
Why: for a small model, data-parallel job sharding usually outperforms tensor parallel inference.

## Measurement Guidance

- Use the same dataset slice and checkpoint for all runs.
- Record total wall time, number of completed predictions, and whether any outputs truncate.
- Compare end-to-end job duration, not just vLLM token speed, because rendering and I/O are likely major contributors.

## Likely Next Change After Experiments

If experiments 1 and 2 win, the first code change should be to add optional render saving and optional dataset sharding to [examples/inference_vllm.py](/scratch/498rustam/cad_refine_m/examples/inference_vllm.py), then update [slurm/legacy_infer.sh](/scratch/498rustam/cad_refine_m/slurm/legacy_infer.sh) to use the best configuration.