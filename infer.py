#!/usr/bin/env python3
import argparse

import torch

from cad_rl.config import ModelSpec, TaskSpec, load_profile_document
from cad_rl.algorithms.frozen import (
    build_generation_kwargs,
    create_processor,
    load_qwen_model,
)
from cad_rl.data.inference_dataset import load_inference_dataset
from cad_rl.pipelines.inference import generate_inference_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run profile-driven inference and emit JSONL records"
    )
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="Inference engine: Hugging Face generate() (default) or in-process vLLM",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Only for --backend vllm: fraction of GPU memory for the KV cache",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Only for --backend vllm: tensor parallel world size",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max sequence length passed to vLLM LLM()",
    )
    parser.add_argument(
        "--raw-recursive",
        action="store_true",
        help="When the split path is a raw STL directory, include *.stl in subdirectories",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="When using a raw STL directory, cap the number of files (after sorting)",
    )
    args = parser.parse_args()

    task_profile = load_profile_document(args.task_profile, profiles_root="configs")
    task_spec = TaskSpec.from_mapping(task_profile)
    model_spec = ModelSpec.from_mapping(
        load_profile_document(args.model_profile, profiles_root="configs")
    )
    processor = create_processor(
        model_spec.processor_name or model_spec.base_checkpoint,
        **dict(model_spec.processor_kwargs),
    )
    dataset = load_inference_dataset(
        task_profile,
        task_spec.prepared_datasets[args.split],
        raw_recursive=args.raw_recursive,
        max_samples=args.max_samples,
    )

    generate_kwargs = build_generation_kwargs(processor)
    generate_kwargs.update(dict(model_spec.generation_defaults))
    tokenizer = processor.tokenizer
    if "eos_token_id" not in generate_kwargs:
        unk = getattr(tokenizer, "unk_token_id", None)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != unk:
            generate_kwargs["eos_token_id"] = im_end_id
        elif tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = tokenizer.eos_token_id
    generate_kwargs.setdefault("pad_token_id", pad)

    if args.backend == "vllm":
        if not torch.cuda.is_available():
            raise RuntimeError("--backend vllm requires CUDA")
        from vllm import LLM

        from cad_rl.pipelines.inference_vllm import (
            generate_inference_records_vllm,
            hf_generate_kwargs_to_vllm_sampling_params,
        )

        llm_kw: dict = {
            "model": args.checkpoint,
            "trust_remote_code": model_spec.trust_remote_code,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": 1},
        }
        if args.vllm_max_model_len is not None:
            llm_kw["max_model_len"] = args.vllm_max_model_len
        llm = LLM(**llm_kw)
        sampling_params = hf_generate_kwargs_to_vllm_sampling_params(
            tokenizer, generate_kwargs
        )
        generate_inference_records_vllm(
            llm=llm,
            processor=processor,
            dataset=dataset,
            output_path=args.output,
            generation_config={
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "sampling_params": sampling_params,
            },
            task_spec=task_spec,
        )
    else:
        model = load_qwen_model(
            args.checkpoint,
            trust_remote_code=model_spec.trust_remote_code,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        generate_inference_records(
            model=model,
            processor=processor,
            dataset=dataset,
            output_path=args.output,
            generation_config={
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "generate_kwargs": generate_kwargs,
            },
            task_spec=task_spec,
        )


if __name__ == "__main__":
    main()
