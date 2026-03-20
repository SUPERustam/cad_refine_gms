#!/usr/bin/env python3
import argparse

from datasets import load_from_disk
import torch

from cad_rl.algorithms.frozen import (
    build_generation_kwargs,
    create_processor,
    load_qwen_model,
)
from cad_rl.config import ModelSpec, TaskSpec, load_profile_document
from cad_rl.pipelines.inference import generate_inference_records


def _load_task_spec(path: str) -> TaskSpec:
    return TaskSpec.from_mapping(load_profile_document(path, profiles_root="configs"))


def _load_model_spec(path: str) -> ModelSpec:
    return ModelSpec.from_mapping(load_profile_document(path, profiles_root="configs"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run profile-driven inference and emit JSONL records"
    )
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    task_spec = _load_task_spec(args.task_profile)
    model_spec = _load_model_spec(args.model_profile)
    processor = create_processor(
        model_spec.processor_name or model_spec.base_checkpoint,
        **dict(model_spec.processor_kwargs),
    )
    model = load_qwen_model(
        args.checkpoint,
        trust_remote_code=model_spec.trust_remote_code,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_from_disk(task_spec.prepared_datasets[args.split])

    generate_kwargs = build_generation_kwargs(processor)
    generate_kwargs.update(dict(model_spec.generation_defaults))
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
