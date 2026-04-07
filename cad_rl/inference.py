from __future__ import annotations

import json
import logging
import time
from pathlib import Path

try:  # pragma: no cover - optional inference dependency
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - lightweight environments
    torch = None
    DataLoader = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None

import cad_rl.config
import cad_rl.data
import cad_rl.modeling
import cad_rl.runtime
try:  # pragma: no cover - optional training dependency
    import cad_rl.grpo
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.grpo = None  # type: ignore[attr-defined]


def _collate_for_qwen(batch, processor):
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": sample["image"]}]}]
        for sample in batch
    ]
    texts = [
        processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    if process_vision_info is None:
        images = [sample["image"] for sample in batch]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    else:
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=texts, images=images, videos=videos, padding=True, return_tensors="pt"
        )
    inputs["mesh_path"] = [str(sample["mesh_path"]) for sample in batch]
    return inputs


def resolve_checkpoint_reference(
    config: cad_rl.config.RunConfig,
    *,
    registry: cad_rl.runtime.RunRegistry | None = None,
) -> cad_rl.runtime.CheckpointRef:
    run_id = str(config.runtime.run_id or config.experiment_id)
    checkpoint_reference = config.infer.checkpoint
    if checkpoint_reference is None:
        if config.infer.checkpoint_kind in {"latest", "best"}:
            checkpoint_reference = config.infer.checkpoint_kind
        else:
            return cad_rl.runtime.CheckpointRef(
                run_id=run_id,
                step=None,
                path=str(config.model.base_checkpoint),
                kind="specific",
            )
    active_registry = registry or cad_rl.runtime.RunRegistry(config.system.run_root)
    return cad_rl.runtime.select_checkpoint(
        run_id, active_registry, checkpoint_reference
    )


def generate_inference_records(
    model,
    processor,
    dataset,
    output_path: str | Path,
    generation_config: dict,
    task_spec: cad_rl.config.TaskSpec,
    *,
    run_id: str,
    checkpoint: cad_rl.runtime.CheckpointRef,
) -> Path:
    if torch is None or DataLoader is None:
        raise RuntimeError("torch is required for inference")
    if cad_rl.grpo is None:
        raise RuntimeError("cad_rl.grpo dependencies are not available")
    cad_rl.grpo.configure_process_environment()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    var_name = task_spec.output_var_name or "result"

    with output_path.open("w", encoding="utf-8") as handle:
        loader = DataLoader(
            dataset,
            batch_size=generation_config.get("batch_size", 8),
            shuffle=False,
            num_workers=generation_config.get("num_workers", 0),
            collate_fn=lambda batch: _collate_for_qwen(batch, processor),
        )
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                    if key != "mesh_path"
                }
                started = time.time()
                generated = model.generate(
                    **inputs, **generation_config["generate_kwargs"]
                )
                prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
                generated_list = generated.tolist()
                trimmed = [
                    generated_list[idx][prompt_lengths[idx] :]
                    for idx in range(len(prompt_lengths))
                ]
                texts = processor.tokenizer.batch_decode(
                    trimmed, skip_special_tokens=True
                )
                for mesh_path, text in zip(batch["mesh_path"], texts):
                    record = cad_rl.runtime.InferenceRecord(
                        run_id=run_id,
                        checkpoint=checkpoint,
                        sample_id=Path(mesh_path).stem,
                        raw_generation=text,
                        wrapped_code=text,
                        status="ok",
                        duration_s=time.time() - started,
                        metadata={
                            "task_id": task_spec.task_id,
                            "target_mesh_path": mesh_path,
                            "output_variable": var_name,
                        },
                    )
                    handle.write(
                        json.dumps(cad_rl.config.to_serializable(record)) + "\n"
                    )
    return output_path


def run_inference_from_resolved(config: cad_rl.config.RunConfig) -> Path:
    log_path = cad_rl.runtime.setup_run_logging(config, stage="infer")
    logger = logging.getLogger(__name__)
    if torch is None:
        raise RuntimeError("torch is required for inference")
    checkpoint = resolve_checkpoint_reference(config)
    logger.info("Logging to %s", log_path)
    logger.info(
        "Resolved checkpoint %s for run %s",
        checkpoint.path,
        checkpoint.run_id,
    )
    processor = cad_rl.modeling.create_processor_from_spec(config.model)
    model = cad_rl.modeling.load_model_from_spec(config.model, checkpoint.path).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = cad_rl.data.load_inference_dataset(
        {"hf_dataset": config.data.hf_dataset},
        config.data.prepared_datasets[config.infer.split],
        raw_recursive=config.infer.raw_recursive,
        max_samples=config.infer.max_samples,
    )
    generate_kwargs = cad_rl.modeling.build_generation_kwargs(processor)
    generate_kwargs.update(dict(config.model.generation_defaults))
    run_id = str(config.runtime.run_id or config.experiment_id)
    output_path = config.infer.output_path or str(
        Path(config.system.run_root) / run_id / "infer" / "inference_records.jsonl"
    )
    logger.info("Writing inference records to %s", output_path)
    return generate_inference_records(
        model=model,
        processor=processor,
        dataset=dataset,
        output_path=output_path,
        generation_config={
            "batch_size": config.infer.batch_size,
            "num_workers": config.infer.num_workers,
            "generate_kwargs": generate_kwargs,
        },
        task_spec=config.task,
        run_id=checkpoint.run_id,
        checkpoint=checkpoint,
    )


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_jsonl(records: list[dict], key: str) -> dict:
    values = [
        record[key]
        for record in records
        if key in record and isinstance(record[key], (int, float))
    ]
    if not values:
        return {
            "count": len(records),
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    values = sorted(values)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return {
        "count": len(records),
        "mean": sum(values) / len(values),
        "median": median,
        "min": values[0],
        "max": values[-1],
    }
