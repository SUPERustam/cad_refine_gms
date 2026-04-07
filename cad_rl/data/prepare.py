from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cad_rl.config
import cad_rl.data.datasets
import cad_rl.modeling

try:  # pragma: no cover - optional dataset dependency
    from datasets import Dataset, Features, Image as HFImage, Value
except Exception:  # pragma: no cover - lightweight environments
    Dataset = None
    Features = None
    HFImage = None
    Value = None

try:  # pragma: no cover - optional dependency fallback
    from qwen_vl_utils.vision_process import fetch_image
except Exception:  # pragma: no cover - optional dependency fallback
    fetch_image = None


def _normalize_image(image: Any) -> Any:
    if fetch_image is None:
        return image
    return fetch_image({"image": image})


def _render_prompt(processor: Any, image: Any) -> str:
    message = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    return processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )


def _build_rows(
    dataset: cad_rl.data.datasets.STLImageToCode, processor: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        if example is None:
            continue
        rows.append(
            {
                "image": _normalize_image(example["image"]),
                "mesh_path": str(example["mesh_path"]),
                "prompt": _render_prompt(processor, example["image"]),
            }
        )
    return rows


def prepare_dataset_from_resolved(config: cad_rl.config.RunConfig):
    if None in {Dataset, Features, HFImage, Value}:
        raise RuntimeError("datasets is required for dataset preparation")
    processor = cad_rl.modeling.create_processor_from_spec(config.model)
    split = config.prepare.split
    raw_root = config.prepare.raw_root or config.data.raw_dataset_root
    pickle_file = config.prepare.pickle_file or config.data.raw_dataset_pickle
    output_dir = Path(
        config.prepare.output_path or config.data.prepared_datasets.get(split, "")
    )
    if not raw_root:
        raise ValueError(
            "Dataset preparation requires runtime.raw_root or task.raw_dataset_root"
        )
    if not output_dir:
        raise ValueError(
            "Dataset preparation requires runtime.output_path or task.prepared_datasets[split]"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source = cad_rl.data.datasets.STLImageToCode(
        Path(raw_root),
        pickle_file=Path(pickle_file) if pickle_file else None,
        shuffle=config.prepare.shuffle,
        size=config.prepare.size,
        split=split,
    )
    rows = _build_rows(source, processor)
    features = Features(
        {
            "mesh_path": Value("string"),
            "image": HFImage(),
            "prompt": Value("string"),
        }
    )
    dataset = Dataset.from_dict(
        {
            "mesh_path": [row["mesh_path"] for row in rows],
            "image": [row["image"] for row in rows],
            "prompt": [row["prompt"] for row in rows],
        }
    ).cast(features)
    dataset.save_to_disk(str(output_dir))

    fingerprint = cad_rl.data.datasets.fingerprint_prepared_dataset(output_dir)
    manifest = cad_rl.data.datasets.PreparedDatasetManifest(
        task_id=config.task.task_id,
        split=split,
        source_root=str(raw_root),
        output_path=str(output_dir),
        row_count=len(dataset),
        model_id=config.model.base_checkpoint,
        prompt_template_id=config.data.prompt_profile_id,
        render_profile_id=config.data.render_profile_id,
        fingerprint=fingerprint,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"config": config, "manifest": manifest, "output_dir": output_dir}
