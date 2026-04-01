from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cad_rl.config import RunConfig, resolve_run_config, to_serializable
from cad_rl.data.datasets import (
    PreparedDatasetManifest,
    STLImageToCode,
    fingerprint_prepared_dataset,
)
from cad_rl.modeling import create_processor_from_spec

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


def _build_rows(dataset: STLImageToCode, processor: Any) -> list[dict[str, Any]]:
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


def resolve_prepare_config(
    config_path: str | Path,
    *,
    system: str | Path | None = None,
) -> RunConfig:
    return resolve_run_config(
        config_path, profiles_root=Path(config_path).parents[1], system=system
    )


def export_prepare_contract(config: RunConfig) -> dict:
    return to_serializable(config)


def prepare_dataset_from_config(config_path: str | Path, *, system: str | Path | None = None):
    from datasets import Dataset, Features, Image as HFImage, Value

    config = resolve_prepare_config(config_path, system=system)
    processor = create_processor_from_spec(config.model)
    split = config.prepare.split
    raw_root = config.prepare.raw_root or config.data.raw_dataset_root
    pickle_file = config.prepare.pickle_file or config.data.raw_dataset_pickle
    output_dir = Path(
        config.prepare.output_path or config.data.prepared_datasets.get(split, "")
    )
    if not raw_root:
        raise ValueError("Dataset preparation requires runtime.raw_root or task.raw_dataset_root")
    if not output_dir:
        raise ValueError("Dataset preparation requires runtime.output_path or task.prepared_datasets[split]")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source = STLImageToCode(
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

    fingerprint = fingerprint_prepared_dataset(output_dir)
    manifest = PreparedDatasetManifest(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a dataset from a stage config"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    config = resolve_prepare_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_prepare_contract(config), indent=2))
        return
    prepare_dataset_from_config(args.config, system=args.system)
