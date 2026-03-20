from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cad_rl.data import fingerprint_prepared_dataset
from cad_rl.models import QwenAdapter
from cad_rl.specs.task import default_task_spec

from .multiview_dataset import STLImageToCode

try:  # pragma: no cover - optional dependency fallback
    from qwen_vl_utils.vision_process import fetch_image
except Exception:  # pragma: no cover - optional dependency fallback
    fetch_image = None


def _normalize_image(image: Any) -> Any:
    if fetch_image is None:
        return image
    return fetch_image({"image": image})


def _build_rows(
    dataset: STLImageToCode, processor: Any, task_spec
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        ex = dataset[idx]
        if ex is None:
            continue
        rows.append(
            {
                "image": _normalize_image(ex["image"]),
                "mesh_path": str(ex["mesh_path"]),
                "prompt": task_spec.render_prompt(processor, ex["image"]),
            }
        )
    return rows


def main() -> None:
    from datasets import Dataset, Features, Image as HFImage, Value

    parser = argparse.ArgumentParser(
        description="Prepare a Qwen-style HF dataset from STL renders"
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Source STL root")
    parser.add_argument(
        "--pickle-file", type=Path, default=None, help="Optional annotation pickle"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to read"
    )
    parser.add_argument("--size", type=int, default=None, help="Optional sample cap")
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle source examples"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output HF dataset directory"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Task spec id to record in the manifest",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model spec id to record in the manifest",
    )
    args = parser.parse_args()

    task_spec = default_task_spec()
    model_spec = task_spec.default_model
    adapter = QwenAdapter(model_spec)
    processor = adapter.load_processor()

    raw_root = args.raw_root or (
        Path(task_spec.raw_dataset_root) if task_spec.raw_dataset_root else None
    )
    if raw_root is None:
        raise ValueError("A raw dataset root is required")

    pickle_file = args.pickle_file or (
        Path(task_spec.raw_dataset_pickle) if task_spec.raw_dataset_pickle else None
    )
    output_dir = args.output_dir or Path(task_spec.prepared_dataset_path(args.split))
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source = STLImageToCode(
        raw_root,
        pickle_file=pickle_file,
        shuffle=bool(args.shuffle),
        size=args.size,
        split=args.split,
    )

    rows = _build_rows(source, processor, task_spec)
    features = Features(
        {
            "mesh_path": Value("string"),
            "image": HFImage(),
            "prompt": Value("string"),
        }
    )
    columns = {
        "mesh_path": [row["mesh_path"] for row in rows],
        "image": [row["image"] for row in rows],
        "prompt": [row["prompt"] for row in rows],
    }
    dataset = Dataset.from_dict(columns).cast(features)
    dataset.save_to_disk(str(output_dir))

    fingerprint = fingerprint_prepared_dataset(output_dir)
    manifest = {
        "task_id": args.task_id or task_spec.task_id,
        "model_id": args.model_id or model_spec.model_id,
        "split": args.split,
        "raw_root": str(raw_root),
        "pickle_file": str(pickle_file) if pickle_file else None,
        "output_dir": str(output_dir),
        "row_count": len(dataset),
        "fingerprint": fingerprint.to_dict(),
        "prompt_template_id": task_spec.prompt_render.template_id,
        "render_profile_id": task_spec.prompt_render.template_id,
        "output_var_name": task_spec.output_var_name,
        "normalization_mode": task_spec.normalization_mode,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
