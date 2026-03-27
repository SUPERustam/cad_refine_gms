#!/usr/bin/env python3
# coding: utf-8
"""
Inference for Qwen2-VL on STL renders using vLLM (same I/O as inference_cad_model.py).

- Recursively finds .stl files, renders with Plotter (visualization_iso), builds Qwen chat prompts.
- Uses in-process vLLM instead of Hugging Face ``generate()``; no task profiles or run manifests.

Requires CUDA and the ``vllm`` package.

Usage:
  python examples/inference_vllm.py \\
      --stl_dir /path/to/stls \\
      --out_dir ./preds \\
      --model_path ./work_dirs/.../final_model \\
      --batch_size 4 --max_new_tokens 2048 --do_sample --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from cad_rl.data.visualization_iso import Plotter
from cad_rl.pipelines.inference import _extract_assistant_text
from cad_rl.pipelines.inference_vllm import hf_generate_kwargs_to_vllm_sampling_params

warnings.filterwarnings("ignore")


def _configure_vllm_multiprocessing() -> None:
    # vLLM + CUDA must not run with forked worker processes.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # Start method may already be set by the runtime.
        pass


def save_img_any(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path)
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
    elif torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        t = t.clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(t).save(path)
    else:
        Image.fromarray(np.asarray(img)).save(path)


def list_stls(stl_root: Path) -> List[Path]:
    return sorted(
        [p for p in stl_root.rglob("*.stl") if p.is_file() and p.stat().st_size > 0]
    )


def build_vllm_inputs_for_images(imgs: List[Image.Image], processor) -> list[dict]:
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": img}]}] for img in imgs
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    vis_imgs, _vis_vids = process_vision_info(messages)
    return [
        {"prompt": prompt, "multi_modal_data": {"image": img}}
        for prompt, img in zip(texts, vis_imgs)
    ]


def wrap_dsl(pred_body: str) -> str:
    header = "from api.dsl_api import *\n\n"
    footer = "\n\nresult = n0.build()\n"
    return header + pred_body.rstrip() + footer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STL render inference for Qwen2-VL via vLLM"
    )
    parser.add_argument(
        "--stl_dir",
        type=Path,
        required=True,
        help="Folder with .stl files (recursively searched)",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Where to write predictions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(
            Path(__file__).parent
            / "work_dirs"
            / "qwen2vl_sampling_finetune_cadevolve_debug"
            / "final_model"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--apply_augs",
        action="store_true",
        help="Use Plotter augmentations (default: off for eval)",
    )
    parser.add_argument(
        "--wrap_runnable",
        action="store_true",
        help="Wrap predicted body into runnable DSL .py",
    )
    parser.add_argument(
        "--save_jsonl", action="store_true", help="Also save a preds.jsonl manifest"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction for vLLM KV cache",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel world size for vLLM",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional max_model_len for vLLM LLM()",
    )
    parser.add_argument("--resized_width", type=int, default=14 * 17 * 2)
    parser.add_argument("--resized_height", type=int, default=14 * 17 * 4)

    args = parser.parse_args()
    _configure_vllm_multiprocessing()

    try:
        from vllm import LLM
    except ImportError as e:
        print("Install vllm to use this script.", file=sys.stderr)
        raise SystemExit(1) from e

    if args.num_beams != 1:
        print(
            "Warning: vLLM path ignores --num_beams > 1 (beam search not mapped).",
            file=sys.stderr,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stls = list_stls(args.stl_dir)
    if not stls:
        print(f"No non-empty .stl files found under: {args.stl_dir}")
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        resized_width=args.resized_width,
        resized_height=args.resized_height,
        padding_side="left",
    )
    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_token_id = tokenizer.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    sampling_params = hf_generate_kwargs_to_vllm_sampling_params(tokenizer, gen_kwargs)

    llm_kw: dict = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 1},
    }
    if args.vllm_max_model_len is not None:
        llm_kw["max_model_len"] = args.vllm_max_model_len
    llm = LLM(**llm_kw)

    plotter = Plotter()
    manifest: list[dict] = []
    batch_imgs: list[Image.Image] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal batch_imgs, batch_paths
        if not batch_imgs:
            return

        vllm_inputs = build_vllm_inputs_for_images(batch_imgs, processor)
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

        for i, o in enumerate(outputs):
            raw = o.outputs[0].text
            pred_body = _extract_assistant_text(raw)
            stl_path = batch_paths[i]
            stem = stl_path.stem
            if args.wrap_runnable:
                text_to_save = wrap_dsl(pred_body)
                out_ext = ".dsl.py"
            else:
                text_to_save = pred_body
                out_ext = ".txt"

            rel_dir = (
                stl_path.parent.relative_to(args.stl_dir)
                if stl_path.parent != args.stl_dir
                else Path(".")
            )
            save_dir = args.out_dir / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{stem}{out_ext}"
            out_path.write_text(text_to_save, encoding="utf-8")

            finish_reason = getattr(o.outputs[0], "finish_reason", None)
            manifest.append(
                {
                    "stl": str(stl_path),
                    "pred_path": str(out_path),
                    "wrapped": bool(args.wrap_runnable),
                    "finish_reason": finish_reason,
                }
            )

        batch_imgs, batch_paths = [], []

    print(len(stls))
    for stl in stls:
        try:
            img = plotter.get_img(stl, None, apply_augs=args.apply_augs)
            save_img_any(
                img,
                (
                    args.out_dir
                    / "renders"
                    / stl.parent.relative_to(args.stl_dir)
                    / f"{stl.stem}.png"
                ),
            )
            batch_imgs.append(img)
            batch_paths.append(stl)
        except Exception as e:
            print(f"[WARN] render failed for {stl}: {e}")
            try:
                plotter.reload()
            except Exception:
                pass
            continue

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

    flush_batch()

    if args.save_jsonl:
        jsonl_path = args.out_dir / "preds.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved manifest: {jsonl_path}")

    print(
        f"Done. Processed {len(manifest)} / {len(stls)} STLs. Outputs -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
