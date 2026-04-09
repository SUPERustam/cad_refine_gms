#!/usr/bin/env python3
# coding: utf-8
"""
vLLM-based inference for Qwen2-VL fine-tuned on CadQuery STL renders.

This script mirrors benchmark/inference.py behavior:
- scans STL files recursively
- renders each STL with visualization_iso.Plotter
- builds a Qwen chat prompt with one image
- generates code with vLLM
- writes one output file per STL (+ optional JSONL manifest)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from visualization_iso import Plotter

warnings.filterwarnings("ignore")


def save_img_any(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path)
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    elif torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(np.asarray(img)).save(path)


def to_pil_image(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def list_stls(stl_root: Path) -> List[Path]:
    return sorted(
        [p for p in stl_root.rglob("*.stl") if p.is_file() and p.stat().st_size > 0]
    )


def extract_assistant_text(decoded: str) -> str:
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    if start_tag in decoded:
        part = decoded.split(start_tag, maxsplit=1)[1]
        if part.startswith("\n"):
            part = part[1:]
        if end_tag in part:
            part = part.split(end_tag, maxsplit=1)[0]
        return part.strip()
    return decoded.strip()


def wrap_dsl(pred_body: str) -> str:
    header = "from api.dsl_api import *\n\n"
    footer = "\n\nresult = n0.build()\n"
    return header + pred_body.rstrip() + footer


def build_prompt(processor: AutoProcessor, image: Image.Image) -> str:
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
            ],
        }
    ]
    return processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference for Qwen2-VL on STL renders using vLLM"
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
    parser.add_argument("--resized_width", type=int, default=14 * 17 * 2)
    parser.add_argument("--resized_height", type=int, default=14 * 17 * 4)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stls = list_stls(args.stl_dir)
    if not stls:
        print(f"No non-empty .stl files found under: {args.stl_dir}")
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        resized_width=args.resized_width,
        resized_height=args.resized_height,
        padding_side="left",
    )

    llm = LLM(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )

    # Qwen chat format uses this token for assistant turn end.
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id is None:
        eos_token_id = processor.tokenizer.eos_token_id

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
        top_k=args.top_k if args.do_sample else -1,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
    )

    plotter = Plotter()
    manifest = []
    batch_reqs = []
    batch_paths = []

    def flush_batch() -> None:
        nonlocal batch_reqs, batch_paths, manifest
        if not batch_reqs:
            return

        outputs = llm.generate(batch_reqs, sampling_params=sampling)

        for request_out, stl_path in zip(outputs, batch_paths):
            if not request_out.outputs:
                pred_body = ""
                completion_tokens = 0
            else:
                best = request_out.outputs[0]
                pred_body = extract_assistant_text(best.text)
                completion_tokens = (
                    len(best.token_ids) if best.token_ids is not None else 0
                )

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

            manifest.append(
                {
                    "stl": str(stl_path),
                    "pred_path": str(out_path),
                    "wrapped": bool(args.wrap_runnable),
                    "completion_tokens": int(completion_tokens),
                }
            )

        batch_reqs = []
        batch_paths = []

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

            pil_img = to_pil_image(img)
            prompt = build_prompt(processor, pil_img)
            req = {"prompt": prompt, "multi_modal_data": {"image": pil_img}}

            batch_reqs.append(req)
            batch_paths.append(stl)
        except Exception as e:
            print(f"[WARN] render/prompt failed for {stl}: {e}")
            try:
                plotter.reload()
            except Exception:
                pass
            continue

        if len(batch_reqs) >= args.batch_size:
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
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
