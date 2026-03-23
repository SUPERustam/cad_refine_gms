#!/usr/bin/env python3
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from metrics_async import init_pool, close_pool, get_metrics_from_texts
from metrics_stl_pool import run_texts
from pathlib import Path
from multiview_dataset import STLImageToCode
from qwen_vl_utils import process_vision_info

DATAROOT = Path(
    "/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/MCB_A_batch/groudtruth"
)
# DATAROOT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/vsevolod/datasets/fusion360_test_mesh")
# DATAROOT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/vsevolod/datasets/deepcad_test_mesh")

PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
BATCH_SIZE = 128
NUM_WORKERS = 4
POOL_SIZE = 64
MAX_NEW_TOKENS = 4200
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50
SEED = 16

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def collate_fn(batch, processor):
    mesh_paths = [str(b["mesh_path"]) for b in batch]
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": b["image"]}]}]
        for b in batch
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    vis_imgs, vis_vids = process_vision_info(messages)
    inputs = processor(
        text=texts, images=vis_imgs, videos=vis_vids, padding=True, return_tensors="pt"
    )
    inputs["mesh_path"] = mesh_paths
    return inputs


def evaluate(
    model, processor, ds, normalize="fixed", var_name="result", nc_params=None
):
    model.eval()
    device = next(model.parameters()).device

    print("\n" + "=" * 50)
    print("EVALUATION ON", len(ds), "EXAMPLES")
    print("=" * 50)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    ious, cds, mae_sims = [], [], []
    n_incorrect, n_failed_intersect = 0, 0
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_token_id = processor.tokenizer.eos_token_id
    with torch.inference_mode():
        for batch in tqdm(dl):
            # to device
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k != "mesh_path"
            }

            gen_kwargs = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            gen_ids = model.generate(**inputs, **gen_kwargs)

            # trim prompts
            in_lens = batch["attention_mask"].sum(dim=1).tolist()
            gen_list = gen_ids.tolist()
            trimmed = [gen_list[i][in_lens[i] :] for i in range(len(in_lens))]

            # decode
            texts = processor.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            # print(texts)

            # texts = extract_assistant_text(decoded)
            t0 = time.perf_counter()
            # metrics_batch = run_texts(texts, batch["mesh_path"], var_name="result", normalize=normalize)
            metrics_batch = get_metrics_from_texts(
                texts, batch["mesh_path"], nc_params=nc_params, var_name=var_name
            )
            print(
                f"metrics time: {time.perf_counter() - t0:.3f}s for {len(texts)} samples"
            )
            # print(metrics_batch)

            for m in metrics_batch:
                if m is None or m.get("iou") is None or m.get("cd") is None:
                    n_incorrect += 1
                    continue
                if m["iou"] < 0:
                    n_failed_intersect += 1
                    continue
                ious.append(m["iou"])
                cds.append(m["cd"])
                if (
                    nc_params
                    and nc_params.get("get_mae_render")
                    and m.get("mae_similarity") is not None
                ):
                    mae_sims.append(m["mae_similarity"])

    mn = lambda x: float(np.mean(x)) if len(x) else float("nan")
    md = lambda x: float(np.median(x)) if len(x) else float("nan")

    print(f"IoU mean {mn(ious)}, median {md(ious)}")
    print(f"CD mean {mn(cds)}, median {md(cds)}")
    if mae_sims:
        print(f"MAE similarity mean {mn(mae_sims)}, median {md(mae_sims)}")
    print(f"Invalid generations fraction: {n_incorrect / len(ds):.6f}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(ds):.6f}")
    print("=" * 50)

    out = {
        "ious": ious,
        "cds": cds,
        "invalid_frac": n_incorrect / len(ds),
        "intersect_fail_frac": n_failed_intersect / len(ds),
    }
    if mae_sims:
        out["mae_similarity"] = mae_sims
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help="Path or hub id for Qwen2-VL model"
    )
    parser.add_argument(
        "--normalize", type=str, default="fixed", choices=["fixed", "mesh_extents"]
    )
    parser.add_argument(
        "--var_name",
        type=str,
        default=None,
        help="Variable name holding final CAD object.",
    )
    parser.add_argument(
        "--get_mae_render",
        action="store_true",
        help="Compute MAE render similarity metric (slower).",
    )
    args = parser.parse_args()

    var_name = args.var_name or os.getenv("METRICS_VAR_NAME", "result")

    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"

    # -----------------------------------------------------------------------------
    # Environment tweaks – important for head-less servers and reproducibility
    # -----------------------------------------------------------------------------
    os.environ["PYGLET_HEADLESS"] = "True"  # Avoid opening windows when trimesh renders

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        resized_width=14 * 17 * 2,
        resized_height=14 * 17 * 4,
        padding_side="left",
    )

    attn_impl = "flash_attention_2" if torch.cuda.is_available() else None
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    init_pool(POOL_SIZE)
    eval_ds = STLImageToCode(
        DATAROOT,
        split="val",
        size=256,
        # pickle_file = "/workspace-SR008.nfs2/users/barannikov/cad_refine_rl/datasets/MCB_A_batch_groundtruth.pkl",
        shuffle=False,
    )

    nc_params = {"get_mae_render": args.get_mae_render} if args.get_mae_render else None
    res = evaluate(
        model,
        processor,
        eval_ds,
        normalize=args.normalize,
        var_name=var_name,
        nc_params=nc_params,
    )
    close_pool()

    mn = lambda x: float(np.mean(x)) if len(x) else float("nan")
    md = lambda x: float(np.median(x)) if len(x) else float("nan")

    metrics = {
        "eval/img/IoU mean": mn(res["ious"]),
        "eval/img/CD mean": mn(res["cds"]),
        "eval/img/IoU median": md(res["ious"]),
        "eval/img/CD median": md(res["cds"]),
        "eval/img/Invalid frac": res["invalid_frac"],
        "eval/img/Intersect fail frac": res["intersect_fail_frac"],
    }
    if "mae_similarity" in res:
        metrics["eval/img/MAE similarity mean"] = mn(res["mae_similarity"])
        metrics["eval/img/MAE similarity median"] = md(res["mae_similarity"])

    print("\n==== FINAL METRICS ====")
    print(json.dumps(metrics, indent=2))
    print("=======================\n")


if __name__ == "__main__":
    main()
