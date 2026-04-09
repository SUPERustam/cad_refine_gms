# this script creates a serialized huggingface dataset from an STLImageToCode by
# rendering each mesh and normalizing the image based on qwen_vl process_vision_info

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from multiview_dataset import STLImageToCode
from datasets import Dataset, Features, Value, Image as HFImage
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from torch.utils.data import ConcatDataset, DataLoader
import torch
from qwen_vl_utils.vision_process import fetch_image

SEED = 99
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# path to meshes
MAXPYROOT = Path(
    "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/monkey_patching/work_with_augmented/train_1_py"
)
MAXSTLS_ROOT = Path(
    "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/elistratovm/monkey_patching/work_with_augmented/train_1_stls"
)

DP_ROOT = Path(
    "/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/deepcad_fusion_train"
)
MCBROOT = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/MCB_A/")

# we need to load the model to get the config VISION_PATCH padding value
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    # min_pixels=256*28*28,
    # max_pixels=1280*28*28,
    resized_width=14 * 17 * 2,
    resized_height=14 * 17 * 4,
    padding_side="left",
)


data_dp = STLImageToCode(DP_ROOT, pickle_file=DP_ROOT / "pkls/train_small.pkl")
print(f"dp has length {len(data_dp)}")
# data_mcb = STLImageToCode(MCBROOT)
# print(f"mcb has length {len(data_mcb)}")
# data = ConcatDataset([data_dp, data_mcb])
data = data_dp
#### render image and add conversational format prompt --------------------------------------------------


def make_data(_, idx):
    ex = data[idx]
    if ex is None:
        return {"image": None, "mesh_path": "", "prompt": ""}
    conv = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    ex["image"] = fetch_image({"image": ex["image"]})
    ex["mesh_path"] = str(ex["mesh_path"])
    return {**ex, "prompt": prompt}


### ---------------------------------------------------

new_features = Features(
    {
        "mesh_path": Value("string"),
        "image": HFImage(),
        "prompt": Value("string"),
    }
)

base = Dataset.from_dict({"i": list(range(len(data)))})
ds = base.map(
    make_data,
    with_indices=True,
    num_proc=156,
    features=new_features,
    writer_batch_size=64,
    remove_columns=["i"],
).filter(lambda ex: ex["image"] is not None)

ds_processed_path = "./datasets/rendered_cadevolve_normalized_1_1_deepcadf360"
ds.save_to_disk(ds_processed_path)
