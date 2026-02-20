
import os, sys, random, json, warnings, time
from pathlib import Path
from functools import partial
from typing import List, Tuple, Dict, Any
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.transform import resize
import trimesh
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration,
    Trainer, TrainingArguments, TrainerCallback,
)

# sys.path.append("/home/jovyan/users/zhemchuzhnikov/miniconda3/envs/zhemchuzhnikov/lib/python3.10/site-packages") # ME: comment this
import open3d as o3d

IMAGE_SIZE  = 130     # base thumb size (isometric is 2x)
PAD = 16
ROW1_W = 2*IMAGE_SIZE + PAD + 2 * IMAGE_SIZE  + 2 * PAD
ROW2_W = 4 * IMAGE_SIZE + 3 * PAD
TOTAL_W = max(ROW1_W, ROW2_W)   # = 568 for IMAGE_SIZE=130, PAD=16
TOTAL_H = 2*IMAGE_SIZE + PAD + IMAGE_SIZE  # = 406

TEXT_LABEL_HEIGHT = 18


def transform_gt_mesh_cad_0875(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(0.875 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

# sys.path.append('/workspace-SR008.fs2/users/gennadii/cadeau') # ME: comment this
from vis_for_norm_parts import Plotter
from helper_visu import Plotter1_1

class STLImageToCode(Dataset):
    """
    Scans split file (comma-separated class folders). For each class folder F:
      stl:  STLS_ROOT/F/<stem>.stl
    """
    def __init__(self, stls_root: Path, split_file: Path = None, pickle_file=None, shuffle=True, size=None, split = "train"):
        super().__init__()
        self.stls_root = stls_root
        self.plotter = None

        self.items: List[dict] = []
        i=0
        go = True
        if split_file:
            folders = Path(split_file).read_text().strip().split(",")
            folders = [f.strip() for f in folders if f.strip()]
            for folder in folders:
                if go==False:
                    break
                sdir = stls_root / folder
                if not sdir.exists():
                    continue
                for stl in sorted(sdir.glob("*.stl")):
                    if stl.exists() and stl.stat().st_size > 0:
                        self.items.append(stl.resolve())
                        i+=1
                        if size and i >= size :
                            go = False
                            break
        elif pickle_file:
            with open(pickle_file, "rb") as f:
                annotations = pickle.load(f)

            self.items = [
                (stls_root / ann["mesh_path"]).resolve()
                for ann in annotations.values()
                if (stls_root / ann["mesh_path"]).exists()
                and (stls_root / ann["mesh_path"]).stat().st_size > 0]
        else:
            print(f"getting directly from root")
            from tqdm import tqdm
            gt_dir = Path("/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/MCB_A_batch/groudtruth")

            # take directly stls from stls_root
            for i, stl in enumerate(tqdm(stls_root.rglob("*.stl"), total=size or None), 1):
                if (gt_dir / stl.name).exists() and split != "val":
                    continue
                
                if stl.exists() and stl.stat().st_size > 0:
                    self.items.append(stl.resolve())
                    if size and i >= size:
                        break

        if not self.items:
            raise RuntimeError(f"No stl files found")

        if shuffle:
            random.shuffle(self.items)
        if size :
            self.items = self.items[:size]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        stl_path = self.items[idx]
        if self.plotter is None:
            self.plotter = Plotter1_1()
            #self.plotter = Plotter()

        try:
            image = self.plotter.get_img(stl_path, None)
        except Exception as e:
            print("Error in visualization:", e)
            self.plotter.reload()
            return None

        return {
            "image": image,
            "mesh_path": stl_path}