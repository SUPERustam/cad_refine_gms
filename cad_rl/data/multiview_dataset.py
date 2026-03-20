from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from .helper_visu import Plotter1_1


def transform_gt_mesh_cad_0875(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    import numpy as np
    import trimesh

    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(0.875 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


class STLImageToCode(Dataset):
    """Scan STL sources and render them into image/prompt pairs."""

    def __init__(
        self,
        stls_root: Path,
        split_file: Path = None,
        pickle_file=None,
        shuffle=True,
        size=None,
        split="train",
    ):
        super().__init__()
        self.stls_root = stls_root
        self.plotter = None

        self.items: List[dict] = []
        i = 0
        go = True
        if split_file:
            folders = Path(split_file).read_text().strip().split(",")
            folders = [f.strip() for f in folders if f.strip()]
            for folder in folders:
                if go is False:
                    break
                sdir = stls_root / folder
                if not sdir.exists():
                    continue
                for stl in sorted(sdir.glob("*.stl")):
                    if stl.exists() and stl.stat().st_size > 0:
                        self.items.append(stl.resolve())
                        i += 1
                        if size and i >= size:
                            go = False
                            break
        elif pickle_file:
            with open(pickle_file, "rb") as f:
                annotations = pickle.load(f)

            self.items = [
                (stls_root / ann["mesh_path"]).resolve()
                for ann in annotations.values()
                if (stls_root / ann["mesh_path"]).exists()
                and (stls_root / ann["mesh_path"]).stat().st_size > 0
            ]
        else:
            print("getting directly from root")
            from tqdm import tqdm

            gt_dir = Path(
                "/workspace-SR008.nfs2/users/zhemchuzhnikov/datasets/MCB_A_batch/groudtruth"
            )

            for i, stl in enumerate(
                tqdm(stls_root.rglob("*.stl"), total=size or None), 1
            ):
                if (gt_dir / stl.name).exists() and split != "val":
                    continue

                if stl.exists() and stl.stat().st_size > 0:
                    self.items.append(stl.resolve())
                    if size and i >= size:
                        break

        if not self.items:
            raise RuntimeError("No stl files found")

        if shuffle:
            random.shuffle(self.items)
        if size:
            self.items = self.items[:size]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stl_path = self.items[idx]
        if self.plotter is None:
            self.plotter = Plotter1_1()

        try:
            image = self.plotter.get_img(stl_path, None)
        except Exception as e:
            print("Error in visualization:", e)
            self.plotter.reload()
            return None

        return {"image": image, "mesh_path": stl_path}
