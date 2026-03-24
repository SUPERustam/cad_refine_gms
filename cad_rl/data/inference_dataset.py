from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from datasets import load_from_disk
from torch.utils.data import Dataset

from .helper_visu import Plotter1_1


def _should_use_hf_dataset(task_profile: Mapping[str, Any]) -> bool:
    if "hf_dataset" in task_profile:
        return bool(task_profile["hf_dataset"])
    return True


class RawSTLInferenceDataset(Dataset):
    """Load raw STL files directly and render them for inference."""

    def __init__(self, root: str | Path, *, recursive: bool = False, size: int | None = None) -> None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Raw dataset root not found: {root_path}")

        iterator = root_path.rglob("*.stl") if recursive else root_path.glob("*.stl")
        self.paths = [
            path.resolve()
            for path in iterator
            if path.is_file() and path.stat().st_size > 0
        ]
        if not self.paths:
            raise RuntimeError(f"No non-empty STL files found in {root_path}")

        self.paths = sorted(self.paths)
        if size is not None:
            self.paths = self.paths[:size]

        self.plotter = Plotter1_1()
        self.plotter.reload()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        mesh_path = self.paths[index]
        try:
            image = self.plotter.get_img(mesh_path, None)
        except Exception as exc:
            self.plotter.reload()
            raise RuntimeError(f"Failed to render {mesh_path}") from exc
        return {"image": image, "mesh_path": mesh_path}


def load_inference_dataset(
    task_profile: Mapping[str, Any],
    path: str,
    *,
    raw_recursive: bool = False,
    max_samples: int | None = None,
):
    if _should_use_hf_dataset(task_profile):
        return load_from_disk(path)
    return RawSTLInferenceDataset(path, recursive=raw_recursive, size=max_samples)
