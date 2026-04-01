from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import pickle
import random
from pathlib import Path
from typing import Any, Iterable, List, Mapping

try:  # pragma: no cover - optional dependency
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - lightweight test environments
    class Dataset:  # type: ignore[no-redef]
        pass

from cad_rl.data.render import Plotter1_1


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            yield path


def _hash_path_tree(root: Path) -> tuple[str, int, int]:
    hasher = sha256()
    file_count = 0
    total_bytes = 0
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest(), file_count, total_bytes


@dataclass(frozen=True)
class PreparedDatasetFingerprint:
    dataset_path: str
    fingerprint: str
    file_count: int
    byte_size: int
    source_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedDatasetFingerprint":
        return cls(**data)


@dataclass(frozen=True)
class PreparedDatasetManifest:
    task_id: str
    split: str
    source_root: str
    output_path: str
    row_count: int
    model_id: str
    prompt_template_id: str
    render_profile_id: str
    fingerprint: PreparedDatasetFingerprint

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedDatasetManifest":
        payload = dict(data)
        payload["fingerprint"] = PreparedDatasetFingerprint.from_dict(
            payload["fingerprint"]
        )
        return cls(**payload)


def fingerprint_prepared_dataset(dataset_path: str | Path) -> PreparedDatasetFingerprint:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)
    tree_hash, file_count, byte_size = _hash_path_tree(path)
    source_fingerprint = None
    try:
        from datasets import load_from_disk

        loaded = load_from_disk(str(path))
        source_fingerprint = getattr(loaded, "_fingerprint", None)
    except Exception:
        source_fingerprint = None

    return PreparedDatasetFingerprint(
        dataset_path=str(path),
        fingerprint=tree_hash,
        file_count=file_count,
        byte_size=byte_size,
        source_fingerprint=source_fingerprint,
    )


def load_prepared_hf_dataset(dataset_path: str | Path, split: str | None = None) -> Any:
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if split is None:
        return dataset
    try:
        return dataset[split]
    except Exception:
        return dataset


class STLImageToCode(Dataset):
    def __init__(
        self,
        stls_root: Path,
        split_file: Path | None = None,
        pickle_file: Path | None = None,
        shuffle: bool = True,
        size: int | None = None,
        split: str = "train",
    ):
        super().__init__()
        self.stls_root = stls_root
        self.plotter = None
        self.items: List[Path] = []
        count = 0
        active = True
        if split_file:
            folders = Path(split_file).read_text().strip().split(",")
            folders = [f.strip() for f in folders if f.strip()]
            for folder in folders:
                if not active:
                    break
                sdir = stls_root / folder
                if not sdir.exists():
                    continue
                for stl in sorted(sdir.glob("*.stl")):
                    if stl.exists() and stl.stat().st_size > 0:
                        self.items.append(stl.resolve())
                        count += 1
                        if size and count >= size:
                            active = False
                            break
        elif pickle_file:
            with open(pickle_file, "rb") as handle:
                annotations = pickle.load(handle)
            self.items = [
                (stls_root / ann["mesh_path"]).resolve()
                for ann in annotations.values()
                if (stls_root / ann["mesh_path"]).exists()
                and (stls_root / ann["mesh_path"]).stat().st_size > 0
            ]
        else:
            for index, stl in enumerate(stls_root.rglob("*.stl"), 1):
                if stl.exists() and stl.stat().st_size > 0:
                    self.items.append(stl.resolve())
                    if size and index >= size:
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
        except Exception as exc:
            self.plotter.reload()
            raise RuntimeError(f"Failed to render {stl_path}") from exc
        return {"image": image, "mesh_path": stl_path}


def _should_use_hf_dataset(task_profile: Mapping[str, Any]) -> bool:
    if "hf_dataset" in task_profile:
        return bool(task_profile["hf_dataset"])
    return True


class RawSTLInferenceDataset(Dataset):
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
        return load_prepared_hf_dataset(path)
    return RawSTLInferenceDataset(path, recursive=raw_recursive, size=max_samples)
