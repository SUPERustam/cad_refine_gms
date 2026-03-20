from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


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


def fingerprint_prepared_dataset(
    dataset_path: str | Path,
) -> PreparedDatasetFingerprint:
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
    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "datasets is required to load prepared Hugging Face datasets"
        ) from exc

    dataset = load_from_disk(str(dataset_path))
    if split is None:
        return dataset
    try:
        return dataset[split]
    except Exception:
        return dataset
