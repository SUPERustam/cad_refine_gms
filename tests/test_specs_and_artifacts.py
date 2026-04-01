from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_rl.config import to_serializable
from cad_rl.data import (
    PreparedDatasetFingerprint,
    PreparedDatasetManifest,
    fingerprint_prepared_dataset,
    load_prepared_hf_dataset,
)
from cad_rl.comparison import compare_summaries
from cad_rl.inference import load_jsonl
from cad_rl.runtime import CheckpointRef, InferenceRecord, MeshRecord


def test_prepared_dataset_fingerprint_and_loader(tmp_path: Path):
    datasets = pytest.importorskip("datasets")

    root = tmp_path / "dataset"
    dataset = datasets.Dataset.from_dict({"value": [1, 2, 3]})
    dataset.save_to_disk(str(root))

    fp = fingerprint_prepared_dataset(root)
    loaded = load_prepared_hf_dataset(root)

    assert fp.file_count > 0
    assert fp.byte_size > 0
    assert fp.dataset_path == str(root)
    assert fp.fingerprint
    assert len(loaded) == 3

    restored_fp = PreparedDatasetFingerprint.from_dict(fp.to_dict())
    manifest = PreparedDatasetManifest(
        task_id="task",
        split="train",
        source_root=str(root),
        output_path=str(root),
        row_count=3,
        model_id="model",
        prompt_template_id="template",
        render_profile_id="render",
        fingerprint=fp,
    )
    restored_manifest = PreparedDatasetManifest.from_dict(manifest.to_dict())

    assert restored_fp.dataset_path == str(root)
    assert restored_manifest.fingerprint.fingerprint == fp.fingerprint


def test_jsonl_records_and_comparison_report(tmp_path: Path) -> None:
    checkpoint = CheckpointRef(run_id="run-1", step=10, path="/tmp/checkpoint")
    records_path = tmp_path / "inference.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(
                    to_serializable(
                        InferenceRecord(
                            run_id="run-1",
                            checkpoint=checkpoint,
                            sample_id="sample-1",
                            raw_generation="result = cq.Workplane()",
                            metadata={"target_mesh_path": "/tmp/target.stl"},
                        )
                    )
                ),
                json.dumps(
                    to_serializable(
                        MeshRecord(
                            run_id="run-1",
                            checkpoint=checkpoint,
                            sample_id="sample-1",
                            mesh_path="/tmp/pred.stl",
                            metadata={"target_mesh_path": "/tmp/target.stl"},
                        )
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )
    summary_a = tmp_path / "a.summary.json"
    summary_b = tmp_path / "b.summary.json"
    summary_a.write_text(json.dumps({"run_id": "run-a", "iou_mean": 0.8, "cd_mean": 0.1}), encoding="utf-8")
    summary_b.write_text(json.dumps({"run_id": "run-b", "iou_mean": 0.9, "cd_mean": 0.2}), encoding="utf-8")

    rows = load_jsonl(records_path)
    report = compare_summaries([summary_a, summary_b], tmp_path / "report.json")

    assert len(rows) == 2
    assert rows[0]["sample_id"] == "sample-1"
    assert report.run_ids == ("run-a", "run-b")
    assert report.leaderboard[0]["run_id"] == "run-b"
