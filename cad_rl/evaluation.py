from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cad_rl.config import RunConfig, resolve_run_config, to_serializable
from cad_rl.inference import load_jsonl
from cad_rl.metrics import compute_metrics_from_mesh_paths
from cad_rl.runtime import EvalRecord, MeshRecord


def resolve_evaluation_config(
    config_path: str | Path,
    *,
    system: str | Path | None = None,
) -> RunConfig:
    return resolve_run_config(
        config_path, profiles_root=Path(config_path).parents[1], system=system
    )


def export_evaluation_contract(config: RunConfig) -> dict:
    return to_serializable(config)


def evaluate_mesh_records(
    records: list[dict],
    output_path: str | Path,
    *,
    split: str = "val",
    suite: str = "standard",
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    typed_records = [
        item if isinstance(item, MeshRecord) else MeshRecord.from_mapping(item)
        for item in records
    ]
    metrics = [
        compute_metrics_from_mesh_paths(
            record.mesh_path, record.metadata.get("target_mesh_path")
        )
        for record in typed_records
    ]
    eval_rows = []
    ious = []
    cds = []
    invalid = 0
    for record, metric in zip(typed_records, metrics):
        status = (
            "ok"
            if metric and metric.get("iou") is not None and metric.get("cd") is not None
            else "invalid"
        )
        if status == "invalid":
            invalid += 1
        else:
            ious.append(metric["iou"])
            cds.append(metric["cd"])
        eval_rows.append(
            EvalRecord(
                run_id=record.run_id,
                checkpoint=record.checkpoint,
                task_id=str(record.metadata.get("task_id", "unknown")),
                suite=suite,
                split=split,
                metric_name="iou",
                value=float(metric["iou"])
                if metric.get("iou") is not None
                else float("nan"),
                sample_count=1,
                invalid_count=0 if status == "ok" else 1,
                failure_count=0 if record.status == "ok" else 1,
                summary={
                    "cd": metric.get("cd"),
                    "auc": metric.get("auc"),
                    "auc_gms": metric.get("auc_gms"),
                    "mesh_path": record.mesh_path,
                },
                metadata={"status": status},
            )
        )

    summary = {
        "samples": len(typed_records),
        "invalid_fraction": invalid / len(typed_records) if typed_records else 0.0,
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou_median": float(np.median(ious)) if ious else None,
        "iou_min": float(np.min(ious)) if ious else None,
        "iou_max": float(np.max(ious)) if ious else None,
        "cd_mean": float(np.mean(cds)) if cds else None,
        "cd_median": float(np.median(cds)) if cds else None,
        "cd_min": float(np.min(cds)) if cds else None,
        "cd_max": float(np.max(cds)) if cds else None,
        "missing_sample_count": invalid,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(to_serializable(row)) + "\n")
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_from_resolved(config: RunConfig) -> dict:
    if config.eval.input_path is None or config.eval.output_path is None:
        raise ValueError(
            "Evaluation stage requires eval.input_path and eval.output_path"
        )
    records = load_jsonl(config.eval.input_path)
    return evaluate_mesh_records(
        records,
        config.eval.output_path,
        split=config.eval.split,
        suite=config.eval.suite,
    )
