from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cad_rl.metrics.async_metrics import get_metrics_from_texts


def evaluate_inference_records(
    records: list[dict], output_path: str | Path, var_name: str = "result"
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    texts = [record["raw_generation"] for record in records]
    mesh_paths = [record["mesh_path"] for record in records]
    metrics = get_metrics_from_texts(texts, mesh_paths, var_name=var_name)

    eval_rows = []
    ious = []
    cds = []
    invalid = 0
    for record, metric in zip(records, metrics):
        row = {
            "mesh_path": record["mesh_path"],
            "task_id": record.get("task_id"),
            "iou": metric.get("iou") if metric else None,
            "cd": metric.get("cd") if metric else None,
            "auc": metric.get("auc") if metric else None,
            "auc_gms": metric.get("auc_gms") if metric else None,
            "status": "ok"
            if metric and metric.get("iou") is not None and metric.get("cd") is not None
            else "invalid",
        }
        if row["status"] == "invalid":
            invalid += 1
        else:
            ious.append(row["iou"])
            cds.append(row["cd"])
        eval_rows.append(row)

    summary = {
        "samples": len(records),
        "invalid_fraction": invalid / len(records) if records else 0.0,
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
            handle.write(json.dumps(row) + "\n")
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
