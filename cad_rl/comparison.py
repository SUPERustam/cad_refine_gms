from __future__ import annotations

import json
import logging
from pathlib import Path

import cad_rl.config
import cad_rl.runtime


def _load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_summaries(
    summary_paths: list[str | Path], output_path: str | Path
) -> cad_rl.runtime.ComparisonReport:
    summaries = []
    checkpoint_refs = []
    for path in summary_paths:
        data = _load_summary(path)
        data["source"] = str(path)
        summaries.append(data)
        checkpoint_path = data.get("checkpoint_path")
        if checkpoint_path:
            checkpoint_refs.append(
                cad_rl.runtime.CheckpointRef(
                    run_id=str(data.get("run_id", path)),
                    step=None
                    if data.get("checkpoint_step") is None
                    else int(data["checkpoint_step"]),
                    path=str(checkpoint_path),
                    kind="specific",
                )
            )

    leaderboard = sorted(
        summaries,
        key=lambda item: (
            item.get("iou_mean") is None,
            -(item.get("iou_mean") or -1.0),
            item.get("cd_mean") or float("inf"),
        ),
    )
    report = cad_rl.runtime.ComparisonReport(
        report_id=Path(output_path).stem,
        run_ids=tuple(str(item.get("run_id", item["source"])) for item in summaries),
        checkpoint_refs=tuple(checkpoint_refs),
        leaderboard=tuple(leaderboard),
        checkpoint_history=tuple(
            {
                "source": item["source"],
                "iou_mean": item.get("iou_mean"),
                "cd_mean": item.get("cd_mean"),
                "invalid_fraction": item.get("invalid_fraction"),
            }
            for item in summaries
        ),
        summary={
            "diff_report": [
                {
                    "source": item["source"],
                    "missing_sample_count": item.get("missing_sample_count"),
                    "iou_mean": item.get("iou_mean"),
                    "cd_mean": item.get("cd_mean"),
                }
                for item in leaderboard
            ]
        },
    )
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(cad_rl.config.to_serializable(report), indent=2), encoding="utf-8"
    )
    return report


def compare_from_resolved(
    config: cad_rl.config.RunConfig,
) -> cad_rl.runtime.ComparisonReport:
    log_path = cad_rl.runtime.setup_run_logging(config, stage="compare-runs")
    logger = logging.getLogger(__name__)
    if config.compare.output_path is None:
        raise ValueError("Comparison stage requires compare.output_path")
    logger.info("Logging to %s", log_path)
    logger.info("Comparing summaries: %s", ", ".join(config.compare.summaries))
    report = compare_summaries(
        list(config.compare.summaries),
        config.compare.output_path,
    )
    logger.info("Wrote comparison report to %s", config.compare.output_path)
    return report
