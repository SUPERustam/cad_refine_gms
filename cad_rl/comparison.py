from __future__ import annotations

import json
from pathlib import Path

from cad_rl.config import RunConfig, resolve_run_config, to_serializable
from cad_rl.runtime import ComparisonReport


def _load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_comparison_config(
    config_path: str | Path,
    *,
    system: str | Path | None = None,
) -> RunConfig:
    return resolve_run_config(
        config_path, profiles_root=Path(config_path).parents[1], system=system
    )


def export_comparison_contract(config: RunConfig) -> dict:
    return to_serializable(config)


def compare_summaries(summary_paths: list[str | Path], output_path: str | Path) -> ComparisonReport:
    summaries = []
    for path in summary_paths:
        data = _load_summary(path)
        data["source"] = str(path)
        summaries.append(data)

    leaderboard = sorted(
        summaries,
        key=lambda item: (
            item.get("iou_mean") is None,
            -(item.get("iou_mean") or -1.0),
            item.get("cd_mean") or float("inf"),
        ),
    )
    report = ComparisonReport(
        report_id=Path(output_path).stem,
        run_ids=tuple(str(item.get("run_id", item["source"])) for item in summaries),
        checkpoint_refs=(),
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
        json.dumps(to_serializable(report), indent=2), encoding="utf-8"
    )
    return report


def compare_from_resolved(config: RunConfig) -> ComparisonReport:
    if config.compare.output_path is None:
        raise ValueError("Comparison stage requires compare.output_path")
    return compare_summaries(
        list(config.compare.summaries),
        config.compare.output_path,
    )
