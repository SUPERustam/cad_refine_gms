from __future__ import annotations

import json
from pathlib import Path


def _load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_summaries(summary_paths: list[str | Path], output_path: str | Path) -> dict:
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
    report = {
        "leaderboard": leaderboard,
        "checkpoint_over_time": [
            {
                "source": item["source"],
                "iou_mean": item.get("iou_mean"),
                "cd_mean": item.get("cd_mean"),
                "invalid_fraction": item.get("invalid_fraction"),
            }
            for item in summaries
        ],
        "diff_report": [
            {
                "source": item["source"],
                "missing_sample_count": item.get("missing_sample_count"),
                "iou_mean": item.get("iou_mean"),
                "cd_mean": item.get("cd_mean"),
            }
            for item in leaderboard
        ],
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
