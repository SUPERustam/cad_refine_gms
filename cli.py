#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections.abc
from dataclasses import replace
import json
from pathlib import Path
import typing

import cad_rl
import cad_rl.config
import cad_rl.data
import cad_rl.runtime
try:  # pragma: no cover - optional stage dependency
    import cad_rl.comparison
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.comparison = None  # type: ignore[attr-defined]
try:  # pragma: no cover - optional stage dependency
    import cad_rl.data.prepare
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.data.prepare = None  # type: ignore[attr-defined]
try:  # pragma: no cover - optional stage dependency
    import cad_rl.evaluation
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.evaluation = None  # type: ignore[attr-defined]
try:  # pragma: no cover - optional stage dependency
    import cad_rl.execution
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.execution = None  # type: ignore[attr-defined]
try:  # pragma: no cover - optional stage dependency
    import cad_rl.inference
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.inference = None  # type: ignore[attr-defined]
try:  # pragma: no cover - optional stage dependency
    import cad_rl.training
except Exception:  # pragma: no cover - lightweight environments
    cad_rl.training = None  # type: ignore[attr-defined]


ResolvedConfig = cad_rl.config.RunConfig
StageRunner = collections.abc.Callable[..., typing.Any] | None


STAGES: dict[str, tuple[StageRunner, bool, str]] = {
    "prepare-dataset": (
        None
        if cad_rl.data.prepare is None
        else cad_rl.data.prepare.prepare_dataset_from_resolved,
        False,
        "Prepare a dataset from a stage config",
    ),
    "train": (
        None
        if cad_rl.training is None
        else cad_rl.training.train_from_resolved_config,
        False,
        "Train from a stage config",
    ),
    "resume-train": (
        None
        if cad_rl.training is None
        else cad_rl.training.train_from_resolved_config,
        True,
        "Resume training from a stage config",
    ),
    "infer": (
        None
        if cad_rl.inference is None
        else cad_rl.inference.run_inference_from_resolved,
        False,
        "Run inference from a stage config and emit inference records",
    ),
    "build-meshes": (
        None
        if cad_rl.execution is None
        else cad_rl.execution.build_meshes_from_resolved,
        False,
        "Execute CadQuery generations and emit mesh records",
    ),
    "evaluate": (
        None
        if cad_rl.evaluation is None
        else cad_rl.evaluation.evaluate_from_resolved,
        False,
        "Evaluate mesh records and write evaluation summaries",
    ),
    "compare-runs": (
        None
        if cad_rl.comparison is None
        else cad_rl.comparison.compare_from_resolved,
        False,
        "Compare evaluation summaries from a stage config",
    ),
}


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_checkpoint: bool = False,
) -> None:
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            default="latest",
            help="latest, best, step, or explicit checkpoint path",
        )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )


def _resolve_config(config_path: str, *, debug: bool = False) -> ResolvedConfig:
    resolved = cad_rl.config.resolve_run_config(
        config_path, profiles_root=Path(config_path).parents[1]
    )
    effective_debug = bool(debug or resolved.runtime.debug)
    if resolved.runtime.debug != effective_debug:
        resolved = replace(
            resolved,
            runtime=replace(resolved.runtime, debug=effective_debug),
        )
    return resolved


def _run_stage(args: argparse.Namespace) -> typing.Any:
    resolved = _resolve_config(args.config, debug=args.debug)
    if args.dry_run:
        print(json.dumps(cad_rl.config.to_serializable(resolved), indent=2))
        return None
    if args.run_stage is None:
        raise RuntimeError(f"Stage {args.command!r} is unavailable in this environment")
    cad_rl.runtime.setup_run_logging(
        resolved,
        stage=args.command,
        debug=resolved.runtime.debug,
    )
    if getattr(args, "checkpoint", None) is not None:
        return args.run_stage(resolved, resume_reference=args.checkpoint)
    return args.run_stage(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CAD RL stage CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (run_stage, include_checkpoint, description) in STAGES.items():
        stage_parser = subparsers.add_parser(name, help=description)
        stage_parser.description = description
        _add_common_args(stage_parser, include_checkpoint=include_checkpoint)
        stage_parser.set_defaults(handler=_run_stage, run_stage=run_stage)

    return parser


def main(argv: collections.abc.Sequence[str] | None = None) -> typing.Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
