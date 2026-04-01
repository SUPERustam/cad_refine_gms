#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

from cad_rl.comparison import (
    compare_from_resolved,
    export_comparison_contract,
    resolve_comparison_config,
)
from cad_rl.data.prepare import (
    export_prepare_contract,
    prepare_dataset_from_resolved,
    resolve_prepare_config,
)
from cad_rl.evaluation import (
    evaluate_from_resolved,
    export_evaluation_contract,
    resolve_evaluation_config,
)
from cad_rl.execution import (
    build_meshes_from_resolved,
    export_mesh_contract,
    resolve_mesh_config,
)
from cad_rl.inference import (
    export_inference_contract,
    resolve_inference_config,
    run_inference_from_resolved,
)
from cad_rl.training import (
    export_training_contract,
    resolve_training_config,
    train_from_resolved_config,
)


ResolvedConfig = Any
ResolveConfig = Callable[..., ResolvedConfig]
ExportContract = Callable[[ResolvedConfig], dict[str, Any]]
RunStage = Callable[..., Any]


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_checkpoint: bool = False,
) -> None:
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            default="latest",
            help="latest, best, step, or explicit checkpoint path",
        )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )


def _run_stage(args: argparse.Namespace) -> Any:
    resolved = args.resolve_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(args.export_contract(resolved), indent=2))
        return None
    if getattr(args, "checkpoint", None) is not None:
        return args.run_stage(resolved, resume_reference=args.checkpoint)
    return args.run_stage(resolved)


def _register_stage(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    name: str,
    description: str,
    resolve_config: ResolveConfig,
    export_contract: ExportContract,
    run_stage: RunStage,
    aliases: Sequence[str] = (),
    include_checkpoint: bool = False,
) -> None:
    parser = subparsers.add_parser(name, aliases=list(aliases), help=description)
    parser.description = description
    _add_common_args(parser, include_checkpoint=include_checkpoint)
    parser.set_defaults(
        resolve_config=resolve_config,
        export_contract=export_contract,
        run_stage=run_stage,
        handler=_run_stage,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CAD RL stage CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_stage(
        subparsers,
        name="prepare-dataset",
        aliases=("prepare",),
        description="Prepare a dataset from a stage config",
        resolve_config=resolve_prepare_config,
        export_contract=export_prepare_contract,
        run_stage=prepare_dataset_from_resolved,
    )
    _register_stage(
        subparsers,
        name="train",
        description="Train from a stage config",
        resolve_config=resolve_training_config,
        export_contract=export_training_contract,
        run_stage=train_from_resolved_config,
    )
    _register_stage(
        subparsers,
        name="resume-train",
        aliases=("resume",),
        description="Resume training from a stage config",
        resolve_config=resolve_training_config,
        export_contract=export_training_contract,
        run_stage=train_from_resolved_config,
        include_checkpoint=True,
    )
    _register_stage(
        subparsers,
        name="infer",
        description="Run inference from a stage config and emit inference records",
        resolve_config=resolve_inference_config,
        export_contract=export_inference_contract,
        run_stage=run_inference_from_resolved,
    )
    _register_stage(
        subparsers,
        name="build-meshes",
        aliases=("build",),
        description="Execute CadQuery generations and emit mesh records",
        resolve_config=resolve_mesh_config,
        export_contract=export_mesh_contract,
        run_stage=build_meshes_from_resolved,
    )
    _register_stage(
        subparsers,
        name="evaluate",
        description="Evaluate mesh records and write evaluation summaries",
        resolve_config=resolve_evaluation_config,
        export_contract=export_evaluation_contract,
        run_stage=evaluate_from_resolved,
    )
    _register_stage(
        subparsers,
        name="compare-runs",
        aliases=("compare",),
        description="Compare evaluation summaries from a stage config",
        resolve_config=resolve_comparison_config,
        export_contract=export_comparison_contract,
        run_stage=compare_from_resolved,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
