#!/usr/bin/env python3
import argparse
import json

from cad_rl.pipelines.training import (
    export_training_contract,
    load_resolved_and_grpo,
    train_from_resolved_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume CAD RL training from the filesystem run registry"
    )
    parser.add_argument("--experiment", required=True, help="Experiment profile path")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest, best, step, or explicit checkpoint path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    resolved, grpo_args = load_resolved_and_grpo(args.experiment)
    if args.dry_run:
        print(json.dumps(export_training_contract(resolved), indent=2))
        return
    train_from_resolved_config(resolved, grpo_args, resume_reference=args.checkpoint)


if __name__ == "__main__":
    main()
