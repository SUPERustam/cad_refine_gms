#!/usr/bin/env python3
import argparse
import json

from cad_rl.training import (
    export_training_contract,
    resolve_training_config,
    train_from_resolved_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume training from a stage config"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="latest, best, step, or explicit checkpoint path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    resolved = resolve_training_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_training_contract(resolved), indent=2))
        return
    train_from_resolved_config(resolved, resume_reference=args.checkpoint)


if __name__ == "__main__":
    main()
