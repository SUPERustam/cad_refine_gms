#!/usr/bin/env python3
import argparse
import json

from cad_rl.evaluation import (
    evaluate_from_resolved,
    export_evaluation_contract,
    resolve_evaluation_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate mesh records and write evaluation summaries"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    config = resolve_evaluation_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_evaluation_contract(config), indent=2))
        return
    evaluate_from_resolved(config)


if __name__ == "__main__":
    main()
