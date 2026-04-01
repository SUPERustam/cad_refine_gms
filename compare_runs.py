#!/usr/bin/env python3
import argparse
import json

from cad_rl.comparison import (
    compare_from_resolved,
    export_comparison_contract,
    resolve_comparison_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation summaries from a stage config"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    config = resolve_comparison_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_comparison_contract(config), indent=2))
        return
    compare_from_resolved(config)


if __name__ == "__main__":
    main()
