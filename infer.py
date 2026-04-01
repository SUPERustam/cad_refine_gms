#!/usr/bin/env python3
import argparse
import json

from cad_rl.inference import (
    export_inference_contract,
    resolve_inference_config,
    run_inference_from_resolved,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference from a stage config and emit inference records"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    config = resolve_inference_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_inference_contract(config), indent=2))
        return
    run_inference_from_resolved(config)


if __name__ == "__main__":
    main()
