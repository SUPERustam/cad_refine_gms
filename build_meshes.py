#!/usr/bin/env python3
import argparse
import json

from cad_rl.execution import (
    build_meshes_from_resolved,
    export_mesh_contract,
    resolve_mesh_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute CadQuery generations and emit mesh records"
    )
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--system", help="Optional system overlay path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the resolved contract"
    )
    args = parser.parse_args()

    config = resolve_mesh_config(args.config, system=args.system)
    if args.dry_run:
        print(json.dumps(export_mesh_contract(config), indent=2))
        return
    build_meshes_from_resolved(config)


if __name__ == "__main__":
    main()
