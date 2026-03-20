#!/usr/bin/env python3
import argparse

from cad_rl.pipelines.inference import load_jsonl
from cad_rl.pipelines.mesh import build_mesh_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--var-name", default="result")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    build_mesh_records(records, args.output_dir, var_name=args.var_name)


if __name__ == "__main__":
    main()
