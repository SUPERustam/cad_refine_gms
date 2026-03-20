#!/usr/bin/env python3
import argparse

from cad_rl.metrics.async_metrics import close_pool, init_pool

from cad_rl.pipelines.evaluation import evaluate_inference_records
from cad_rl.pipelines.inference import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--var-name", default="result")
    parser.add_argument("--pool-size", type=int, default=16)
    args = parser.parse_args()

    init_pool(args.pool_size)
    try:
        records = load_jsonl(args.input)
        evaluate_inference_records(records, args.output, var_name=args.var_name)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
