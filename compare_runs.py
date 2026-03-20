#!/usr/bin/env python3
import argparse

from cad_rl.pipelines.comparison import compare_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_summaries(args.summaries, args.output)


if __name__ == "__main__":
    main()
