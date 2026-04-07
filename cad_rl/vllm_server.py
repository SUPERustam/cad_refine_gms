from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import yaml


def _load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return dict(payload)


def _environment_from_config(config_path: str) -> dict[str, str]:
    payload = _load_config(config_path)
    system = dict(payload["system"])
    train = dict(payload["train"])
    model = dict(payload["model"])
    environment = {str(key): str(value) for key, value in system["environment"].items()}
    environment["CONFIG_PATH"] = str(config_path)
    environment["WORLD_SIZE"] = str(system["world_size"])
    environment["VLLM_PORT"] = str(train["vllm_server_port"])
    environment["SFT_CHECKPOINT"] = str(
        model.get("sft_checkpoint") or model["base_checkpoint"]
    )
    return environment


def _command_from_environment(environment: dict[str, str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        environment["SFT_CHECKPOINT"],
        "--host",
        environment["VLLM_HOST"],
        "--port",
        environment["VLLM_PORT"],
        "--dtype",
        environment["VLLM_DTYPE"],
        "--tensor-parallel-size",
        environment["VLLM_TENSOR_PARALLEL_SIZE"],
        "--gpu-memory-utilization",
        environment["VLLM_GPU_MEMORY_UTILIZATION"],
        "--max-model-len",
        environment["VLLM_MAX_MODEL_LEN"],
        "--max-num-seqs",
        environment["VLLM_MAX_NUM_SEQS"],
        "--limit-mm-per-prompt",
        environment["VLLM_LIMIT_MM_PER_PROMPT"],
    ]


def _print_exports(config_path: str) -> int:
    environment = _environment_from_config(config_path)
    for key, value in environment.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def _serve(config_path: str) -> int:
    environment = _environment_from_config(config_path)
    os.environ.update(environment)
    os.environ["CUDA_VISIBLE_DEVICES"] = environment["VLLM_VISIBLE_DEVICES"]
    os.execvp(sys.executable, _command_from_environment(environment))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM launcher for CAD RL configs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Print shell exports for a config")
    env_parser.add_argument("--config", required=True, help="Stage config path")

    serve_parser = subparsers.add_parser("serve", help="Start vLLM for a config")
    serve_parser.add_argument("--config", required=True, help="Stage config path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "env":
        return _print_exports(args.config)
    if args.command == "serve":
        return _serve(args.config)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
