from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cad_rl.config import (
    ConfigError,
    CompareSpec,
    DataSpec,
    EvalSpec,
    InferSpec,
    MeshSpec,
    ModelSpec,
    PrepareSpec,
    ProfileResolver,
    RewardSpec,
    RunConfig,
    RuntimeSpec,
    SystemConfig,
    TaskSpec,
    TrainSpec,
    to_serializable,
)
from cad_rl.runtime import (
    CheckpointRef,
    ComparisonReport,
    EvalRecord,
    InferenceRecord,
    MeshRecord,
    RunManifest,
    RunRegistry,
    materialize_resolved_run,
    select_best_checkpoint,
    select_checkpoint,
    select_latest_checkpoint,
    select_specific_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_stage_config(tmp_path: Path, *, stage: str = "train") -> Path:
    config = {
        "profile_id": "demo_cad_rl",
        "experiment_id": "demo",
        "stage": stage,
        "task": {
            "task_id": "cadquery_v1",
            "output_var_name": "result",
            "default_eval_suites": ["quick", "standard", "full"],
        },
        "data": {
            "prepared_datasets": {
                "train": "datasets/rendered_cadevolve_normalized_1_1_fixed",
                "val": "datasets/rendered_cadevolve_normalized_1_1_fixed",
            },
            "prompt_profile_id": "qwen_vl_chat",
            "render_profile_id": "cadquery_default",
            "normalization_mode": "fixed",
            "prompt_convention": "multimodal_chat",
            "render_convention": "cadquery_compound",
            "hf_dataset": True,
        },
        "model": {
            "base_checkpoint": "Qwen/Qwen2-VL-2B-Instruct",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "sft_checkpoint": None,
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "processor_kwargs": {
                "trust_remote_code": True,
                "resized_width": 476,
                "resized_height": 952,
                "padding_side": "left",
            },
            "generation_defaults": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 4200,
            },
        },
        "prepare": {"split": "train"},
        "train": {
            "loss_type": "dr_grpo",
            "importance_sampling_level": "sequence",
            "scheduler": "constant",
            "reward": {
                "failure_reward": -10.0,
                "iou_coef": 10.0,
                "cd_coef": 0.0,
                "auc_coef": 0.0,
                "aoc_gms_coef": 0.0,
                "r_mode": "10_iou",
            },
            "clip_cov": False,
            "top_samples": 4,
            "use_vllm": True,
            "vllm_server_port": 8000,
            "bf16": True,
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
            "ddp_find_unused_parameters": False,
            "beta": 0,
            "weight_decay": 0,
            "output_dir": "models/test_cadrecodev2",
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "max_completion_length": 3000,
            "log_completions": False,
            "logging_steps": 5,
            "num_generations": 16,
            "generation_batch_size": 64,
            "report_to": ["comet_ml"],
            "run_name": "grpo_cadrecodev2_0",
            "num_train_epochs": 20,
            "save_strategy": "steps",
            "save_steps": 150,
            "save_total_limit": None,
            "temperature": 1.0,
            "top_p": 0.99,
            "top_k": 50,
            "epsilon": 0.1,
            "num_iterations": 3,
            "scale_rewards": False,
            "learning_rate": 3e-5,
        },
        "infer": {
            "split": "val",
            "output_path": "runs/demo/infer/inference_records.jsonl",
            "checkpoint_kind": "specific",
            "batch_size": 8,
            "num_workers": 0,
        },
        "mesh": {
            "input_path": "runs/demo/infer/inference_records.jsonl",
            "output_dir": "runs/demo/build_meshes",
        },
        "eval": {
            "input_path": "runs/demo/build_meshes/mesh_records.jsonl",
            "output_path": "runs/demo/evaluate/eval_records.jsonl",
            "split": "val",
            "suite": "standard",
        },
        "compare": {
            "summaries": ["runs/demo/evaluate/eval_records.summary.json"],
            "output_path": "runs/demo/compare/report.json",
        },
        "runtime": {
            "seed": 16,
            "dataset_split": "train",
            "run_id": "demo_train",
            "debug": False,
            "resume_path": "",
            "scheduler_training_steps": 200000,
        },
        "system": {
            "profile_id": "local",
            "run_root": str(tmp_path / "runs"),
            "cache_dir": None,
        },
    }
    config_path = tmp_path / f"{stage}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_stage_resolution_uses_single_file_contract(tmp_path: Path) -> None:
    config_path = _write_stage_config(tmp_path)
    resolved = ProfileResolver(tmp_path).resolve(config_path)

    assert resolved.profile_id == "demo_cad_rl"
    assert resolved.experiment_id == "demo"
    assert resolved.stage == "train"
    assert not hasattr(resolved, "common_config_path")
    assert not hasattr(resolved, "system_config_path")
    assert resolved.config_path == str(config_path)
    assert resolved.task.task_id == "cadquery_v1"
    assert (
        resolved.data.prepared_datasets["train"]
        == "datasets/rendered_cadevolve_normalized_1_1_fixed"
    )
    assert resolved.model.base_checkpoint == "Qwen/Qwen2-VL-2B-Instruct"
    assert resolved.train.scheduler == "constant"
    assert resolved.train.num_generations == 16
    assert resolved.system.profile_id == "local"
    assert resolved.runtime.seed == 16
    assert resolved.runtime.dataset_split == "train"


def test_resolver_rejects_missing_required_train_keys(tmp_path: Path) -> None:
    config_path = _write_stage_config(tmp_path)
    payload = json.loads(config_path.read_text())
    del payload["train"]["loss_type"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="train"):
        ProfileResolver(tmp_path).resolve(config_path)


@pytest.mark.parametrize(
    ("section", "unknown_key"),
    [
        ("train", "trainer_type"),
        ("model", "checkpoint_path"),
        ("runtime", "machine"),
        ("system", "checkpoint_path"),
    ],
)
def test_resolver_rejects_unknown_keys_in_strict_sections(
    tmp_path: Path, section: str, unknown_key: str
) -> None:
    config_path = _write_stage_config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload[section][unknown_key] = "unexpected"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError):
        ProfileResolver(tmp_path).resolve(config_path)


def test_dataclasses_round_trip_and_serialization() -> None:
    checkpoint = CheckpointRef(
        run_id="run-1", step=12, path="/tmp/checkpoints/12", kind="best", score=0.98
    )
    config = RunConfig(
        profile_id="demo_cad_rl",
        experiment_id="demo",
        stage="infer",
        config_path="/tmp/configs/demo/infer.yaml",
        task=TaskSpec(task_id="cadquery_v1"),
        data=DataSpec(
            prepared_datasets={"train": "./datasets/train", "val": "./datasets/val"},
            prompt_profile_id="qwen_vl_chat",
            render_profile_id="cadquery_default",
            normalization_mode="fixed",
            hf_dataset=True,
        ),
        model=ModelSpec(
            base_checkpoint="Qwen/Qwen2-VL-2B-Instruct",
            sft_checkpoint=None,
            processor_name="Qwen/Qwen2-VL-2B-Instruct",
            processor_kwargs={},
            generation_defaults={},
            trust_remote_code=True,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        ),
        prepare=PrepareSpec(split="train"),
        train=TrainSpec(
            loss_type="dr_grpo",
            importance_sampling_level="sequence",
            scheduler="constant",
            reward=RewardSpec(),
            clip_cov=False,
            top_samples=4,
            use_vllm=True,
            vllm_server_port=8000,
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            beta=0.0,
            weight_decay=0.0,
            output_dir="models/test",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            max_completion_length=3000,
            log_completions=False,
            logging_steps=5,
            num_generations=16,
            generation_batch_size=64,
            report_to=("comet_ml",),
            run_name="demo",
            num_train_epochs=20,
            save_strategy="steps",
            save_steps=150,
        ),
        infer=InferSpec(split="val", output_path="/tmp/infer.jsonl"),
        mesh=MeshSpec(input_path="/tmp/infer.jsonl", output_dir="/tmp/meshes"),
        eval=EvalSpec(input_path="/tmp/meshes.jsonl", output_path="/tmp/eval.jsonl"),
        compare=CompareSpec(
            summaries=("/tmp/a.summary.json",), output_path="/tmp/report.json"
        ),
        runtime=RuntimeSpec(seed=16, dataset_split="train"),
        system=SystemConfig(profile_id="local", run_root="./runs"),
    )
    manifest = RunManifest(
        run_id="run-1",
        resolved_config_path="/tmp/run-1/resolved_config.json",
        latest_checkpoint=checkpoint.path,
        checkpoint_inventory=(checkpoint,),
    )
    inference = InferenceRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        sample_id="sample-1",
        raw_generation="result = cq.Workplane()",
        wrapped_code="result = cq.Workplane()",
        metadata={"target_mesh_path": "/tmp/target.stl"},
    )
    mesh = MeshRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        sample_id="sample-1",
        mesh_path="/tmp/pred.stl",
        metadata={"target_mesh_path": "/tmp/target.stl"},
    )
    eval_record = EvalRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        task_id="cadquery_v1",
        suite="standard",
        split="val",
        metric_name="iou",
        value=0.91,
        sample_count=1,
    )
    comparison = ComparisonReport(
        report_id="comparison",
        run_ids=("run-1",),
        checkpoint_refs=(checkpoint,),
        leaderboard=({"iou_mean": 0.91},),
    )

    assert RunConfig.from_mapping(to_serializable(config)) == config
    assert InferenceRecord.from_mapping(to_serializable(inference)) == inference
    assert MeshRecord.from_mapping(to_serializable(mesh)) == mesh
    assert EvalRecord.from_mapping(to_serializable(eval_record)) == eval_record
    assert ComparisonReport.from_mapping(to_serializable(comparison)) == comparison
    assert (
        json.loads(json.dumps(to_serializable(manifest)))["latest_checkpoint"]
        == checkpoint.path
    )


def test_run_registry_materializes_and_selects_checkpoints(tmp_path: Path) -> None:
    resolved = ProfileResolver(tmp_path).resolve(_write_stage_config(tmp_path))
    registry = RunRegistry(tmp_path / "runs")

    manifest = registry.materialize_run(
        "run-1",
        resolved,
        seed=resolved.runtime.seed,
        dataset_fingerprint=resolved.runtime.dataset_fingerprint,
        latest_checkpoint="/tmp/runs/run-1/checkpoints/000002",
    )
    assert (tmp_path / "runs" / "run-1" / "resolved_config.json").exists()
    assert (tmp_path / "runs" / "run-1" / "logs").is_dir()
    assert manifest.latest_checkpoint == "/tmp/runs/run-1/checkpoints/000002"

    latest = registry.append_checkpoint(
        CheckpointRef(
            run_id="run-1",
            step=1,
            path="/tmp/runs/run-1/checkpoints/000001",
            kind="specific",
        )
    )
    best = registry.append_checkpoint(
        CheckpointRef(
            run_id="run-1",
            step=2,
            path="/tmp/runs/run-1/checkpoints/000002",
            kind="best",
            score=0.91,
        )
    )
    assert registry.read_latest_checkpoint("run-1").path == best.path
    assert select_latest_checkpoint("run-1", registry).path == best.path
    assert select_best_checkpoint("run-1", registry).path == best.path
    assert select_specific_checkpoint("run-1", registry, 1).path == latest.path
    assert select_checkpoint("run-1", registry, "best").path == best.path


def test_materialize_run_writes_json_files(tmp_path: Path) -> None:
    resolved = ProfileResolver(tmp_path).resolve(_write_stage_config(tmp_path))
    registry = RunRegistry(tmp_path / "runs")
    manifest = registry.write_manifest_from_resolved(resolved, "run-2", seed=16)
    payload = materialize_resolved_run(tmp_path / "runs" / "run-2", resolved, manifest)
    assert payload["manifest"].exists()
    assert payload["resolved_config"].exists()
    assert (tmp_path / "runs" / "run-2" / "logs").is_dir()
    resolved_payload = json.loads(
        (tmp_path / "runs" / "run-2" / "resolved_config.json").read_text()
    )
    assert resolved_payload["stage"] == "train"
    assert resolved_payload["system"]["profile_id"] == "local"


def test_public_cli_subcommands_support_dry_run_without_heavy_dependencies(
    tmp_path: Path,
) -> None:
    config_path = _write_stage_config(tmp_path)
    commands = [
        ("prepare-dataset", "prepare"),
        ("train", "train"),
        ("resume-train", "resume"),
        ("infer", "infer"),
        ("build-meshes", "build"),
        ("evaluate", "evaluate"),
        ("compare-runs", "compare"),
    ]
    for command, stage in commands:
        staged_config = (
            config_path
            if stage == "train"
            else _write_stage_config(tmp_path, stage=stage)
        )
        proc = _run_cli("cli.py", command, "--config", str(staged_config), "--dry-run")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["config_path"] == str(staged_config)
        assert "common_config_path" not in payload
        assert "system_config_path" not in payload


def test_cli_dry_run_accepts_debug_override(tmp_path: Path) -> None:
    config_path = _write_stage_config(tmp_path)
    proc = _run_cli(
        "cli.py",
        "train",
        "--config",
        str(config_path),
        "--dry-run",
        "--debug",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["config_path"] == str(config_path)
    assert payload["runtime"]["debug"] is True
    assert set(payload) == {
        "profile_id",
        "experiment_id",
        "stage",
        "config_path",
        "task",
        "data",
        "model",
        "prepare",
        "train",
        "infer",
        "mesh",
        "eval",
        "compare",
        "runtime",
        "system",
    }
    assert set(payload["train"]) == {
        "loss_type",
        "importance_sampling_level",
        "scheduler",
        "reward",
        "clip_cov",
        "top_samples",
        "use_vllm",
        "vllm_server_port",
        "bf16",
        "gradient_checkpointing",
        "remove_unused_columns",
        "ddp_find_unused_parameters",
        "beta",
        "weight_decay",
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_completion_length",
        "log_completions",
        "logging_steps",
        "num_generations",
        "generation_batch_size",
        "report_to",
        "run_name",
        "num_train_epochs",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "temperature",
        "top_p",
        "top_k",
        "epsilon",
        "num_iterations",
        "scale_rewards",
        "learning_rate",
    }
    assert set(payload["train"]["reward"]) == {
        "failure_reward",
        "iou_coef",
        "cd_coef",
        "auc_coef",
        "aoc_gms_coef",
        "get_nc",
        "nc_n_points",
        "nc_tol",
        "print_sample_steps",
        "pool_size",
        "r_mode",
        "get_aoc_gms",
        "aoc_gms_n_points",
        "aoc_gms_n_angles",
        "aoc_gms_rel_tol",
        "aoc_gms_cube_trick",
        "aoc_gms_pc_cache_enable",
        "aoc_gms_upper_bound_tol_rt",
        "aoc_gms_autofix_sampling",
    }
    assert "trainer_type" not in payload["train"]
    assert "loss_mode" not in payload["train"]
    assert "importance_sampling_mode" not in payload["train"]
    assert "top_k_policy" not in payload["train"]
    assert "optimizer_policy" not in payload["train"]
    assert "scheduler_policy" not in payload["train"]
    assert "reward_config" not in payload["train"]
    assert "trainer_kwargs" not in payload["train"]
