from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cad_rl.config import (
    CompareSpec,
    DataSpec,
    EvalSpec,
    InferSpec,
    MeshSpec,
    ModelSpec,
    PrepareSpec,
    ProfileResolver,
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
CONFIGS_ROOT = ROOT / "configs"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage_resolution_merges_common_and_stage_configs() -> None:
    resolver = ProfileResolver(CONFIGS_ROOT)
    resolved = resolver.resolve("demo/train.yaml")

    assert resolved.profile_id == "demo_cad_rl"
    assert resolved.experiment_id == "demo"
    assert resolved.stage == "train"
    assert resolved.common_config_path.endswith("configs/demo/common.yaml")
    assert resolved.config_path.endswith("configs/demo/train.yaml")
    assert resolved.task.task_id == "cadquery_v1"
    assert resolved.data.prepared_datasets["train"] == "datasets/rendered_cadevolve_normalized_1_1_fixed"
    assert resolved.model.base_checkpoint == "Qwen/Qwen2-VL-2B-Instruct"
    assert resolved.train.scheduler_policy == "constant"
    assert resolved.train.num_generations == 16
    assert resolved.system.profile_id == "local"
    assert resolved.runtime.seed == 16
    assert resolved.runtime.dataset_split == "train"


def test_dataclasses_round_trip_and_serialization() -> None:
    checkpoint = CheckpointRef(
        run_id="run-1", step=12, path="/tmp/checkpoints/12", kind="best", score=0.98
    )
    config = RunConfig(
        profile_id="demo_cad_rl",
        experiment_id="demo",
        stage="infer",
        config_path="/tmp/configs/demo/infer.yaml",
        common_config_path="/tmp/configs/demo/common.yaml",
        system_config_path="/tmp/configs/systems/local.yaml",
        task=TaskSpec(task_id="cadquery_v1"),
        data=DataSpec(
            prepared_datasets={"train": "./datasets/train", "val": "./datasets/val"},
            hf_dataset=True,
        ),
        model=ModelSpec(
            model_family_adapter_id="qwen_family",
            base_checkpoint="Qwen/Qwen2-VL-2B-Instruct",
            attn_implementation="flash_attention_2",
        ),
        prepare=PrepareSpec(),
        train=TrainSpec(
            trainer_type="TopSampleGRPOTrainer",
            loss_mode="dr_grpo",
            importance_sampling_mode="sequence",
            top_k_policy="advantage_top_samples",
            optimizer_policy="adamw",
            scheduler_policy="constant",
        ),
        infer=InferSpec(output_path="/tmp/infer.jsonl"),
        mesh=MeshSpec(input_path="/tmp/infer.jsonl", output_dir="/tmp/meshes"),
        eval=EvalSpec(input_path="/tmp/meshes.jsonl", output_path="/tmp/eval.jsonl"),
        compare=CompareSpec(
            summaries=("/tmp/a.summary.json",), output_path="/tmp/report.json"
        ),
        runtime=RuntimeSpec(seed=16, dataset_split="train"),
        system=SystemConfig(profile_id="local", run_root="./runs"),
        extras={"note": "demo"},
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
    assert json.loads(json.dumps(to_serializable(manifest)))["latest_checkpoint"] == checkpoint.path


def test_run_registry_materializes_and_selects_checkpoints(tmp_path: Path) -> None:
    resolved = ProfileResolver(CONFIGS_ROOT).resolve("demo/train.yaml")
    registry = RunRegistry(tmp_path / "runs")

    manifest = registry.materialize_run(
        "run-1",
        resolved,
        seed=resolved.runtime.seed,
        dataset_fingerprint=resolved.runtime.dataset_fingerprint,
        latest_checkpoint="/tmp/runs/run-1/checkpoints/000002",
    )
    assert (tmp_path / "runs" / "run-1" / "resolved_config.json").exists()
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
    resolved = ProfileResolver(CONFIGS_ROOT).resolve("demo/train.yaml")
    registry = RunRegistry(tmp_path / "runs")
    manifest = registry.write_manifest_from_resolved(resolved, "run-2", seed=16)
    payload = materialize_resolved_run(tmp_path / "runs" / "run-2", resolved, manifest)
    assert payload["manifest"].exists()
    assert payload["resolved_config"].exists()
    resolved_payload = json.loads(
        (tmp_path / "runs" / "run-2" / "resolved_config.json").read_text()
    )
    assert resolved_payload["stage"] == "train"
    assert resolved_payload["system"]["profile_id"] == "local"


def test_public_clis_support_dry_run_without_heavy_dependencies() -> None:
    commands = [
        ("prepare_dataset.py", "configs/demo/prepare_dataset.yaml"),
        ("train.py", "configs/demo/train.yaml"),
        ("resume_train.py", "configs/demo/train.yaml"),
        ("infer.py", "configs/demo/infer.yaml"),
        ("build_meshes.py", "configs/demo/build_meshes.yaml"),
        ("evaluate.py", "configs/demo/evaluate.yaml"),
        ("compare_runs.py", "configs/demo/compare.yaml"),
    ]
    for script, config in commands:
        proc = _run_cli(script, "--config", config, "--dry-run")
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["config_path"].endswith(config)
