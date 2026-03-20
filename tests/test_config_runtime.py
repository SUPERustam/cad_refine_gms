from __future__ import annotations

import json
from pathlib import Path

from cad_rl.config import (
    AlgorithmSpec,
    CheckpointRef,
    ComparisonReport,
    EvalRecord,
    InferenceRecord,
    MachineProfile,
    MeshRecord,
    ModelSpec,
    ResolvedExperimentConfig,
    RunManifest,
    TaskSpec,
    TrainerSpec,
    ProfileResolver,
    to_serializable,
)
from cad_rl.runtime.checkpoints import (
    select_best_checkpoint,
    select_checkpoint,
    select_latest_checkpoint,
    select_specific_checkpoint,
)
from cad_rl.runtime.comet import CometAdapter
from cad_rl.runtime.materialize import materialize_resolved_run
from cad_rl.runtime.registry import RunRegistry


CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_profile_resolution_loads_component_profiles() -> None:
    resolver = ProfileResolver(CONFIGS_ROOT)
    resolved = resolver.resolve("experiment.demo.yaml")

    assert resolved.profile_id == "demo_cad_rl"
    assert resolved.task.task_id == "cadquery_v1"
    assert resolved.task.output_var_name == "result"
    assert resolved.model.base_checkpoint == "Qwen/Qwen2-VL-2B-Instruct"
    assert resolved.algorithm.scheduler_policy == "constant"
    assert resolved.algorithm.importance_sampling_mode == "sequence"
    assert resolved.trainer.output_dir == "models/test_cadrecodev2"
    assert resolved.trainer.num_generations == 16
    assert resolved.machine.profile_id == "local"
    assert resolved.runtime["seed"] == 16


def test_dataclasses_round_trip_and_serialization() -> None:
    checkpoint = CheckpointRef(
        run_id="run-1", step=12, path="/tmp/checkpoints/12", kind="best", score=0.98
    )
    task = TaskSpec(
        task_id="cadquery_v1",
        prepared_datasets={"train": "./datasets/train", "val": "./datasets/val"},
        prompt_profile_id="qwen_vl_chat",
        render_profile_id="cadquery_default",
        output_var_name="result",
        normalization_mode="fixed",
        default_eval_suites=("quick", "standard"),
    )
    model = ModelSpec(
        model_family_adapter_id="qwen_family",
        base_checkpoint="Qwen/Qwen2-VL-2B-Instruct",
        sft_checkpoint="qwen-sft",
        processor_name="Qwen/Qwen2-VL-2B-Instruct",
        processor_kwargs={"padding_side": "left"},
        generation_defaults={"max_new_tokens": 64},
    )
    algorithm = AlgorithmSpec(
        trainer_type="TopSampleGRPOTrainer",
        loss_mode="dr_grpo",
        importance_sampling_mode="token",
        top_k_policy="advantage_top_samples",
        optimizer_policy="adamw",
        scheduler_policy="constant",
        reward_config={"failure_reward": -10.0},
        trainer_kwargs={"top_samples": 4},
    )
    machine = MachineProfile(
        profile_id="local",
        run_root="./runs",
        cache_dir="./.cache/cad_rl",
        checkpoints_root="checkpoints",
        artifact_root="artifacts",
        logs_root="logs",
        vllm_port=8001,
        world_size=1,
        cpu_workers=4,
        vllm_placement_policy="local",
        comet_enabled=False,
        environment={"TOKENIZERS_PARALLELISM": "false"},
    )
    trainer = TrainerSpec(
        output_dir="models/test_cadrecodev2",
        run_name="grpo_cadrecodev2_0",
        report_to=("comet_ml",),
    )
    manifest = RunManifest(
        run_id="run-1",
        resolved_config_path="/tmp/run-1/resolved_config.json",
        git_revision="abc123",
        seed=16,
        dataset_fingerprint="fingerprint",
        machine_profile_id="local",
        task_id="cadquery_v1",
        model_id="qwen_family",
        algorithm_id="TopSampleGRPOTrainer",
        checkpoint_inventory=(checkpoint,),
        latest_checkpoint=checkpoint.path,
        comet_experiment_id="comet-123",
    )
    resolved = ResolvedExperimentConfig(
        profile_id="demo",
        task=task,
        model=model,
        algorithm=algorithm,
        machine=machine,
        trainer=trainer,
        runtime={"seed": 16},
        extras={"note": "demo"},
    )
    assert ResolvedExperimentConfig.from_mapping(to_serializable(resolved)) == resolved
    assert TaskSpec.from_mapping(to_serializable(task)) == task
    assert ModelSpec.from_mapping(to_serializable(model)) == model
    assert AlgorithmSpec.from_mapping(to_serializable(algorithm)) == algorithm
    assert MachineProfile.from_mapping(to_serializable(machine)) == machine
    assert TrainerSpec.from_mapping(to_serializable(trainer)) == trainer
    assert ComparisonReport.from_mapping(
        {
            "report_id": "r1",
            "run_ids": ["run-1"],
            "checkpoint_refs": [to_serializable(checkpoint)],
            "leaderboard": [{"metric": "iou", "value": 0.9}],
            "checkpoint_history": [{"step": 12}],
            "summary": {"mean": 0.9},
            "metadata": {"source": "unit-test"},
        }
    ) == ComparisonReport(
        report_id="r1",
        run_ids=("run-1",),
        checkpoint_refs=(checkpoint,),
        leaderboard=({"metric": "iou", "value": 0.9},),
        checkpoint_history=({"step": 12},),
        summary={"mean": 0.9},
        metadata={"source": "unit-test"},
    )

    inference = InferenceRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        sample_id="sample-1",
        prompt="hello",
        raw_generation="print('hi')",
        wrapped_code="wrapped",
        render_path="/tmp/render.png",
        output_text="output",
        status="ok",
        duration_s=1.5,
        metadata={"source": "generate"},
    )
    mesh = MeshRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        sample_id="sample-1",
        mesh_path="/tmp/mesh.stl",
        status="ok",
        failure_reason=None,
        duration_s=2.0,
        metadata={"source": "mesh"},
    )
    eval_record = EvalRecord(
        run_id="run-1",
        checkpoint=checkpoint,
        task_id="cadquery_v1",
        suite="standard",
        split="val",
        metric_name="iou",
        value=0.91,
        sample_count=10,
        invalid_count=1,
        failure_count=0,
        summary={"median": 0.9},
        metadata={"source": "eval"},
    )
    assert InferenceRecord.from_mapping(to_serializable(inference)) == inference
    assert MeshRecord.from_mapping(to_serializable(mesh)) == mesh
    assert EvalRecord.from_mapping(to_serializable(eval_record)) == eval_record

    serialized = to_serializable(manifest)
    assert serialized["checkpoint_inventory"][0]["path"] == "/tmp/checkpoints/12"
    assert manifest.checkpoint_inventory[0].kind == "best"
    assert checkpoint.score == 0.98
    assert json.loads(json.dumps(to_serializable(checkpoint)))["run_id"] == "run-1"


def test_run_registry_materializes_and_selects_checkpoints(tmp_path: Path) -> None:
    resolver = ProfileResolver(CONFIGS_ROOT)
    resolved = resolver.resolve("experiment.demo.yaml")
    registry = RunRegistry(tmp_path / "runs")

    manifest = registry.materialize_run(
        "run-1",
        resolved,
        seed=resolved.runtime.get("seed"),
        dataset_fingerprint=str(resolved.runtime.get("dataset_fingerprint")),
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
    resolved = ProfileResolver(CONFIGS_ROOT).resolve("experiment.demo.yaml")
    registry = RunRegistry(tmp_path / "runs")
    manifest = registry.write_manifest_from_resolved(resolved, "run-2", seed=16)
    payload = materialize_resolved_run(tmp_path / "runs" / "run-2", resolved, manifest)
    assert payload["manifest"].exists()
    assert payload["resolved_config"].exists()
    assert (
        json.loads((tmp_path / "runs" / "run-2" / "resolved_config.json").read_text())[
            "profile_id"
        ]
        == "demo_cad_rl"
    )


def test_all_profile_variants_resolve() -> None:
    resolver = ProfileResolver(CONFIGS_ROOT)

    constant = resolver.resolve("experiment.dr_cppo_constant.yaml")
    gms = resolver.resolve("experiment.dr_cppo_gms.yaml")
    aoc_gms = resolver.resolve("experiment.dr_cppo_aoc_gms.yaml")
    cosine = resolver.resolve("experiment.dr_cppo_cosine.yaml")

    assert constant.trainer.output_dir == "models/test_cadrecodev2"
    assert gms.algorithm.reward_config["aoc_gms_coef"] == 10.0
    assert aoc_gms.trainer.output_dir == "models/test_auc_gms"
    assert aoc_gms.algorithm.reward_config["get_aoc_gms"] is True
    assert cosine.algorithm.scheduler_policy == "cosine"


def test_comet_adapter_is_noop_when_disabled() -> None:
    adapter = CometAdapter.disabled()
    adapter.log_config({"a": 1})
    adapter.log_metrics({"loss": 1.0}, step=1)
    adapter.log_text("note", "hello")
    adapter.log_asset("/tmp/file")
    adapter.finish()
    assert adapter.enabled is False


def test_supported_docs_do_not_reference_legacy_training_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    doc_paths = (
        root / "README.md",
        root / "docs" / "RL_Practical.md",
        root / "docs" / "Setup.md",
        root / "docs" / "Troubleshooting.md",
    )
    contents = "\n".join(path.read_text(encoding="utf-8") for path in doc_paths)

    legacy_tokens = (
        "--trl-config",
        "profiles/",
        "rl_train.py",
        "rl_train_cos_sched.py",
    )
    assert all(token not in contents for token in legacy_tokens)

    expected_tokens = (
        "prepare_dataset.py",
        "train.py",
        "resume_train.py",
        "infer.py",
        "build_meshes.py",
        "evaluate.py",
        "compare_runs.py",
        "configs/experiment.demo.yaml",
        "resolved_config.json",
        "manifest.json",
    )
    assert all(token in contents for token in expected_tokens)
