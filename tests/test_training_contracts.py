from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

import cad_rl.training


class _TrainSpecStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def trainer_kwargs(self):
        raise AssertionError("trainer_kwargs should not be accessed")

    @property
    def reward_config(self):
        raise AssertionError("reward_config should not be accessed")


def _base_train_stub(**overrides):
    payload = {
        "output_dir": "models/test_cadrecodev2",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 3000,
        "log_completions": False,
        "logging_steps": 5,
        "num_generations": 16,
        "generation_batch_size": 64,
        "report_to": ("comet_ml",),
        "run_name": "grpo_cadrecodev2_0",
        "num_train_epochs": 20,
        "save_strategy": "steps",
        "save_steps": 150,
        "save_total_limit": None,
        "temperature": 1.0,
        "top_p": 0.99,
        "top_k": 50,
        "importance_sampling_level": "sequence",
        "loss_type": "dr_grpo",
        "epsilon": 0.1,
        "num_iterations": 3,
        "scale_rewards": False,
        "learning_rate": 3e-5,
        "use_vllm": True,
        "vllm_server_port": 8000,
        "bf16": True,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "beta": 0.0,
        "weight_decay": 0.0,
        "clip_cov": False,
        "top_samples": 4,
        "scheduler": "constant",
        "reward": {"failure_reward": -10.0, "iou_coef": 10.0},
    }
    payload.update(overrides)
    return _TrainSpecStub(**payload)


def _base_resolved_stub(train, tmp_path):
    return SimpleNamespace(
        experiment_id="demo",
        stage="train",
        task=SimpleNamespace(task_id="cadquery_v1"),
        data=SimpleNamespace(prepared_datasets={}, hf_dataset=True),
        model=SimpleNamespace(
            sft_checkpoint=None,
            base_checkpoint="Qwen/Qwen2-VL-2B-Instruct",
            processor_name=None,
            processor_kwargs={},
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ),
        train=train,
        runtime=SimpleNamespace(
            run_id="demo_train",
            dataset_split="train",
            git_revision=None,
            seed=16,
            comet_experiment_id=None,
            resume_path="",
            scheduler_training_steps=200000,
        ),
        system=SimpleNamespace(
            run_root=str(tmp_path / "runs"),
            profile_id="local",
            environment={},
            world_size=None,
            cache_dir=None,
        ),
    )


def test_grpo_payload_uses_allowlisted_keys_only() -> None:
    train = _base_train_stub()

    payload = cad_rl.training._build_grpo_config_payload(train)

    assert set(payload) == {
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
        "importance_sampling_level",
        "loss_type",
        "epsilon",
        "num_iterations",
        "scale_rewards",
        "learning_rate",
        "use_vllm",
        "vllm_server_port",
        "bf16",
        "gradient_checkpointing",
        "remove_unused_columns",
        "ddp_find_unused_parameters",
        "beta",
        "weight_decay",
    }
    assert "trainer_kwargs" not in payload
    assert "reward_config" not in payload


def test_reward_payload_uses_explicit_subsection_only() -> None:
    train = _base_train_stub(
        reward={
            "failure_reward": -12.0,
            "iou_coef": 11.0,
            "cd_coef": 0.25,
        }
    )

    payload = cad_rl.training._build_reward_config_payload(train)

    assert payload == {
        "failure_reward": -12.0,
        "iou_coef": 11.0,
        "cd_coef": 0.25,
    }
    assert "trainer_kwargs" not in payload
    assert "reward_config" not in payload


def test_missing_required_repo_owned_train_field_fails_before_trainer_construction(
    tmp_path, monkeypatch
) -> None:
    train = _base_train_stub()
    delattr(train, "scheduler")
    resolved = _base_resolved_stub(train, tmp_path)

    monkeypatch.setattr(
        cad_rl.training,
        "build_trainer",
        lambda *args, **kwargs: pytest.fail("trainer should not be built"),
    )
    stub_grpo = types.ModuleType("cad_rl.grpo")
    stub_grpo.FrozenTrainingConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    stub_grpo.FrozenRewardConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "cad_rl.grpo", stub_grpo)
    monkeypatch.setattr(
        sys.modules.setdefault("cad_rl.modeling", types.ModuleType("cad_rl.modeling")),
        "resolve_torch_dtype",
        lambda value: value,
        raising=False,
    )

    with pytest.raises(cad_rl.training.TrainingConfigError, match=r"train\.scheduler"):
        cad_rl.training.build_trainer_from_resolved(resolved, grpo_args=object())


def test_cli_dry_run_prints_canonical_contract_only(monkeypatch, capsys) -> None:
    canonical_contract = {
        "experiment_id": "demo",
        "stage": "train",
        "config_path": "/tmp/demo/train.yaml",
        "task": {"task_id": "cadquery_v1"},
        "data": {
            "prepared_datasets": {"train": "datasets/train"},
            "hf_dataset": True,
        },
        "model": {
            "base_checkpoint": "Qwen/Qwen2-VL-2B-Instruct",
            "sft_checkpoint": None,
            "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
            "processor_kwargs": {},
            "generation_defaults": {},
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
        "train": {
            "loss_type": "dr_grpo",
            "importance_sampling_level": "sequence",
            "scheduler": "constant",
            "clip_cov": False,
            "top_samples": 4,
            "reward": {"failure_reward": -10.0},
        },
        "runtime": {
            "seed": 16,
            "dataset_split": "train",
            "debug": False,
        },
        "system": {
            "profile_id": "local",
            "run_root": "runs",
            "cache_dir": None,
        },
    }

    stub_config = types.ModuleType("cad_rl.config")
    stub_config.RunConfig = object
    stub_config.resolve_run_config = lambda *args, **kwargs: object()
    stub_config.to_serializable = lambda resolved: canonical_contract
    monkeypatch.setitem(sys.modules, "cad_rl.config", stub_config)
    stub_runtime = types.ModuleType("cad_rl.runtime")
    stub_runtime.setup_run_logging = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "cad_rl.runtime", stub_runtime)

    import cad_rl

    cad_rl.config = stub_config
    cad_rl.runtime = stub_runtime

    import cli

    monkeypatch.setattr(cli, "_resolve_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli.cad_rl.config,
        "to_serializable",
        lambda resolved: canonical_contract,
    )

    cli.main(["train", "--config", "ignored.yaml", "--dry-run"])
    printed = json.loads(capsys.readouterr().out)

    assert "trainer_kwargs" not in printed["train"]
    assert "reward_config" not in printed["train"]
    assert printed["train"]["scheduler"] == "constant"
    assert printed["train"]["clip_cov"] is False
    assert printed["train"]["top_samples"] == 4
