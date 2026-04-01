import pytest

from cad_rl.config import ProfileResolver


def test_reward_defaults_preserve_existing_modes():
    grpo = pytest.importorskip("cad_rl.grpo")
    FrozenRewardConfig = grpo.FrozenRewardConfig
    reward = FrozenRewardConfig()
    assert reward.failure_reward == -10.0
    assert reward.iou_coef == 10.0
    assert reward.r_mode == "10_iou"
    assert reward.nc_params()["get_aoc_gms"] is False


def test_training_defaults_preserve_constant_scheduler_profile():
    grpo = pytest.importorskip("cad_rl.grpo")
    FrozenTrainingConfig = grpo.FrozenTrainingConfig
    training = FrozenTrainingConfig(sft_path="/tmp/model")
    assert training.scheduler == "constant"
    assert training.top_samples == 4
    assert training.resume_ckpt_path == ""


def test_train_stage_contract_resolves_from_new_config_tree():
    resolved = ProfileResolver("configs").resolve("demo/train.yaml")
    assert resolved.stage == "train"
    assert resolved.runtime.dataset_split == "train"
    assert resolved.system.profile_id == "local"
