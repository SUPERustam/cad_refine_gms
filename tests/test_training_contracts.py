from cad_rl.algorithms.frozen import FrozenRewardConfig, FrozenTrainingConfig


def test_reward_defaults_preserve_existing_modes():
    reward = FrozenRewardConfig()
    assert reward.failure_reward == -10.0
    assert reward.iou_coef == 10.0
    assert reward.r_mode == "10_iou"
    assert reward.nc_params()["get_aoc_gms"] is False


def test_training_defaults_preserve_constant_scheduler_profile():
    training = FrozenTrainingConfig(sft_path="/tmp/model")
    assert training.scheduler == "constant"
    assert training.top_samples == 4
    assert training.resume_ckpt_path == ""
