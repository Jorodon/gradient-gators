import pytest

from src.configs.training_config import RewardConfig


#ensures valid reward config is created successfully
def test_reward_config_valid():
    config = RewardConfig(
        variant="balanced",
        goal_reward=100.0,
        step_penalty=1.0,
        hazard_penalty=2.0,
        damage_penalty=3.0,
        enemy_penalty=4.0,
        progress_scale=5.0,
        progress_gamma=0.99,
    )

    assert config.variant == "balanced"
    assert config.goal_reward == 100.0
    assert config.progress_gamma == 0.99


#invalid reward variant should be rejected
def test_reward_config_rejects_invalid_variant():
    with pytest.raises(ValueError):
        RewardConfig(variant="invalid")


#nonfinite reward values should be rejected
@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_reward", float("inf")),
        ("step_penalty", float("nan")),
        ("hazard_penalty", float("inf")),
        ("damage_penalty", float("-inf")),
        ("enemy_penalty", float("nan")),
        ("progress_scale", float("inf")),
    ],
)
def test_reward_config_rejects_nonfinite_values(field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError):
        RewardConfig(**kwargs)


#invalid progress gamma should be rejected
@pytest.mark.parametrize(
    "gamma",
    [
        0.0,
        -0.1,
        1.1,
        float("inf"),
        float("nan"),
    ],
)
def test_reward_config_rejects_invalid_progress_gamma(gamma):
    with pytest.raises(ValueError):
        RewardConfig(progress_gamma=gamma)