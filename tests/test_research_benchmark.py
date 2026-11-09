import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_baselines3 import PPO

from src.training.benchmark_research import project_campaign


def test_campaign_projection_keeps_all_three_interaction_budgets_separate():
    rows = [dict(
        reward=reward, seed_slot=0, training_steps=50_000,
        validation_episodes=300, evaluation_status="completed",
        training_steps_per_second=1_000, validation_episodes_per_second=10,
        validation_steps_per_second=1_000, training_wall_seconds=50,
        validation_wall_seconds=30, checkpoint_count=3, checkpoint_bytes=3_000,
        baseline_checkpoint_bytes=1_000, trained_checkpoint_bytes=1_000,
    ) for reward in ("sparse", "efficiency", "safety", "balanced", "progress")]

    report = project_campaign(rows, wall_seconds=410)

    assert report["training_seconds"] == 6_250
    assert report["checkpoint_validation_seconds"] == 2_750
    assert report["m3_seconds_proxy"] == 9_050
    assert report["m4_seconds_at_id_episode_rate_proxy"] == 2_500
    assert report["m4_seconds_at_400_steps_and_id_step_rate_proxy"] == 10_000
    assert report["checkpoint_storage_bytes_proxy"] == 275_000
    assert report["approval_status"] == "pending_human_review"
    assert project_campaign(rows[:-1], 410) is None


@pytest.mark.parametrize("timeout, expected_reward", [(True, 10.9), (False, 1.0)])
def test_installed_sb3_bootstraps_only_time_limit_truncation(timeout, expected_reward):
    """Exercise SB3's actual VecEnv conversion and rollout reward correction"""
    class OneStepEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.ones(1, dtype=np.float32), 1.0, not timeout, timeout, {}

    model = PPO("MlpPolicy", OneStepEnv(), n_steps=2, batch_size=2,
                gamma=0.99, device="cpu", seed=0)
    
    # Known terminal V=10 isolates the installed algorithm's gamma*V correction.
    model.policy.predict_values = lambda obs: torch.full((len(obs), 1), 10.0)
    try:
        _, callback = model._setup_learn(2)
        callback.on_training_start({}, {})
        assert model.collect_rollouts(model.env, callback, model.rollout_buffer, 2)
        assert model.rollout_buffer.rewards[:, 0] == pytest.approx([expected_reward] * 2)
    finally:
        model.env.close()
