from src.configs.training_config import RewardConfig
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents


def test_sparse_reward_when_goal_reached():
    events = StepEvents(reached_goal=True)
    config = RewardConfig(goal_reward=100.0)

    reward = calculate_sparse_reward(events, config)

    assert reward == 100.0


def test_sparse_reward_when_goal_not_reached():
    events = StepEvents(reached_goal=False)
    config = RewardConfig(goal_reward=100.0)

    reward = calculate_sparse_reward(events, config)

    assert reward == 0.0