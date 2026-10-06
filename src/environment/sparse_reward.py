from src.configs.training_config import RewardConfig
from src.environment.step_events import StepEvents

# Sparse reward calculation (if agent reached goal, give reward specified in config file)
def calculate_sparse_reward(events: StepEvents, reward_config: RewardConfig) -> float:
    if events.reached_goal:
        return reward_config.goal_reward

    return 0.0