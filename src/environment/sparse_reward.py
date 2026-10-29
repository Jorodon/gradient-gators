from src.configs.training_config import RewardConfig
from src.rewards.transition_context import TransitionContext

#Calculates sparse reward using transition data
def calculate_sparse_reward(context: TransitionContext, reward_config: RewardConfig) -> float:
    #sparse reward rewards successful goal
    if context.successful_goal:
        return reward_config.goal_reward

    return 0.0