from dataclasses import dataclass

@dataclass
class RewardConfig:
    goal_reward: float = 100.0

@dataclass
class TrainingConfig:
    seed: int = 0
    total_timesteps: int = 10000
    learning_rate: float = 0.0003