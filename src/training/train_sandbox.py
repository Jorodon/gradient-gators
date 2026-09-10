import gymnasium as gym
import torch
from stable_baselines3 import PPO

from src.configs.training_config import TrainingConfig

def main():
    config = TrainingConfig()
    device = "cpu"
    env = gym.make("CartPole-v1")

    print(f"Using device: {device}...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate = config.learning_rate,
        seed = config.seed,
        verbose = 1,
        device=device
    )

    model.learn(total_timesteps = config.total_timesteps)
    env.close()

if __name__ == "__main__":
    main()
