from stable_baselines3 import PPO

from src.configs.training_config import TrainingConfig
from src.environment.gator_env import GatorEnv

def main():
    config = TrainingConfig()
    userDevice = "cpu"
    env = GatorEnv()

    print(f"Using device: {userDevice}...")

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate = config.learning_rate,
        seed = config.seed,
        verbose = 1,
        device=userDevice
    )

    model.learn(total_timesteps = config.total_timesteps)
    env.close()

if __name__ == "__main__":
    main()
