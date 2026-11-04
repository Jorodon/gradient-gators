from src.configs.training_config import RewardConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv


#verifies reward component diagnostics are shown through env info
def test_reward_components_are_shown_in_info():
    env = GatorEnv(
        reward_config=RewardConfig(
            variant="efficiency",
            step_penalty=1.0,
        )
    )

    env.reset(seed=0)

    _, _, _, _, info = env.step(int(Action.WAIT))

    assert "reward_components" in info
    assert info["reward_components"]["step"] == -1.0

    env.close()