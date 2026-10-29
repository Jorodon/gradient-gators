from src.configs.environment_config import EnvironmentConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.map import GameMap, HazardType, Tile
from src.environment.sparse_reward import calculate_sparse_reward


#creates a basic flat map
def make_flat_map(width=3, height=3, *, start=(0, 0), goal=None):
    return GameMap(
        width=width,
        height=height,
        tiles=[
            [Tile() for _ in range(width)]
            for _ in range(height)
        ],
        start=start,
        goal=goal if goal is not None else (width - 1, height - 1),
    )


#captures each TransitionContext that is passed to sparse reward
def capture_contexts(monkeypatch):
    contexts = []

    def recording_sparse_reward(context, reward_config):
        contexts.append(context)
        return calculate_sparse_reward(context, reward_config)

    monkeypatch.setattr(
        "src.environment.gator_env.calculate_sparse_reward",
        recording_sparse_reward,
    )

    return contexts


#lethal damage on the goal = death not success
def test_context_for_lethal_goal_contact(monkeypatch):
    contexts = capture_contexts(monkeypatch)

    game_map = make_flat_map(
        start=(0, 1),
        goal=(1, 1),
    )
    game_map.get_tile(1, 1).hazard = HazardType.DAMAGE

    env = GatorEnv(
        game_map=game_map,
        environment_config=EnvironmentConfig(
            max_hp=5,
            hazard_damage=7.0,
        ),
    )

    try:
        env.reset(seed=0)
        env.step(Action.RIGHT)

        context = contexts[-1]

        assert context.next.position == game_map.goal
        assert context.next.hp == 0.0
        assert context.death is True
        assert context.successful_goal is False
        assert context.terminated is True
        assert context.truncated is False
        assert context.events.reached_goal is False
        assert context.events.hazard_contact is True
    finally:
        env.close()


#living goal on the final allowed step counts as reaching goal successfully
def test_context_for_living_goal_on_final_step(monkeypatch):
    contexts = capture_contexts(monkeypatch)

    env = GatorEnv(
        game_map=make_flat_map(
            width=5,
            height=5,
            goal=(1, 0),
        ),
        environment_config=EnvironmentConfig(
            episode_limit_mode="research",
        ),
    )

    try:
        env.reset(seed=0)

        for _ in range(99):
            env.step(Action.WAIT)

        env.step(Action.RIGHT)

        context = contexts[-1]

        assert context.step_number == 100
        assert context.successful_goal is True
        assert context.death is False
        assert context.timeout is False
        assert context.terminated is True
        assert context.truncated is False
        assert context.events.reached_goal is True
    finally:
        env.close()


#repeated WAIT actions must produce valid finalized contexts
def test_context_for_repeated_wait(monkeypatch):
    contexts = capture_contexts(monkeypatch)

    env = GatorEnv(
        game_map=make_flat_map(),
        environment_config=EnvironmentConfig(max_steps=10),
    )

    try:
        env.reset(seed=0)

        env.step(Action.WAIT)
        env.step(Action.WAIT)
        env.step(Action.WAIT)

        assert len(contexts) == 3

        for context in contexts:
            assert context.previous.position == context.next.position
            assert context.events.waited is True
            assert context.events.step_taken is True
            assert context.events.distance_change == 0.0
    finally:
        env.close()


#time limit ending must be truncation rather than true termination
def test_context_for_time_limit_truncation(monkeypatch):
    contexts = capture_contexts(monkeypatch)

    env = GatorEnv(
        game_map=make_flat_map(width=5, height=5),
        environment_config=EnvironmentConfig(
            episode_limit_mode="research",
        ),
    )

    try:
        env.reset(seed=0)

        for _ in range(100):
            env.step(Action.WAIT)

        context = contexts[-1]

        assert context.step_number == 100
        assert context.timeout is True
        assert context.successful_goal is False
        assert context.death is False
        assert context.terminated is False
        assert context.truncated is True
        assert context.end_reason == "timestep_limit"
    finally:
        env.close()