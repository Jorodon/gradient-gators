"""Cross-reward state and transition invariance.

The controlled comparison this project is built around only means something if
the reward calculator is the *only* thing that differs between runs. Same
scenario, same actions, same dynamics, same damage, same observations, same
termination - different scores. These tests are what makes that claim checkable
rather than asserted.

How it stays valid as variants land
-----------------------------------
Only `sparse` has a calculator today, so comparing "across variants" would be a
comparison of one trace against itself and would pass vacuously. So the suite
runs over every registered variant *plus* a deliberately hostile stub standing
in for each declared-but-unimplemented variant. The stubs read different parts
of the transition and return wildly different finite values, which is what makes
the invariance assertion bite right now. When a real calculator is registered it
simply replaces its stub, and this file needs no edit - so these tests gate the
reward PRs rather than trailing them.

`test_every_declared_variant_has_a_calculator` skips with the names of the
variants currently standing on stubs, so CI output never implies more coverage
than exists.

What counts as a reward-related output
--------------------------------------
`REWARD_INFO_KEYS` is the set of info keys allowed to differ. It is empty today:
`info` carries no reward metadata, because reward is returned separately. Every
other info key is compared, including keys added later. That direction is
deliberate - a new non-reward key is covered automatically, and a new *reward*
key makes these tests fail until someone adds it to the set on purpose.
"""

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.configs.training_config import RewardConfig
from src.environment.action_space import Action
from src.environment.gator_env import GatorEnv
from src.environment.generated_scenario import build_generated_scenario
from src.environment.map import GameMap, HazardType, Tile
from src.environment.scenario_collections import write_collection
from src.environment.scenario_serialization import scenario_to_dict
from src.environment.survival_validator import generate_surviving_research_world
from src.rewards.registry import (
    DECLARED_VARIANTS,
    REWARD_CALCULATORS,
    missing_variants,
    registered_variants,
)

# Info keys permitted to differ between variants. Empty on purpose: see the
# module docstring. Adding a key here is a deliberate statement that it is
# reward metadata and not part of the simulation.
REWARD_INFO_KEYS: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# Stand-in calculators for variants that are not implemented yet
# --------------------------------------------------------------------------- #


def _stub_efficiency(context, config) -> float:
    return -3.0 * context.step_number + 10.0 * context.events.distance_change


def _stub_safety(context, config) -> float:
    penalty = -25.0 * context.events.damage_taken
    return penalty - (100.0 if context.events.enemy_contact else 0.0)


def _stub_balanced(context, config) -> float:
    goal = 50.0 if context.successful_goal else 0.0
    return goal - 5.0 * context.events.damage_taken - 1.0


def _stub_progress(context, config) -> float:
    return 40.0 * context.events.distance_change + context.next.hp


# Each reads different fields and returns a different magnitude, so two traces
# that agree on everything but reward really are insensitive to the scoring.
STUB_CALCULATORS = {
    "efficiency": _stub_efficiency,
    "safety": _stub_safety,
    "balanced": _stub_balanced,
    "progress": _stub_progress,
}


@pytest.fixture
def comparison_variants(monkeypatch) -> tuple[str, ...]:
    """Every declared variant, real calculator where one exists, stub otherwise.

    GatorEnv resolves its calculator in __init__, so envs must be built after
    this fixture has run. Every helper below constructs its own env, so that
    ordering holds.
    """
    for variant in missing_variants():
        stub = STUB_CALCULATORS.get(variant)

        if stub is None:
            raise AssertionError(
                f"Declared variant {variant!r} has neither a calculator nor a stub"
            )

        monkeypatch.setitem(REWARD_CALCULATORS, variant, stub)

    return DECLARED_VARIANTS


def test_every_declared_variant_has_a_calculator():
    """Reports the implementation gap instead of hiding it behind the stubs."""
    missing = missing_variants()

    if missing:
        pytest.skip(
            "cross-reward invariance is running against stubs for: "
            + ", ".join(missing)
        )


def test_each_unimplemented_variant_has_a_stub_standing_in():
    """Without this, a newly declared variant would silently drop out of the sweep."""
    assert set(missing_variants()) <= set(STUB_CALCULATORS)


# --------------------------------------------------------------------------- #
# Trace capture
# --------------------------------------------------------------------------- #


def observation_signature(observation: dict) -> tuple:
    """An exactly comparable, hashable stand-in for a Dict observation.

    Comparing dicts of numpy arrays with `==` yields elementwise arrays, and
    asserting on those raises a truth-value error rather than failing cleanly -
    or, worse, gets wrapped in `all()` somewhere and stops checking anything.
    Raw bytes plus dtype and shape sidesteps that and catches dtype drift too.
    """
    return tuple(
        (key, str(observation[key].dtype), observation[key].shape, observation[key].tobytes())
        for key in sorted(observation)
    )


def map_signature(game_map: GameMap) -> tuple:
    """Full static map content, so a reward choice cannot have edited the world."""
    return (
        game_map.width,
        game_map.height,
        game_map.start,
        game_map.goal,
        tuple(
            tuple(
                (tile.elevation, tile.obstacle, tile.hazard, tile.special_traversal)
                for tile in row
            )
            for row in game_map.tiles
        ),
    )


@dataclass(frozen=True)
class StepRecord:
    """Everything one step produces except the reward."""

    action: int
    observation: tuple
    terminated: bool
    truncated: bool
    info: tuple
    game_map: tuple


def comparable_info(info: dict) -> tuple:
    """Info as a sorted tuple of raw values, minus the keys reward may vary.

    Raw values rather than their reprs: every value the env emits (ints, floats,
    strings, tuples, and the events dict) compares correctly with `==`, and
    keeping them raw lets the per-case witnesses read the same records the
    comparison uses, instead of needing a second capture that could drift from
    it.
    """
    return tuple(
        (key, value)
        for key, value in sorted(info.items())
        if key not in REWARD_INFO_KEYS
    )


def run_trace(
    build: Callable[[RewardConfig], GatorEnv],
    actions: Sequence[Action],
    reward_config: RewardConfig,
    *,
    reset_options: dict | None = None,
) -> tuple[list[StepRecord], list[float]]:
    """Drive one episode and split what it produced into non-reward and reward.

    Returns:
        tuple: (records, rewards). `records` is what must match across variants;
            `rewards` is the only thing allowed to differ.
    """
    env = build(reward_config)

    try:
        observation, info = env.reset(seed=0, options=reset_options)

        # The post-reset state is part of the invariant too: a reward choice must
        # not change where the episode starts.
        records = [
            StepRecord(
                action=-1,
                observation=observation_signature(observation),
                terminated=False,
                truncated=False,
                info=comparable_info(info),
                game_map=map_signature(env.game_map),
            )
        ]
        rewards: list[float] = []

        for action in actions:
            observation, reward, terminated, truncated, info = env.step(int(action))

            records.append(
                StepRecord(
                    action=int(action),
                    observation=observation_signature(observation),
                    terminated=terminated,
                    truncated=truncated,
                    info=comparable_info(info),
                    game_map=map_signature(env.game_map),
                )
            )
            rewards.append(reward)

            assert env.observation_space.contains(observation), (
                f"observation left the declared space under "
                f"{reward_config.variant!r} at step {len(rewards)}"
            )
            assert np.isfinite(reward), (
                f"non-finite reward under {reward_config.variant!r} "
                f"at step {len(rewards)}"
            )

            if terminated or truncated:
                break

        return records, rewards
    finally:
        env.close()


def assert_traces_match(baseline_variant: str, baseline, variant: str, candidate) -> None:
    """Compare two traces field by field, naming the first divergence."""
    assert len(candidate) == len(baseline), (
        f"{variant!r} ran {len(candidate)} steps, {baseline_variant!r} ran "
        f"{len(baseline)}: the reward changed the length of the episode"
    )

    for index, (expected, actual) in enumerate(zip(baseline, candidate)):
        for field in ("game_map", "observation", "terminated", "truncated", "info"):
            assert getattr(actual, field) == getattr(expected, field), (
                f"{field} differs at step {index} between {baseline_variant!r} "
                f"and {variant!r}"
            )


# --------------------------------------------------------------------------- #
# The cases the ticket requires
# --------------------------------------------------------------------------- #


def flat_map(width, height, *, start=(0, 0), goal=None, hazards=(), obstacles=()) -> GameMap:
    tiles = [
        [
            Tile(
                hazard=HazardType.DAMAGE if (x, y) in hazards else None,
                obstacle=(x, y) in obstacles,
            )
            for x in range(width)
        ]
        for y in range(height)
    ]
    return GameMap(
        width=width,
        height=height,
        tiles=tiles,
        start=start,
        goal=goal if goal is not None else (width - 1, height - 1),
    )


def fixed_env_builder(game_map, environment_config, **enemy):
    """A builder closing over the world, leaving only the reward config free."""

    def build(reward_config: RewardConfig) -> GatorEnv:
        return GatorEnv(
            game_map=game_map,
            environment_config=environment_config,
            map_config=MapConfig(width=game_map.width, height=game_map.height),
            reward_config=reward_config,
            **enemy,
        )

    return build


@dataclass(frozen=True)
class InvarianceCase:
    """One episode shape, plus proof that it exercised what it claims to.

    `witness` matters as much as the comparison. A swap-contact case in which no
    swap ever happens passes every invariance assertion while testing nothing,
    and nothing in the trace would reveal it.
    """

    name: str
    build: Callable[[RewardConfig], GatorEnv]
    actions: tuple[Action, ...]
    witness: Callable[[list[StepRecord]], None]


def events_of(record: StepRecord) -> dict:
    return dict(record.info)["events"]


def end_reason_of(record: StepRecord) -> str | None:
    return dict(record.info).get("episode_end_reason")


def case_fatal_goal_contact() -> InvarianceCase:
    """Enemy steps onto the goal tile as the agent arrives; death outranks goal."""
    build = fixed_env_builder(
        flat_map(2, 2, goal=(1, 0)),
        EnvironmentConfig(enemy_enabled=True, enemy_contact_damage=100.0, max_hp=100),
        enemy_spawn=(1, 1),
        enemy_patrol_route=[(1, 1), (1, 0)],
        enemy_patrol_index=0,
    )

    def witness(records):
        final = records[-1]
        events = events_of(final)
        assert events["enemy_contact"] is True
        assert events["reached_goal"] is False
        assert end_reason_of(final) == "death"
        assert final.terminated and not final.truncated

    return InvarianceCase("fatal_goal_contact", build, (Action.RIGHT,), witness)


def case_injury_after_wait() -> InvarianceCase:
    """Agent waits; the patrolling enemy walks onto its tile."""
    build = fixed_env_builder(
        flat_map(3, 3),
        EnvironmentConfig(enemy_enabled=True, enemy_contact_damage=10.0, max_hp=100),
        enemy_spawn=(1, 0),
        enemy_patrol_route=[(1, 0), (0, 0)],
        enemy_patrol_index=0,
    )

    def witness(records):
        first = records[1]
        events = events_of(first)
        assert events["waited"] is True
        assert events["enemy_contact"] is True
        assert events["enemy_damage"] == 10.0
        assert dict(first.info)["agent_hp"] == 90.0

    return InvarianceCase(
        "injury_after_wait", build, (Action.WAIT, Action.WAIT), witness
    )


def case_enemy_swap_contact() -> InvarianceCase:
    """Agent and enemy exchange tiles in one step.

    Patrol only. Under pursuit this branch is unreachable, because a pursuer
    moves toward the agent's post-move tile, so it can never be the tile the
    agent just left.
    """
    build = fixed_env_builder(
        flat_map(3, 3),
        EnvironmentConfig(enemy_enabled=True, enemy_contact_damage=10.0, max_hp=100),
        enemy_spawn=(1, 0),
        enemy_patrol_route=[(1, 0), (0, 0)],
        enemy_patrol_index=0,
    )

    def witness(records):
        first = records[1]
        info = dict(first.info)
        assert events_of(first)["enemy_contact"] is True
        # The swap itself: each ended on the other's starting tile.
        assert info["agent_position"] == (1, 0)
        assert info["enemy_position"] == (0, 0)

    return InvarianceCase(
        "enemy_swap_contact", build, (Action.RIGHT, Action.DOWN), witness
    )


def case_invalid_move() -> InvarianceCase:
    """A blocked step is recorded and costs a timestep without moving the agent."""
    build = fixed_env_builder(
        flat_map(3, 3, obstacles=((1, 0),)),
        EnvironmentConfig(),
    )

    def witness(records):
        off_grid = events_of(records[1])
        into_obstacle = events_of(records[2])

        assert off_grid["invalid_move"] is True
        assert into_obstacle["invalid_move"] is True
        assert dict(records[2].info)["agent_position"] == (0, 0)
        assert dict(records[2].info)["steps"] == 2

    return InvarianceCase(
        "invalid_move", build, (Action.LEFT, Action.RIGHT, Action.DOWN), witness
    )


def case_mask_padding() -> InvarianceCase:
    """A map well under the padded observation bound, so the mask carries zeros."""
    build = fixed_env_builder(flat_map(3, 3), EnvironmentConfig())

    def witness(records):
        # Signature is (key, dtype, shape, bytes) tuples sorted by key.
        mask = next(entry for entry in records[0].observation if entry[0] == "map_mask")
        _, dtype, shape, raw = mask

        assert shape == (10, 10)
        assert np.frombuffer(raw, dtype=dtype).sum() == 9

    return InvarianceCase(
        "mask_padding", build, (Action.RIGHT, Action.DOWN), witness
    )


def case_horizon_truncation() -> InvarianceCase:
    """Research-mode horizon reached by waiting; truncated, never terminated."""
    build = fixed_env_builder(
        flat_map(3, 3),
        EnvironmentConfig(episode_limit_mode="research"),
    )

    def witness(records):
        final = records[-1]

        assert final.truncated and not final.terminated
        assert end_reason_of(final) == "timestep_limit"
        assert dict(final.info)["steps"] == dict(final.info)["max_episode_steps"]

    # 4 steps per tile on a 3x3 map is 36; a few extra are never reached.
    return InvarianceCase(
        "horizon_truncation", build, tuple([Action.WAIT] * 40), witness
    )


CASES = [
    case_fatal_goal_contact(),
    case_injury_after_wait(),
    case_enemy_swap_contact(),
    case_invalid_move(),
    case_mask_padding(),
    case_horizon_truncation(),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_non_reward_outputs_are_identical_across_variants(case, comparison_variants):
    """The central invariance claim, on a fixed map, for each required case."""
    traces = {
        variant: run_trace(case.build, case.actions, RewardConfig(variant=variant))
        for variant in comparison_variants
    }

    baseline_variant = comparison_variants[0]
    baseline, _ = traces[baseline_variant]

    # The case really happened, before claiming anything about invariance over it.
    case.witness(baseline)

    for variant in comparison_variants[1:]:
        candidate, _ = traces[variant]
        assert_traces_match(baseline_variant, baseline, variant, candidate)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rewards_actually_differ_across_variants(case, comparison_variants):
    """Guards the invariance tests above from passing for the wrong reason.

    If every variant scored identically, the comparisons would hold trivially
    and tell us nothing about insensitivity to the reward.
    """
    scored = {
        variant: tuple(
            run_trace(case.build, case.actions, RewardConfig(variant=variant))[1]
        )
        for variant in comparison_variants
    }

    assert len(set(scored.values())) > 1, (
        f"all variants produced identical rewards on {case.name}, so the "
        "invariance comparison for it is vacuous"
    )


# --------------------------------------------------------------------------- #
# Mixed-size frozen snapshots
# --------------------------------------------------------------------------- #


def build_snapshot(size: int, seed: int, split: str):
    map_config = MapConfig(width=size, height=size, max_elevation=0)
    settings = MapGenerationConfig(
        dimension_mode="fixed",
        patch_count=0,
        generated_max_elevation=0,
        obstacle_density_range=(0.0, 0.1),
        hazard_density_range=(0.0, 0.1),
        traversal_density_range=(0.0, 0.0),
        enemy_required=True,
    )
    config = EnvironmentConfig(episode_limit_mode="research", enemy_enabled=True)

    world, witness = generate_surviving_research_world(
        map_config, settings, config, seed
    )

    return build_generated_scenario(
        world,
        witness,
        map_config,
        settings,
        config,
        collection_id="reward-invariance",
        split=split,
        code_version="test-sha",
    )


@pytest.fixture(scope="module")
def mixed_size_collection(tmp_path_factory):
    """One evaluation collection holding maps of different sizes.

    Different sizes in one collection is what makes the padded Dict observation
    load-bearing, so this is also the fixture the MultiInputPolicy check uses.
    """
    root = tmp_path_factory.mktemp("collections")
    scenarios = [
        build_snapshot(5, 123, "FINAL_ID"),
        build_snapshot(7, 77, "FINAL_ID"),
        build_snapshot(9, 2024, "FINAL_ID"),
    ]
    write_collection(root, "reward-invariance", "FINAL_ID", scenarios)

    sizes = {scenario.game_map.width for scenario in scenarios}
    assert len(sizes) > 1, "collection is not mixed-size"

    return root / "reward-invariance", scenarios


def snapshot_env_builder(collection_dir):
    def build(reward_config: RewardConfig) -> GatorEnv:
        return GatorEnv(
            map_config=MapConfig(width=5, height=5, map_mode="procedural"),
            reward_config=reward_config,
            scenario_collection_dir=collection_dir,
        )

    return build


def test_snapshot_invariance_across_variants_and_map_sizes(
    mixed_size_collection, comparison_variants
):
    """The same claim in frozen-scenario mode, over three different map sizes.

    Each scenario is driven by its own stored survival witness, so the action
    sequence is the one the generator proved survivable rather than something
    invented here.
    """
    collection_dir, scenarios = mixed_size_collection
    build = snapshot_env_builder(collection_dir)

    for scenario in scenarios:
        scenario_id = scenario_to_dict(scenario)["scenario_id"]
        options = {"scenario_id": scenario_id}
        actions = scenario.witness.actions

        traces = {
            variant: run_trace(
                build, actions, RewardConfig(variant=variant), reset_options=options
            )
            for variant in comparison_variants
        }

        baseline_variant = comparison_variants[0]
        baseline, _ = traces[baseline_variant]

        # The witness reached the goal alive, so the episode really ran.
        assert end_reason_of(baseline[-1]) == "goal"

        for variant in comparison_variants[1:]:
            candidate, _ = traces[variant]
            assert_traces_match(baseline_variant, baseline, variant, candidate)


def test_snapshot_identity_is_unaffected_by_the_reward(
    mixed_size_collection, comparison_variants
):
    """Scenario id, checksum and provenance must not depend on the reward variant."""
    collection_dir, scenarios = mixed_size_collection
    build = snapshot_env_builder(collection_dir)
    scenario_id = scenario_to_dict(scenarios[0])["scenario_id"]

    identities = set()

    for variant in comparison_variants:
        env = build(RewardConfig(variant=variant))
        _, info = env.reset(seed=0, options={"scenario_id": scenario_id})
        identities.add(
            (
                info["scenario_id"],
                info["collection_id"],
                info["split"],
                info["snapshot_checksum"],
                info["source_code_version"],
            )
        )
        env.close()

    assert len(identities) == 1


# --------------------------------------------------------------------------- #
# Gymnasium / SB3 research-mode behavior, per variant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", DECLARED_VARIANTS)
def test_api_compliance_holds_for_every_variant(variant, comparison_variants):
    """check_env on a research world, once per variant.

    The baseline suite already runs these checks under sparse; the point here is
    that selecting a different calculator does not change the API surface.
    """
    def make() -> GatorEnv:
        return GatorEnv(
            map_config=MapConfig(width=6, height=6, map_mode="procedural"),
            environment_config=EnvironmentConfig(episode_limit_mode="research"),
            reward_config=RewardConfig(variant=variant),
        )

    gym_check_env(make(), skip_render_check=True)
    sb3_check_env(make(), warn=False)


@pytest.mark.parametrize("variant", DECLARED_VARIANTS)
def test_rewards_stay_finite_and_observations_in_space(variant, comparison_variants):
    """Asserted inside run_trace for every step of every case; this is the
    procedural-world version, where the map is not hand-built."""
    def build(reward_config: RewardConfig) -> GatorEnv:
        return GatorEnv(
            map_config=MapConfig(width=6, height=6, map_mode="procedural"),
            environment_config=EnvironmentConfig(
                episode_limit_mode="research", enemy_enabled=True
            ),
            reward_config=reward_config,
        )

    actions = tuple([Action.RIGHT, Action.DOWN] * 12)
    records, rewards = run_trace(build, actions, RewardConfig(variant=variant))

    assert rewards
    assert all(np.isfinite(reward) for reward in rewards)
    assert len(records) == len(rewards) + 1


def test_multi_input_policy_trains_on_mixed_size_snapshots(
    mixed_size_collection, comparison_variants
):
    """MultiInputPolicy on padded Dict observations, over a mixed-size collection.

    A short learn() rather than a long one: the question is whether the policy
    accepts the padded Dict space across differing map sizes, not whether it
    learns anything.
    """
    from stable_baselines3 import PPO

    collection_dir, scenarios = mixed_size_collection
    scenario_id = scenario_to_dict(scenarios[0])["scenario_id"]

    class SingleScenarioEnv(GatorEnv):
        """Resets to one named scenario, since an evaluation split needs one."""

        def reset(self, *, seed=None, options=None):
            return super().reset(seed=seed, options={"scenario_id": scenario_id})

    env = SingleScenarioEnv(
        map_config=MapConfig(width=5, height=5, map_mode="procedural"),
        reward_config=RewardConfig(variant="sparse"),
        scenario_collection_dir=collection_dir,
    )

    model = PPO("MultiInputPolicy", env, n_steps=64, batch_size=32, verbose=0, seed=0)
    model.learn(total_timesteps=64)

    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)

    assert env.action_space.contains(int(action))
    env.close()
