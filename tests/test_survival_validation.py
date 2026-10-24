import pytest

import src.environment.research_world_generator as world_generator
import src.environment.survival_validator as validator
from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.map import GameMap, HazardType, Tile
from src.environment.map_generator import has_valid_route
from src.environment.patrol_generator import EnemySpec
from src.environment.witness_simulator import simulate_actions


def flat_map(*, start=(0, 0), goal=(2, 0)) -> GameMap:
    return GameMap(
        width=5,
        height=5,
        tiles=[[Tile() for _ in range(5)] for _ in range(5)],
        start=start,
        goal=goal,
    )


def research_config(**changes) -> EnvironmentConfig:
    values = {"episode_limit_mode": "research", "enemy_enabled": False}
    values.update(changes)
    return EnvironmentConfig(**values)


def simple_settings(**changes) -> MapGenerationConfig:
    values = {
        "patch_count": 0,
        "enemy_required": False,
        "obstacle_density_range": (0.0, 0.0),
        "hazard_density_range": (0.0, 0.0),
        "traversal_density_range": (0.0, 0.0),
    }
    values.update(changes)
    return MapGenerationConfig(**values)

# Ensures BFS returns actions and handles uphill traversal
def test_shortest_route_returns_actions_and_respects_directed_ascent():
    game_map = flat_map()
    assert validator.find_shortest_route(game_map) == (Action.RIGHT, Action.RIGHT)

    game_map = flat_map(goal=(1, 0))
    game_map.get_tile(1, 0).elevation = 1
    game_map.get_tile(0, 1).obstacle = True

    assert validator.find_shortest_route(game_map) is None
    assert not has_valid_route(game_map)

    game_map.get_tile(1, 0).special_traversal = True
    assert validator.find_shortest_route(game_map) == (Action.RIGHT,)
    assert has_valid_route(game_map)


# Ensures the safer route is chosen when shortest is fatal
def test_damage_aware_alternative_avoids_fatal_static_hazard():
    game_map = flat_map()
    game_map.get_tile(1, 0).hazard = HazardType.DAMAGE
    config = research_config(max_hp=5, hazard_damage=10)

    shortest = validator.find_shortest_route(game_map)
    assert shortest == (Action.RIGHT, Action.RIGHT)

    result = validator.validate_world_survival(game_map, None, config, max_wait_variants=0)

    assert result.witness is not None
    assert result.witness.route_type == "damage_aware"
    assert result.witness.actions != shortest
    assert result.witness.replay.successful
    assert result.witness.final_hp > 0
    assert result.tested_count == 2

# Tests that waiting can avoid a seeded collision
def test_wait_at_start_avoids_seeded_enemy_contact():
    game_map = flat_map()
    enemy = EnemySpec(patrol_route=((1, 0), (1, 1), (2, 1), (1, 1)), initial_patrol_index=3)
    config = research_config(enemy_enabled=True, max_hp=5, enemy_contact_damage=10)

    result = validator.validate_world_survival(game_map, enemy, config, max_wait_variants=4)

    assert result.witness is not None
    assert result.witness.route_type == "shortest"
    assert result.witness.wait_variant == "wait_start"
    assert result.witness.actions == (Action.WAIT, Action.RIGHT, Action.RIGHT)
    assert result.witness.replay.successful
    assert result.tested_count == 2

# Ensures the validator rejects without running simulator
def test_unreachable_map_skips_simulation(monkeypatch):
    game_map = flat_map()
    game_map.get_tile(1, 0).obstacle = True
    game_map.get_tile(0, 1).obstacle = True

    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("A route-free map must not be simulated")

    monkeypatch.setattr(validator, "simulate_actions", unexpected_simulation)

    result = validator.validate_world_survival(game_map, None, research_config())
    assert result.witness is None
    assert result.tested_count == 0
    assert result.failure_reason == "no_directed_route"

# Tests that reaching a goal dead should not produce living witness
def test_fatal_goal_arrival_is_not_accepted():
    game_map = flat_map()
    game_map.get_tile(2, 0).hazard = HazardType.DAMAGE

    result = validator.validate_world_survival(game_map, None, research_config(max_hp=5, hazard_damage=10), max_wait_variants=0)

    assert result.witness is None
    assert result.failure_reason == "no_living_witness_in_bounded_search"
    assert result.tested_count in (1, 2)

# Ensures that the different WAIT variations have deterministic ordering
def test_wait_variations_are_deterministic_unique_and_bounded():
    path = (Action.RIGHT, Action.RIGHT)
    variants = validator.action_variants(path, 4)
    sequences = [actions for _, actions in variants]

    assert variants == (
        ("no_wait", path),
        ("wait_start", (Action.WAIT, Action.RIGHT, Action.RIGHT)),
        ("wait_middle", (Action.RIGHT, Action.WAIT, Action.RIGHT)),
        ("wait_twice_start", (Action.WAIT, Action.WAIT, Action.RIGHT, Action.RIGHT)),
    )
    assert len(sequences) == len(set(sequences))
    assert validator.action_variants(path, 0) == (("no_wait", path),)
    assert len(validator.action_variants(path, 1)) == 2

# Ensures that when survival validation fails, it retries using same RNG object and dims
def test_survival_rejection_retries_same_rng_and_dimensions(monkeypatch):
    original_sampler = world_generator.sample_map_dimensions
    original_generator = world_generator.generate_research_candidate
    original_checker = validator.validate_world_survival

    sampled_dimensions = []
    generated_attempts = []
    checks = []

    def tracked_sampler(rng):
        selected = original_sampler(rng)
        sampled_dimensions.append(selected)
        return selected

    def tracked_generator(map_config, settings, rng, *, width, height):
        generated_attempts.append((width, height, rng))
        return original_generator(map_config, settings, rng, width=width, height=height)

    def reject_once(*args, **kwargs):
        checks.append(1)
        if len(checks) == 1:
            return validator.SurvivalCheck(witness=None, tested_count=1, failure_reason="forced_rejection")
        return original_checker(*args, **kwargs)

    monkeypatch.setattr(world_generator, "sample_map_dimensions", tracked_sampler)
    monkeypatch.setattr(world_generator, "generate_research_candidate", tracked_generator)
    monkeypatch.setattr(validator, "validate_world_survival", reject_once)

    world, witness = validator.generate_surviving_research_world(
        MapConfig(max_elevation=0),
        simple_settings(world_max_attempts=3),
        research_config(),
        root_seed=123,
    )

    assert len(sampled_dimensions) == 1
    assert len(generated_attempts) == 2
    assert len(checks) == 2
    assert generated_attempts[0][2] is generated_attempts[1][2]
    assert generated_attempts[0][:2] == generated_attempts[1][:2]
    assert generated_attempts[0][:2] == sampled_dimensions[0][:2]
    assert world.attempt_index == 2
    assert len(world.rejection_reasons) == 1
    assert "forced_rejection" in world.rejection_reasons[0]
    assert witness.replay.successful

# Ensures that the max retry limit is properly enforced
def test_survival_rejections_obey_world_attempt_cap(monkeypatch):
    def always_reject(*args, **kwargs):
        return validator.SurvivalCheck(witness=None, tested_count=1, failure_reason="forced_failure")

    monkeypatch.setattr(validator, "validate_world_survival", always_reject)

    with pytest.raises(ValueError, match="after 2 attempts") as exc:
        validator.generate_surviving_research_world(
            MapConfig(max_elevation=0),
            simple_settings(world_max_attempts=2),
            research_config(),
            root_seed=42,
        )
    assert "forced_failure" in str(exc.value)

# Tests to make sure that the same seed yields the same world and replay
def test_same_seed_reproduces_accepted_world_and_replay():
    map_config = MapConfig(max_elevation=0)
    settings = simple_settings()
    config = research_config()

    first_world, first_witness = validator.generate_surviving_research_world(map_config, settings, config, root_seed=123)
    second_world, second_witness = validator.generate_surviving_research_world(map_config, settings, config, root_seed=123)

    assert first_world == second_world
    assert first_witness == second_witness
    assert first_witness.route_type == "shortest"
    assert first_witness.wait_variant == "no_wait"
    assert first_witness.replay == simulate_actions(first_world.game_map, first_world.enemy_spec, config, first_witness.actions)
    assert first_witness.replay.successful
    assert first_witness.final_hp > 0

# Validates proper handling of invalid config wait limit
@pytest.mark.parametrize("bad_limit", [-1, 5, True, 1.5])
def test_invalid_wait_variant_limit_is_rejected(bad_limit):
    with pytest.raises(ValueError, match="max_wait_variants"):
        validator.validate_world_survival(flat_map(), None, research_config(), max_wait_variants=bad_limit)

    with pytest.raises(ValueError, match="witness_max_wait_variants"):
        world_generator.validate_world_settings(simple_settings(witness_max_wait_variants=bad_limit))
