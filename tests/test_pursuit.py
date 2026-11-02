"""Tests for the versioned BFS pursuit behavior.

Each test here is written to fail under one specific wrong implementation, so a
failure names the mistake rather than just the symptom:

* `test_pursuit_uses_directed_search_toward_the_agent` fails if the distance
  field is built over forward edges from the agent instead of reversed ones.
* `test_equal_shortest_paths_resolve_down_before_left` fails if moves are
  ordered by `ACTION_DELTAS` or `legal_moves`, whose order is UP, RIGHT, LEFT,
  DOWN rather than the order this behavior declares.
"""

import pytest

from src.environment.action_space import ACTION_DELTAS, Action
from src.environment.map import GameMap, Tile
from src.environment.pursuit import (
    PURSUIT_BEHAVIOR_VERSION,
    TIE_BREAK_ORDER,
    TIE_BREAK_RULE,
    distance_field_to,
    pursuit_target,
)
from src.environment.traversal import can_step_on_map


def build_map(
    elevations: list[list[int]],
    *,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] | None = None,
    obstacles: tuple[tuple[int, int], ...] = (),
    special: tuple[tuple[int, int], ...] = (),
) -> GameMap:
    """A map described by its elevation grid, one inner list per row.

    Rows are y, columns are x, matching `GameMap.get_tile(x, y)` and the
    convention that UP decreases y.
    """
    height = len(elevations)
    width = len(elevations[0])

    tiles = [
        [
            Tile(
                elevation=elevations[y][x],
                obstacle=(x, y) in obstacles,
                special_traversal=(x, y) in special,
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


def stepper(game_map: GameMap):
    """The shared directed movement predicate, bound to one map."""
    return lambda src, dst: can_step_on_map(game_map, src, dst)


# --- The declared behavior contract --- #


def test_behavior_version_and_tie_rule_are_stable_identifiers():
    """These strings land in frozen scenarios, so changing them is a schema event."""
    assert PURSUIT_BEHAVIOR_VERSION == "pursuit_bfs_v1"
    assert TIE_BREAK_ORDER == (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)
    assert TIE_BREAK_RULE == "UP,RIGHT,DOWN,LEFT"


def test_tie_break_order_is_not_action_deltas_order():
    """Guards the trap directly: the dict this module must not order moves by."""
    assert TIE_BREAK_ORDER != tuple(
        action for action in ACTION_DELTAS if action is not Action.WAIT
    )


# --- Shortest-path movement --- #


def test_pursuer_steps_one_tile_toward_the_agent():
    """One legal orthogonal step per turn, never a teleport along the path."""
    game_map = build_map([[0] * 5], start=(0, 0), goal=(4, 0))

    target = pursuit_target(game_map, (4, 0), (0, 0), stepper(game_map))

    assert target == (3, 0)


def test_adjacent_pursuer_steps_onto_the_agent_tile():
    """Same-cell contact has to stay reachable, or pursuit can never deal damage."""
    game_map = build_map([[0] * 3], start=(0, 0), goal=(2, 0))

    target = pursuit_target(game_map, (1, 0), (0, 0), stepper(game_map))

    assert target == (0, 0)


def test_pursuer_routes_around_an_obstacle_wall():
    """The path is a legal map path, not a straight line toward the agent."""
    # A wall down column x=1 with a gap at the bottom row.
    game_map = build_map(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        start=(0, 0),
        goal=(2, 2),
        obstacles=((1, 0), (1, 1)),
    )

    # Agent at (0, 0); pursuer at (2, 0) must go down to reach the gap, since
    # stepping LEFT into the wall is illegal.
    target = pursuit_target(game_map, (2, 0), (0, 0), stepper(game_map))

    assert target == (2, 1)


# --- Directed search: the reverse-BFS requirement --- #


def test_distance_field_measures_steps_toward_the_target_not_away_from_it():
    """A one-way drop makes the two relations disagree, which is the whole point.

    Elevation rises left to right, so (1, 0) -> (2, 0) is an illegal climb while
    (2, 0) -> (1, 0) is a legal descent. Tiles on the high side can therefore
    reach the agent, but the agent cannot reach them.
    """
    game_map = build_map([[0, 0, 1, 1]], start=(0, 0), goal=(3, 0))

    field = distance_field_to(game_map, (0, 0), stepper(game_map))

    # Reachable *to* the agent, over the legal descent.
    assert field[(3, 0)] == 3
    assert field[(2, 0)] == 2
    assert field[(1, 0)] == 1
    assert field[(0, 0)] == 0


def test_pursuit_uses_directed_search_toward_the_agent():
    """Fails if the field is built over forward edges from the agent.

    Under that mutation the illegal climb at (1, 0) -> (2, 0) truncates the
    search, the pursuer's tile is absent from the field, and it waits forever.
    """
    game_map = build_map([[0, 0, 1, 1]], start=(0, 0), goal=(3, 0))

    target = pursuit_target(game_map, (3, 0), (0, 0), stepper(game_map))

    assert target == (2, 0)


def test_pursuer_respects_special_traversal_climbs():
    """Climbing is legal only onto a special-traversal tile, for enemy and agent alike."""
    without_ramp = build_map([[0, 1, 1]], start=(0, 0), goal=(2, 0))
    with_ramp = build_map([[0, 1, 1]], start=(0, 0), goal=(2, 0), special=((1, 0),))

    # Pursuer in the low tile, agent up the step.
    assert pursuit_target(without_ramp, (0, 0), (2, 0), stepper(without_ramp)) == (0, 0)
    assert pursuit_target(with_ramp, (0, 0), (2, 0), stepper(with_ramp)) == (1, 0)


# --- Declared tie-breaking --- #


def test_equal_shortest_paths_resolve_down_before_left():
    """Two neighbours tie at distance 1: DOWN wins, LEFT loses.

    Fails if moves are ordered by `ACTION_DELTAS`, which lists LEFT before DOWN
    and would pick (0, 1).
    """
    game_map = build_map([[0] * 3 for _ in range(3)], start=(0, 0), goal=(2, 2))

    # Pursuer at (1, 1), agent at (0, 2). DOWN -> (1, 2) and LEFT -> (0, 1) are
    # both one step from the agent; UP and RIGHT are three.
    target = pursuit_target(game_map, (1, 1), (0, 2), stepper(game_map))

    assert target == (1, 2)


def test_up_outranks_right_on_a_tie():
    """The first two entries of the declared order, checked independently."""
    game_map = build_map([[0] * 3 for _ in range(3)], start=(0, 0), goal=(2, 2))

    # Pursuer at (1, 1), agent at (2, 0): UP -> (1, 0) and RIGHT -> (2, 1) both
    # sit one step from the agent.
    target = pursuit_target(game_map, (1, 1), (2, 0), stepper(game_map))

    assert target == (1, 0)


def test_pursuit_is_deterministic_across_repeated_calls():
    """Recomputed from scratch each turn, so repeated calls must agree exactly."""
    game_map = build_map([[0] * 4 for _ in range(4)], start=(0, 0), goal=(3, 3))

    targets = {
        pursuit_target(game_map, (3, 3), (1, 2), stepper(game_map))
        for _ in range(16)
    }

    assert len(targets) == 1


# --- Waiting --- #


def test_pursuer_waits_when_the_agent_is_unreachable_behind_a_wall():
    """No legal path means hold position, not drift."""
    game_map = build_map(
        [[0, 0, 0]],
        start=(0, 0),
        goal=(2, 0),
        obstacles=((1, 0),),
    )

    target = pursuit_target(game_map, (2, 0), (0, 0), stepper(game_map))

    assert target == (2, 0)


def test_pursuer_waits_when_trapped_below_a_climb_it_cannot_make():
    """Unreachability can come from the directed rule, not only from obstacles."""
    game_map = build_map([[0, 1, 0]], start=(0, 0), goal=(2, 0))

    # Pursuer at (2, 0) would have to climb into (1, 0) to reach the agent.
    target = pursuit_target(game_map, (2, 0), (0, 0), stepper(game_map))

    assert target == (2, 0)


def test_pursuer_waits_when_the_agent_moved_onto_its_tile():
    """Co-located at the enemy phase: waiting is what preserves same-cell contact.

    Stepping away here would let the pursuer dodge damage it should have dealt,
    because contact is resolved after enemy movement.
    """
    game_map = build_map([[0] * 3 for _ in range(3)], start=(0, 0), goal=(2, 2))

    target = pursuit_target(game_map, (1, 1), (1, 1), stepper(game_map))

    assert target == (1, 1)


@pytest.mark.parametrize("enemy_position", [(0, 0), (2, 0), (1, 1)])
def test_pursuit_never_returns_an_illegal_destination(enemy_position):
    """Whatever the topology, the returned tile is WAIT or one legal step."""
    game_map = build_map(
        [[0, 1, 0], [0, 0, 2]],
        start=(0, 0),
        goal=(2, 1),
        obstacles=((2, 0),),
    )
    can_step = stepper(game_map)

    target = pursuit_target(game_map, enemy_position, (0, 1), can_step)

    assert target == enemy_position or can_step(enemy_position, target)
