"""Versioned enemy pursuit: BFS chase reserved for M4 robustness evaluation.

This module is deliberately pure. It owns no enemy state and imports nothing
from the enemy or environment modules, so `enemy.py` can dispatch to it without
a circular import. It answers one question: given a map, a pursuer tile and the
agent's tile, which tile does the pursuer move to this turn?

Two properties matter for the research result and are both load-bearing:

Directed search. `can_step_on_map` is not symmetric - climbing needs a
special-traversal tile, descending is free. So "the shortest legal path from
tile T to the agent" is not the same relation as "the shortest legal path from
the agent to tile T". The distance field is therefore built by breadth-first
search seeded at the agent and expanded over *reversed* edges: tile `t` is
reached from candidate predecessor `p` only when `can_step(p, t)` holds. A
forward search from the pursuer would silently route it through one-way drops
in the wrong direction.

Declared tie-breaking. Equal shortest paths are resolved by a named constant,
not by the iteration order of a dictionary somewhere else in the codebase.
`ACTION_DELTAS` happens to be ordered UP, RIGHT, LEFT, DOWN, which is *not* the
order this behavior specifies, so nothing here may order moves by iterating it
or by calling `legal_moves`. The order is part of the behavior version and is
recorded in scenario provenance.

No randomness is used anywhere in this module, by design: pursuit must not
consume draws from the environment's generator, or an evaluation episode would
perturb every subsequent seeded decision.
"""

from collections import deque

from src.environment.action_space import ACTION_DELTAS, Action
from src.environment.map import GameMap

Position = tuple[int, int]

# Identifier for this behavior, stored on EnemySpec and in scenario provenance.
PURSUIT_BEHAVIOR_VERSION = "pursuit_bfs_v1"

# Equal shortest paths resolve to the first direction listed here. Part of the
# behavior contract: changing this order requires a new behavior version.
TIE_BREAK_ORDER: tuple[Action, ...] = (
    Action.UP,
    Action.RIGHT,
    Action.DOWN,
    Action.LEFT,
)

# Human-readable form of the rule above, written into scenario provenance so a
# frozen scenario records how its ties were resolved.
TIE_BREAK_RULE = "UP,RIGHT,DOWN,LEFT"


def distance_field_to(
    game_map: GameMap,
    target: Position,
    can_step,
) -> dict[Position, int]:
    """Legal step distance from each reachable tile *to* `target`.

    Breadth-first search seeded at `target`, expanded over reversed edges, so
    the result answers "how many legal steps does it take to get from this tile
    to the target", which is the relation a pursuer needs. Tiles that cannot
    legally reach the target are absent from the mapping rather than present
    with an infinite value.

    Args:
        game_map: The map supplying bounds and tile data.
        target: The tile being moved toward, normally the agent's tile.
        can_step: Predicate `can_step(src, dst)` giving the shared directed
            movement rule. The same predicate the agent moves under.

    Returns:
        dict[Position, int]: Reachable tile -> step count to `target`.
    """
    distances: dict[Position, int] = {target: 0}
    queue: deque[Position] = deque([target])

    while queue:
        tile = queue.popleft()
        next_distance = distances[tile] + 1
        x, y = tile

        # Candidate predecessors are the four orthogonal neighbours. Iterated in
        # the declared order purely so the search itself is reproducible;
        # distances do not depend on it. `can_step` rejects out-of-bounds and
        # illegal candidates, so no separate bounds check is needed here.
        for action in TIE_BREAK_ORDER:
            dx, dy = ACTION_DELTAS[action]
            predecessor = (x + dx, y + dy)

            if predecessor in distances:
                continue

            # Directed edge: the predecessor must be able to step *into* `tile`.
            if not can_step(predecessor, tile):
                continue

            distances[predecessor] = next_distance
            queue.append(predecessor)

    return distances


def distance_field_from(
    game_map: GameMap,
    origin: Position,
    can_step,
) -> dict[Position, int]:
    """Legal step distance from `origin` *to* each reachable tile.

    The mirror of `distance_field_to`, obtained by running the same search with
    the movement predicate flipped: expanding over `can_step(tile, candidate)`
    walks forward edges away from the origin instead of backward edges into the
    target. Under the directed movement rule the two fields genuinely differ, so
    which one a caller wants is a real choice rather than a detail.

    Not used by the pursuit behavior itself. It exists for provenance
    diagnostics, which need the enemy's own reachability outward from its spawn.

    Args:
        game_map: The map supplying bounds and tile data.
        origin: The tile being measured from, normally an enemy spawn.
        can_step: Predicate `can_step(src, dst)` giving the shared directed
            movement rule.

    Returns:
        dict[Position, int]: Reachable tile -> step count from `origin`.
    """
    return distance_field_to(
        game_map,
        origin,
        lambda src, dst: can_step(dst, src),
    )


def pursuit_target(
    game_map: GameMap,
    enemy_position: Position,
    agent_position: Position,
    can_step,
) -> Position:
    """Tile the pursuer moves to this turn.

    Takes one legal orthogonal step along a shortest legal path to the agent's
    current tile, recomputed from scratch every turn. Returning
    `enemy_position` means WAIT.

    The pursuer waits in exactly two situations:

    Unreachable agent. No legal path exists under the directed movement rule,
    for instance when the agent is behind a one-way climb. The pursuer holds
    position rather than drifting.

    Already co-located. The agent moved onto the pursuer's tile earlier in the
    same step. The distance to the agent is then zero and there is no closer
    neighbour. Waiting is the correct answer rather than a fallback: contact is
    resolved after enemy movement on same-cell occupancy, so stepping away here
    would let the pursuer dodge damage it should have dealt.

    Args:
        game_map: The map supplying bounds and tile data.
        enemy_position: The pursuer's current tile.
        agent_position: The agent's tile *after* its own move this step.
        can_step: Predicate `can_step(src, dst)` giving the shared directed
            movement rule.

    Returns:
        Position: The tile to occupy, which may be `enemy_position` for WAIT.
    """
    distances = distance_field_to(game_map, agent_position, can_step)
    current_distance = distances.get(enemy_position)

    # Unreachable agent, or the agent is standing on the pursuer already.
    if current_distance is None or current_distance == 0:
        return enemy_position

    best_distance: int | None = None
    best_tile = enemy_position

    # Strict improvement plus declared ordering: the first direction achieving
    # the minimum wins, which is what resolves equal shortest paths.
    for action in TIE_BREAK_ORDER:
        dx, dy = ACTION_DELTAS[action]
        candidate = (enemy_position[0] + dx, enemy_position[1] + dy)

        if not can_step(enemy_position, candidate):
            continue

        candidate_distance = distances.get(candidate)

        if candidate_distance is None:
            continue

        if best_distance is None or candidate_distance < best_distance:
            best_distance = candidate_distance
            best_tile = candidate

    # A finite, non-zero distance guarantees a strictly closer legal neighbour,
    # because that is how this tile entered the distance field. The guard keeps
    # the function total if `can_step` ever disagrees with the field it built.
    if best_distance is None or best_distance >= current_distance:
        return enemy_position

    return best_tile
