"""Runtime enemy entity: state, spawning and patrol bookkeeping.

Deliberately separate from GameMap. GameMap and its Tiles describe the static
world loaded from JSON; the enemy moves, so its position belongs to the episode
rather than to the map. Writing it into tile data would mean mutating the map on
every step and reloading it on every reset.

This module owns enemy state only. It does not know the traversal rules - the
caller passes in a can_step(src, dst) predicate, so the enemy obeys exactly the
same movement rules as the agent without this module importing GatorEnv.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from src.environment.map import GameMap
from src.environment.pursuit import PURSUIT_BEHAVIOR_VERSION, pursuit_target

Position = tuple[int, int]
CanStep = Callable[[Position, Position], bool]

# Identifier for the original patrol behavior. Written explicitly onto every
# EnemySpec rather than left implicit, so a scenario file states how its enemy
# moves instead of leaving a reader to infer it from the absence of a field.
PATROL_BEHAVIOR_VERSION = "patrol_v1"

EnemyBehavior = Literal["patrol_v1", "pursuit_bfs_v1"]

# Every behavior a frozen scenario may name.
ENEMY_BEHAVIORS: tuple[str, ...] = (
    PATROL_BEHAVIOR_VERSION,
    PURSUIT_BEHAVIOR_VERSION,
)


@dataclass
class Enemy:
    """One enemy's runtime state.

    Attributes:
        position: Current (x, y) grid position.
        patrol_route: Ordered waypoints the enemy cycles through. An empty or
            single-entry route means the enemy is stationary.
        patrol_index: Index in patrol_route of the enemy's current waypoint.
        contact_damage: HP removed from the agent when contact occurs.
    """

    position: Position
    patrol_route: list[Position] = field(default_factory=list)
    patrol_index: int = 0
    contact_damage: float = 0.0

    @property
    def is_stationary(self) -> bool:
        """True when the route cannot produce movement."""
        return len(self.patrol_route) <= 1

    def next_patrol_index(self) -> int:
        """Index of the waypoint the enemy would move to next.

        The route is treated as a cycle, so the last waypoint leads back to the
        first. A back-and-forth patrol is expressed as a cycle that doubles
        back, e.g. [A, B, C, B].
        """
        if self.is_stationary:
            return self.patrol_index
        return (self.patrol_index + 1) % len(self.patrol_route)

    def next_waypoint(self) -> Position | None:
        """The position the enemy would move to next, or None if stationary."""
        if self.is_stationary:
            return None
        return self.patrol_route[self.next_patrol_index()]

    def advance_to_next_waypoint(self) -> None:
        """Commit the move to the next waypoint."""
        if self.is_stationary:
            return
        self.patrol_index = self.next_patrol_index()
        self.position = self.patrol_route[self.patrol_index]

# Helper function to advance one legal patrol step
def advance_enemy(enemy: Enemy | None, can_step: CanStep) -> None:
    if enemy is None or enemy.is_stationary:
        return

    target = enemy.next_waypoint()

    if target is None:
        return

    if target == enemy.position or can_step(enemy.position, target):
        enemy.advance_to_next_waypoint()


def move_enemy(
    enemy: Enemy | None,
    can_step: CanStep,
    *,
    behavior: EnemyBehavior = PATROL_BEHAVIOR_VERSION,
    game_map: GameMap | None = None,
    agent_position: Position | None = None,
) -> None:
    """Advance the enemy one turn under the named behavior.

    The single movement seam for both GatorEnv and the witness simulator, so
    the two cannot drift apart. Patrol keeps its original code path untouched;
    pursuit is a separate branch that recomputes its route every turn.

    Args:
        enemy: The enemy to move, or None when the scenario has no enemy.
        can_step: Predicate `can_step(src, dst)` giving the shared directed
            movement rule.
        behavior: Which movement rule to apply. Carried on the scenario's
            EnemySpec, so a frozen scenario always moves the same way.
        game_map: Required for pursuit, which searches the map. Unused by patrol.
        agent_position: The agent's tile *after* its own move this step.
            Required for pursuit, unused by patrol.

    Raises:
        ValueError: If `behavior` is unknown, or if pursuit is requested
            without the map and agent position it needs. Failing loudly here
            matters more than it looks: silently falling back to patrol would
            make an evaluation run report pursuit results from patrol episodes.
    """
    if enemy is None:
        return

    if behavior == PATROL_BEHAVIOR_VERSION:
        advance_enemy(enemy, can_step)
        return

    if behavior == PURSUIT_BEHAVIOR_VERSION:
        if game_map is None or agent_position is None:
            raise ValueError(
                "Pursuit requires game_map and agent_position"
            )

        # patrol_index is deliberately left alone. The route still defines the
        # enemy's saved spawn and takes part in scenario identity, but a
        # pursuer does not walk it, so the index stays frozen at its initial
        # value for the whole episode.
        enemy.position = pursuit_target(
            game_map,
            enemy.position,
            agent_position,
            can_step,
        )
        return

    raise ValueError(f"Unknown enemy behavior: {behavior!r}")


def in_bounds(position: Position, game_map: GameMap) -> bool:
    """Whether a position lies inside the map."""
    x, y = position
    return 0 <= x < game_map.width and 0 <= y < game_map.height


def is_valid_spawn(
    position: Position,
    game_map: GameMap,
    *,
    forbidden: Sequence[Position] = (),
) -> bool:
        # Returns bool based on results of validate_spawn
        try:
            validate_spawn(position, game_map, forbidden=forbidden)
            return True
        except ValueError:
            return False



def validate_spawn(
    position: Position,
    game_map: GameMap,
    *,
    forbidden: Sequence[Position] = (),
) -> None:
    """Raise ValueError if the spawn position is not usable."""
    if not in_bounds(position, game_map):
        raise ValueError(
            f"Enemy spawn {position} is outside the {game_map.width}x{game_map.height} map."
        )
    if game_map.get_tile(position[0], position[1]).obstacle:
        raise ValueError(f"Enemy spawn {position} is an obstacle tile.")
    if position in tuple(forbidden):
        raise ValueError(f"Enemy spawn {position} is a forbidden tile (agent start or goal).")


def default_spawn(
    game_map: GameMap,
    *,
    forbidden: Sequence[Position] = (),
) -> Position:
    """Pick a spawn deterministically when the caller gives none.

    Scans row-major and takes the first usable tile, so the same map always
    produces the same spawn - no RNG, which keeps episodes reproducible.
    """
    for y in range(game_map.height):
        for x in range(game_map.width):
            if is_valid_spawn((x, y), game_map, forbidden=forbidden):
                return (x, y)
    raise ValueError("No valid enemy spawn position exists on this map.")


def validate_patrol_route(
    route: Sequence[Position],
    game_map: GameMap,
    can_step: CanStep,
) -> None:
    """Raise ValueError if the enemy could not actually walk this route.

    Checked up front rather than during the episode, because a route the enemy
    cannot complete would otherwise show up as an enemy that silently stops
    somewhere partway round and never recovers.

    Every waypoint must be in bounds and not an obstacle, consecutive waypoints
    must be the same tile or orthogonally adjacent, and each step must satisfy
    `can_step` in the direction travelled - elevation rules are directional, so
    a legal step one way is not necessarily legal back.
    """
    if not route:
        return

    for waypoint in route:
        if not in_bounds(waypoint, game_map):
            raise ValueError(f"Patrol waypoint {waypoint} is outside the map.")
        if game_map.get_tile(waypoint[0], waypoint[1]).obstacle:
            raise ValueError(f"Patrol waypoint {waypoint} is an obstacle tile.")

    if len(route) == 1:
        return

    # The route is a cycle, so the final waypoint must lead back to the first.
    for index in range(len(route)):
        src = route[index]
        dst = route[(index + 1) % len(route)]

        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) > 1:
            raise ValueError(
                f"Patrol waypoints {src} and {dst} are not adjacent; "
                "a route must be a sequence of single orthogonal steps, and "
                "the last waypoint must lead back to the first. For a "
                "back-and-forth patrol, write the return leg out: "
                "[A, B, C] becomes [A, B, C, B]."
            )
        if src != dst and not can_step(src, dst):
            raise ValueError(
                f"Patrol step {src} -> {dst} is not a legal move under the "
                "map's traversal rules."
            )


def create_enemy(
    game_map: GameMap,
    *,
    position: Position | None = None,
    patrol_route: Sequence[Position] | None = None,
    patrol_index: int | None = None,
    contact_damage: float = 0.0,
    can_step: CanStep | None = None,
    forbidden: Sequence[Position] = (),
) -> Enemy:
    """Build an enemy for this map, validating spawn and route.

    When `patrol_route` is given, the enemy starts on its first waypoint unless
    `position` says otherwise, in which case the position must appear in the
    route. With no route the enemy is stationary.
    """
    if contact_damage < 0:
        raise ValueError("contact_damage must be a positive magnitude.")

    route = [tuple(waypoint) for waypoint in (patrol_route or [])]

    # Ensures that if a route is provided can_step must not be None then calls validate_patrol_route
    if route:
        if can_step is None:
            raise ValueError("can_step is required when patrol_route is provided.")
        
        validate_patrol_route(route, game_map, can_step)

    if patrol_index is not None:
        if (type(patrol_index) is not int or not route or not (0 <= patrol_index < len(route))):
            raise ValueError("Invalid patrol_index for patrol route")
        
        spawn = route[patrol_index]

        if position is not None and tuple(position) != spawn:
            raise ValueError("Position does not match patrol_index")

    elif position is None:
        spawn = route[0] if route else default_spawn(game_map, forbidden=forbidden)

    else:
        spawn = tuple(position)


    validate_spawn(spawn, game_map, forbidden=forbidden)

    if route and spawn not in route:
        raise ValueError(
            f"Enemy spawn {spawn} is not on its patrol route {route}."
        )

    res_index = (patrol_index if patrol_index is not None 
                 else route.index(spawn) if route
                 else 0
                 )

    return Enemy(
        position=spawn,
        patrol_route=route,
        patrol_index=res_index,
        contact_damage=float(contact_damage),
    )
