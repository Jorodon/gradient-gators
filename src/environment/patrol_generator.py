from dataclasses import dataclass

import numpy as np

from src.environment.enemy import Position, validate_patrol_route
from src.environment.map import GameMap
from src.environment.traversal import can_step_on_map

# Defines the 4 movement directions
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

@dataclass(frozen = True)
class EnemySpec:
    patrol_route: tuple[Position, ...]
    initial_patrol_index: int
    
    @property
    def initial_position(self) -> Position:
        return self.patrol_route[self.initial_patrol_index]

# Generate a seeded patrol route that is also reverseable
def generate_enemy_spec(
        game_map: GameMap,
        rng: np.random.Generator,
        *,
        spawn: Position | None = None,
        max_outward_steps: int = 4,
        max_spawn_attempts: int = 32,
) -> EnemySpec:
    # --- Helper functions --- #
    # Depth first search algoritihm with randomized neighbor odering
    def search(path: list[Position], target_steps: int) -> list[Position] | None:
        # If we've already reached desired lengh, return path
        if len(path) == target_steps + 1:
            return path

        # Retrieves the last position of the current path
        x, y = path[-1]

        # Looks at all of tile's neighbors and saves allowed tiles not currently in the path
        neighbors = [
            (x + dx, y + dy) for dx, dy in DIRECTIONS
            if (x + dx, y + dy) in allowed_set
            and (x + dx, y + dy) not in path
        ]

        # Examines elible neighbors in a seeded order
        for i in rng.permutation(len(neighbors)):
            neighbor = neighbors[int(i)]

            # If movement is not valid (forward or backwards), try next neighbor
            if not can_step_on_map(game_map, path[-1], neighbor):
                continue
            if not can_step_on_map(game_map, neighbor, path[-1]):
                continue

            # Recursively call result with updated path
            result = search(path + [neighbor], target_steps)

            if result is not None:
                return result

        return None

    # --- Route Generator Function Logic --- #
    if type(max_outward_steps) is not int or max_outward_steps < 2:
        raise ValueError("max_outward_steps must be an integer >= 2")

    if type(max_spawn_attempts) is not int or max_spawn_attempts < 1:
        raise ValueError("max_spawn_attempts must be an integer >= 1")

    # Create array of safe, passable, and non-critical (goal, agent spawn) tiles
    allowed = [
        (x, y) for y in range(game_map.height) for x in range(game_map.width)
        if (x, y) not in (game_map.start, game_map.goal)
        and not game_map.get_tile(x, y).obstacle
        and game_map.get_tile(x, y).hazard is None
    ]

    # Rejects a map that cannot provide three distinct, eligible positions and then converts positions into a set for fast checks
    if len(allowed) < 3:
        raise ValueError("Map has fewer than 3 eligible patrol tiles")

    allowed_set = set(allowed)

    # If spawn is provided, checks that provided spawn is eligible and makes it the only candidate
    if spawn is not None:
        if spawn not in allowed_set:
            raise ValueError("Invalid enemy patrol spawn")
        candidate_spawns = [spawn]
    # If spawn is not provided, produces a randomized ordering of indicies and stores the resulting seeded candidate order
    else:
        # Produces a randomized ordering of allowed indicies limited by max_spawn_attempts
        candidate_spawns = [allowed[int(i)] for i in rng.permutation(len(allowed))[:max_spawn_attempts]]


    # Limits the requested outward lenght to the number of actual eligible positions
    upper_steps = min(max_outward_steps, len(allowed) - 1)

    # For each spawn, samples an outward route lemght between two and maximum upper steps (inclusive)
    for candidate_spawn in candidate_spawns:
        desired_steps = int(rng.integers(2, upper_steps + 1))

        # Tries calling recursive search helper with sampled length initially then shorter lenghts down to two moves
        for steps in range(desired_steps, 1, -1):
            outward = search([candidate_spawn], steps)

            # Tries next spawn candidate if search returns None
            if outward is None:
                continue

            # Creates complete route (initial path plus middle tiles) so path can loop (e.g [A, B, C, D] would create route = [A, B, C, D, C, B])
            route = tuple(outward + outward[-2:0:-1])

            # Validates route, validate_patrol_route takes in lambda helper function that returns bool off of two positions
            validate_patrol_route(route, game_map, lambda src, dst: can_step_on_map(game_map, src, dst))

            # Randomizes the initial route starting position
            initial_patrol_index = int(rng.integers(0, len(route)))

            # Returns EnemySpec object with valid route and enemy spawn
            return EnemySpec(patrol_route=route, initial_patrol_index=initial_patrol_index)

    # If all candidates are exhausted, raise ValueError
    raise ValueError("Unable to generate a moving enemy patrol")


        