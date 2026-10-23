from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Literal
from collections import deque

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.map_generation_config import MapGenerationConfig
from src.environment.action_space import Action
from src.environment.enemy import Position
from src.environment.map import GameMap
from src.environment.map_generator import find_shortest_route
from src.environment.patrol_generator import EnemySpec
from src.environment.research_world_generator import ResearchWorldCandidate, generate_research_world_candidate
from src.environment.transition_rules import fall_damage_on_map, hazard_damage_at
from src.environment.traversal import legal_moves
from src.environment.witness_simulator import ActionReplay, simulate_actions


RouteType = Literal["shortest", "damage_aware"]


# A living route through one world
@dataclass(frozen=True)
class SurvivalWitness:
    actions: tuple[Action, ...] # Actions taken on route
    route_type: RouteType # shortest vs damage_aware path
    wait_variant: str
    replay: ActionReplay

    @property
    def final_hp(self) -> float:
        return self.replay.final_hp

    @property
    def steps(self) -> int:
        return len(self.replay.frames)


# Result of the route-validation
@dataclass(frozen=True)
class SurvivalCheck:
    witness: SurvivalWitness | None
    tested_count: int
    failure_reason: str | None


# BFS algorithm to find the shortest legal route (Extremely important for validation)
def find_shortest_route(game_map: GameMap) -> tuple[Action, ...] | None:
    width = game_map.width
    height = game_map.height
    start = game_map.start
    goal = game_map.goal

    # --- Structure validity checks from old 'has_valid_route' --- #
    if (
        width <= 0
        or height <= 0
        or len(game_map.tiles) != height
        or any(len(row) != width for row in game_map.tiles)
    ):
        return None

    for x, y in (start, goal):
        if not (0 <= x < width and 0 <= y < height):
            return None

        if game_map.get_tile(x, y).obstacle:
            return None

    if start == goal:
        return None

    # --- BFS (FIFO queue) --- #
    # Creates queue with start position
    queue = deque([start])
    visited = {start} # Set of visited items for fast lookup

    parents: dict[
        tuple[int, int], # Tile
        tuple[tuple[int, int], Action] # Tile's parent, action
    ] = {}

    while queue:
        # Save the first item in queue and remove
        current = queue.popleft()

        # With FIFO, once we reach goal, the shortest path has been found
        if current == goal:
            actions = []

            # Reverse through the shortest path saving actions taken to actions
            while current != start:
                previous, action = parents[current]
                actions.append(action)
                current = previous

            # Reverse the list so it is in correct order and return shortest path
            actions.reverse()
            return tuple(actions)

        # Calls helper function that iterates all neighbors for legal moves
        for action, neighbor in legal_moves(game_map, current):
            # Continue if already visited
            if neighbor in visited:
                continue

            # Otherwise add the neighbor to visited tiles, add it to parents dict, and add it to the queue
            visited.add(neighbor)
            parents[neighbor] = (current, action)
            queue.append(neighbor)

    return None

# Finds a damage aware route using Dijkstra's alg
def find_damage_aware_route(game_map: GameMap, config: EnvironmentConfig,) -> tuple[Action, ...] | None:
    start = game_map.start
    goal = game_map.goal

    if start == goal:
        return None

    # Unique insertion order makes equal-cost choices consistent with priority queue
    order = count()
    queue = [(0.0, next(order), start)] # (cost, order, position)

    best_cost = {start: 0.0} # Stores cheapest known cost to reach each position
    parents: dict[Position, tuple[Position, Action]] = {} # Dictionary for tracking parent and what action was taken to reach position

    while queue:
        # Removes entry with lowest priority
        cost, _, current = heappop(queue)

        # If the cost is greater than what is already saved, continue with next
        if cost > best_cost[current]:
            continue

        # Once goal is reached, path with least cost has been found
        if current == goal:
            actions = []

            # Reverse through path saving actions to actions
            while current != start:
                previous, action = parents[current]
                actions.append(action)
                current = previous

            # Reverse order of actions and return path taken
            actions.reverse()
            return tuple(actions)

        # Determines damage for hazards and falling
        for action, destination in legal_moves(game_map, current):
            fall = fall_damage_on_map(game_map, config, current, destination)
            hazard = hazard_damage_at(game_map, config, destination)

            # Fixed, reward-independent edge cost. Adds 1 to ensure that progress is still important.
            edge_cost = (1.0 + max(0.0, fall) + max(0.0, hazard))
            new_cost = cost + edge_cost

            # Update current destination's best_cost with lower cost then update parents with the action taken
            if new_cost < best_cost.get(destination, inf):
                best_cost[destination] = new_cost
                parents[destination] = (current, action)

                # Adds destination and cost
                heappush(queue, (new_cost, next(order), destination))

    return None


# Return the base path and bounded wait variations for possible enemy interaction
def action_variants(path: tuple[Action, ...], max_wait_variants: int) -> tuple[tuple[str, tuple[Action, ...]], ...]:
    if (type(max_wait_variants) is not int or not 0 <= max_wait_variants <= 4):
        raise ValueError("max_wait_variants must be an integer from 0 to 4")

    if not path:
        return ()

    # Wait variations
    proposed = (
        ("wait_start", (Action.WAIT,) + path), # WAIT -> path
        ("wait_middle", path[:len(path) // 2] + (Action.WAIT,) + path[len(path) // 2:]) # Half path -> WAIT -> Rest path
        ("wait_before_goal",path[:-1] + (Action.WAIT,) + path[-1:]), # path -> WAIT -> goal
        ("wait_twice_start", (Action.WAIT, Action.WAIT) + path), # WAIT -> WAIT -> path
    )

    # Original path is included first
    variants = [("no_wait", path)] # path
    seen = {path}


    for name, actions in proposed:
        # Breaks if len of the variants is > than the max_wait_variants
        if len(variants) - 1 >= max_wait_variants: # - 1 since no_wait does not count towards wait patterns
            break

        # Continue if already seen action
        if actions in seen:
            continue

        # Add action to seen and append (name, actions) to variants
        seen.add(actions)
        variants.append((name, actions))

    # Returns tuple of [name, Actions] variations
    return tuple(variants)

# Main validation function for determining if world is survivable
def validate_world_survival(game_map: GameMap, enemy_spec: EnemySpec | None, config: EnvironmentConfig, *, max_wait_variants: int = 4,) -> SurvivalCheck:
    if config.episode_limit_mode != "research":
        raise ValueError("Survival validation requires research mode")

    if config.enemy_enabled != (enemy_spec is not None):
        raise ValueError("Enemy configuration and EnemySpec disagree")

    if (type(max_wait_variants) is not int or not 0 <= max_wait_variants <= 4):
        raise ValueError("max_wait_variants must be an integer from 0 to 4")

    # BFS search for shortest route
    shortest = find_shortest_route(game_map)

    # Return immediately if there is no legal movement to goal
    if shortest is None:
        return SurvivalCheck(witness=None, tested_count=0, failure_reason="no_directed_route")

    tested_count = 0

    # Nested function that evaluates a movement path
    def try_route(path: tuple[Action, ...], kind: RouteType) -> SurvivalWitness | None:
        # nonlocal variable for up-to-date test count
        nonlocal tested_count

        # Generate and simulate each variation
        for variant_name, actions in action_variants(path, max_wait_variants): # generates variants
            replay = simulate_actions(game_map, enemy_spec, config, actions) # Simulates actions using witness simulator
            tested_count += 1 # Adds 1 to tested count variable

            # Route is accepted only after complete, living arrival
            if replay.successful and replay.final_hp > 0:
                return SurvivalWitness(actions=actions, route_type=kind, wait_variant=variant_name, replay=replay)

        return None

    # Try the shortest route first
    witness = try_route(shortest, "shortest")

    # if any path variations can be completed alive (living witness) return SurvivalCheck 
    if witness is not None:
        return SurvivalCheck(witness=witness, tested_count=tested_count, failure_reason=None)

    # Considers different damage-aware route
    alternative = find_damage_aware_route(game_map, config)

    if alternative is not None and alternative != shortest:
        # Try the damage-aware route instead
        witness = try_route(alternative, "damage_aware")

        # Living witness returns immediately
        if witness is not None:
            return SurvivalCheck(witness=witness, tested_count=tested_count, failure_reason=None)

    # If world cannot be completed with living witness, return SurvivalCheck with failure reason
    return SurvivalCheck(witness=None, tested_count=tested_count, failure_reason="no_living_witness_in_bounded_search",)

# Generates a research world that requires that it passes survival checks before returning (ResearchWorldCandidate is the world and SurvivalWitness is demonstrated successful route)
def generate_surviving_research_world(map_config: MapConfig, settings: MapGenerationConfig, environment_config: EnvironmentConfig, root_seed: int) -> tuple[ResearchWorldCandidate, SurvivalWitness]:
    if environment_config.episode_limit_mode != "research":
        raise ValueError("Research episode limits are required")

    if environment_config.enemy_enabled != settings.enemy_required:
        raise ValueError("Enemy settings must agree")

    accepted_witness: SurvivalWitness | None = None

    # Nested function for checking worlds (Returns string for failure, None for success)
    def check_candidate(world: ResearchWorldCandidate) -> str | None:
        nonlocal accepted_witness

        result = validate_world_survival(world.game_map, world.enemy_spec, environment_config, max_wait_variants=settings.witness_max_wait_variants)

        # No living witness = failure
        if result.witness is None:
            return (f"survival failed: {result.failure_reason}! {result.tested_count} sequences tested...")

        # Sets accepted_witness to the result and returns None
        accepted_witness = result.witness
        return None

    # Generates worlda and then checks candidate using nested helper
    world = generate_research_world_candidate(map_config, settings, root_seed, accept_candidate=check_candidate)

    if accepted_witness is None:
        raise RuntimeError("World accepted without a living witness")

    return world, accepted_witness
