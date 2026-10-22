from dataclasses import dataclass
from collections.abc import Sequence

from src.configs.environment_config import EnvironmentConfig
from src.environment.action_space import Action, ACTION_DELTAS, ACTION_SPACE
from src.environment.enemy import validate_patrol_route, Position, Enemy, advance_enemy
from src.environment.map import GameMap
from src.environment.patrol_generator import EnemySpec
from src.environment.step_events import StepEvents
from src.environment.traversal import can_step_on_map
from src.environment.transition_rules import fall_damage_on_map, hazard_damage_at, enemy_contact, classify_outcome, apply_damage_to_hp

# Dataclass that can be used to track and simulate timesteps
@dataclass(frozen=True)
class SimulatedFrame:
    action: Action
    agent_position: Position
    agent_hp: float
    enemy_position: Position | None
    enemy_patrol_index: int | None
    events: StepEvents
    end_reason: str | None

# Dataclass that stores frames and shows outcome
@dataclass(frozen=True)
class ActionReplay:
    frames: tuple[SimulatedFrame, ...]
    outcome: str
    final_position: Position
    final_hp: float

    @property
    def successful(self) -> bool:
        return self.outcome == "goal"

# Replay actions using GatorEnv mechanics
def simulate_actions(game_map: GameMap, enemy_spec: EnemySpec | None, config: EnvironmentConfig, actions: Sequence[Action | int]) -> ActionReplay:
    # --- Basic config checks --- #
    if config.episode_limit_mode != "research":
        raise ValueError("Witness replay requires research episode limits")

    if config.enemy_enabled != (enemy_spec is not None):
        raise ValueError("Enemy configuration and EnemySpec disagree")
    
    if enemy_spec  is not None:
        route = enemy_spec.patrol_route
        phase = enemy_spec.initial_patrol_index

        # Reverifies proper patrol settings
        if (
            not route
            or not 0 <= enemy_index < len(route)
            or len(set(route)) < 3
        ):
            raise ValueError("Invalid patrol route")

        enemy = Enemy(position=route[phase], patrol_route=list(route), patrol_index=phase, contact_damage=float(config.enemy_contact_damage))

    horizon = config.get_episode_step_limit(width=game_map.width, height=game_map.height, procedural=True)

    position = game_map.start
    hp = float(config.max_hp)
    frames = []

    def can_step(src: Position, dst: Position) -> bool:
        return can_step_on_map(game_map, src, dst)

    # --- Replay each action --- #
    for step_number, requested_action in enumerate(actions, start=1):
        if not ACTION_SPACE.contains(requested_action):
            raise ValueError(f"Invalid action: {requested_action}")
        
        action = Action(requested_action)
        events = StepEvents(step_taken=True)

        old_position = position

        # --- Agent movement --- #
        if action == Action.WAIT:
            events.waited = True

        else:
            dx, dy = ACTION_DELTAS[action]
            destination = (position[0] + dx, position[1] + dy)

            if can_step(position, destination):
                fall_damage = fall_damage_on_map(game_map, config, position, destination)

                position = destination

                hp, actual_damage = apply_damage_to_hp(hp, fall_damage)
                events.fall_damage += actual_damage
                events.damage_taken += actual_damage
            else:
                events.invalid_move = True

        # --- Hazard damage --- #
        tile = game_map.get_tile(*position)
        events.hazard_contact = tile.hazard is not None

        hazard_damage = hazard_damage_at(game_map, config, position)
        hp, actual_damage = apply_damage_to_hp(hp, hazard_damage)

        events.hazard_damage += actual_damage
        events.damage_taken += actual_damage

        # --- Enemy movement --- #
        previous_enemy_pos = enemy.position if enemy is not None else None

        advance_enemy(enemy, can_step)

        current_enemy_pos = enemy.position if enemy is not None else None

        # --- Enemy contact --- #
        if enemy is not None and enemy_contact(old_position, position, previous_enemy_pos, current_enemy_pos):
            events.enemy_contact = True
            hp, actual_damage = apply_damage_to_hp(hp, enemy.contact_damage)

            events.damage_taken += actual_damage
            events.enemy_damage += actual_damage

        # --- Goal progress --- #
        old_distance = abs(game_map.goal[0] - old_position[0]) + abs(game_map.goal[1] - old_position[1])
        new_distance = abs(game_map.goal[0] - position[0]) + abs(game_map.goal[1] - position[1])
        events.distance_change = float(old_distance - new_distance)

        # --- Episode outcome --- #
        end_reason = classify_outcome(position, game_map.goal, hp, step_number, horizon)
        events.reached_goal = end_reason == "goal"

        frames.append(
            SimulatedFrame(
                action=action,
                agent_position=position,
                agent_hp=hp,
                enemy_position=current_enemy_pos,
                enemy_patrol_index=enemy.patrol_index if enemy is not None else None,
                events=events,
                end_reason=end_reason,
            )
        )

        if end_reason is not None:
            return ActionReplay(frames=tuple(frames), outcome=end_reason, final_position=position, final_hp=hp)
        
    # No terminal state reached
    return ActionReplay(frames=tuple(frames), outcome="actions_exhausted", final_position=position, final_hp=hp)
