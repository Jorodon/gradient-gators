from dataclasses import dataclass
from typing import Literal

@dataclass
class EnvironmentConfig:
    # Fixed map max steps
    max_steps: int = 200

    # Max and starting HP
    max_hp: int = 100

    # Fall-damage mechanics
    safe_fall_height: int = 1
    fall_damage_scale: float = 1.0

    # Use procedural map area to set step limit
    steps_per_tile: int = 4

    # Hazard mechanics. HP removed per timestep the agent spends on a
    # "damage" hazard tile, on entry and for as long as it remains there.
    hazard_damage: float = 10.0

    # Enemy mechanics.
    # Off by default so the Milestone 1 fixed-map tests and the sparse-reward
    # baseline keep their existing behaviour; set True to put an enemy in the world.
    enemy_enabled: bool = False
    enemy_contact_damage: float = 10.0

    # Finds the step limit based on width, height, and the steps_per_tile multiplier variable above
    def get_episode_step_limit(self, *, width: int, height: int, procedural: bool) -> int:
        if procedural:
            multi = self.steps_per_tile
            return width * height * multi

        return self.max_steps

@dataclass
class MapConfig:
    # Fixed map width and height default
    width: int = 8
    height: int = 8

    # Max elevation for both procedural and fixed maps
    max_elevation: int = 10

    # Seed for map generation
    seed: int = 0

    # Variable that only accepts "fixed" and "procedural" for map handling logic
    map_mode: Literal["fixed", "procedural"] = "fixed"