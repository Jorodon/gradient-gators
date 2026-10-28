from dataclasses import dataclass
from typing import Literal

# Defines max width and height for PPO observation
MAX_OBSERVATION_DIMENSION = 10

# Weight for research ep horizon
RESEARCH_STEPS_PER_TILE = 4

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

    #Episode horizon behavior (legacy preserves existing max_steps and can be modified via config, research always uses 4 * actual map area)
    episode_limit_mode: Literal["legacy", "research"] = "legacy"

    # Finds the step limit based on width, height, and RESEARCH_STEPS_PER_TILE. Also validates
    def get_episode_step_limit(self, *, width: int, height: int, procedural: bool) -> int:
        # Config and argument validations
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Map dimensions must be positive ints")
        if self.episode_limit_mode not in ("legacy", "research"):
            raise ValueError("episode_limit_mode must be 'legacy' or 'research'")

        area = width * height

        # --- FIXED, PROCEDURAL, AND RESEARCH RETURNS --- #
        # RESEARCH
        if self.episode_limit_mode == "research":
            return RESEARCH_STEPS_PER_TILE * area

        # PROCEDURAL
        if procedural:
            if type(self.steps_per_tile) is not int or self.steps_per_tile <= 0:
                raise ValueError("steps_per_tile must be a positive int")
            
            multi = self.steps_per_tile
            return area * multi

        # FIXED
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive int")

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

    # Fraction of eligible tiles that can receive hazards (0.0 to 1.0)
    hazard_density: float = 0.10