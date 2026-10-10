from dataclasses import dataclass

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

    # Finds the step limit based on width, height, and the steps_per_tile multiplier variable above
    def get_episode_step_limit(self, *, width: int, height: int, procedural: bool) -> int:
        if procedural:
            multi = self.steps_per_tile
            return width * height * multi

        return

@dataclass
class MapConfig:
    width: int = 8
    height: int = 8
    max_elevation: int = 10
    seed: int = 0