from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    max_steps: int = 200
    starting_hp: int = 100

@dataclass
class MapConfig:
    width: int = 8
    height: int = 8
    seed: int = 0