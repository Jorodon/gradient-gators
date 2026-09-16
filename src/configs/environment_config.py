from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    max_steps: int = 200
    max_hp: int = 100

@dataclass
class MapConfig:
    width: int = 8
    height: int = 8
    max_elevation: int = 10
    seed: int = 0