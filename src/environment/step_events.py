from dataclasses import dataclass

@dataclass
class StepEvents:
    reached_goal: bool = False
    damage_taken: float = 0.0
    fall_damage: float = 0.0
    hazard_contact: bool = False
    enemy_contact: bool = False
    invalid_move: bool = False
    step_taken: bool = False
    distance_change: float = 0.0