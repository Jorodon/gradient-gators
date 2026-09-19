from dataclasses import dataclass

@dataclass
class StepEvents:
    reached_goal: bool = False #Agent reached goal
    damage_taken: float = 0.0 #Total damage during current timestep
    fall_damage: float = 0.0 #Portion of damage_taken caused from falling
    hazard_contact: bool = False #Agent on hazard tile
    enemy_contact: bool = False #Agent interacted with enemy on current timestep
    invalid_move: bool = False #True when attempted movement is not allowed
    step_taken: bool = False #True when a timetep was consumed (INVALID MOVES STILL CONSUME TIMESTEP)
    distance_change: float = 0.0 #Change in distance to the goal (positve = moved closer | negative = moved farther away)