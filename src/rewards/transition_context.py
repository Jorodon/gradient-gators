from dataclasses import dataclass
from math import isfinite

from src.environment.step_events import StepEvents

Position = tuple[int, int]

#frozen snapshot of agent state
@dataclass(frozen=True)
class TransitionState:
    """Read only agent state at one side of a transition"""

    position: Position
    hp: float
    distance_to_goal: float

    #validates state values
    def __post_init__(self):
        if (
            not isinstance(self.position, tuple)
            or len(self.position) != 2
            or any(type(value) is not int for value in self.position)
        ):
            raise ValueError("position must be a tuple or two integers")

        if not isfinite(self.hp) or self.hp < 0:
            raise ValueError("hp must be finite and non negative")
        if not isfinite(self.distance_to_goal) or self.distance_to_goal < 0:
            raise ValueError("distance_to_goal must be finite and non negative")

@dataclass(frozen=True)
class StepEventsSnapshot:
    """Immutable copy of finalized StepEvents for reward calculation"""
    reached_goal: bool
    damage_taken: float

    #each type of damage
    fall_damage:float
    hazard_damage: float
    enemy_damage: float

    #contact events
    hazard_contact: bool
    enemy_contact: bool

    #action events
    invalid_move: bool
    waited: bool
    step_taken: bool

    #positive: moved close to goal
    #negative: moved away from goal
    distance_change: float

    #immutable event snapshot
    @classmethod
    def from_events(cls, events: StepEvents) -> "StepEventsSnapshot":
        return cls(
            reached_goal=events.reached_goal,
            damage_taken=float(events.damage_taken),
            fall_damage=float(events.fall_damage),
            hazard_damage=float(events.hazard_damage),
            enemy_damage=float(events.enemy_damage),
            hazard_contact=events.hazard_contact,
            enemy_contact=events.enemy_contact,
            invalid_move=events.invalid_move,
            waited=events.waited,
            step_taken=events.step_taken,
            distance_change=float(events.distance_change),
        )

#read only info from one complete environment transition
@dataclass(frozen=True)
class TransitionContext:
    """Immutable finalized transition supplied to reward calculators."""

    #before and after the action
    previous: TransitionState
    next: TransitionState

    #finalized events of the timestep
    events: StepEventsSnapshot

    #timestep info
    step_number: int
    max_episode_steps: int

    #different outcomes
    terminated: bool
    truncated: bool
    end_reason: str | None

    #ensures timestep and outcome match
    def __post_init__(self):
        if type(self.step_number) is not int or self.step_number < 1:
            raise ValueError("step_number must be a postive integer")

        if (
            type(self.max_episode_steps) is not int
            or self.max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer"

        )
        if self.end_reason not in (
            None,
            "goal",
            "death",
            "timestep_limit",
        ):
            raise ValueError("Invalid transition end_reason")

        #goal/death are true terminal states
        if self.end_reason == "goal":
            if not self.terminated or self.truncated:
                raise ValueError("Goal must be terminal and not truncated")

        elif self.end_reason == "death":
            if not self.terminated or self.truncated:
                raise ValueError("Death must be terminal and not truncated")

        #timestep limit is considered truncation
        elif self.end_reason == "timestep_limit":
            if self.terminated or not self.truncated:
                raise ValueError("Timestep limit must be truncated and not terminal")
        #no end reason so episode keeps going
        elif self.terminated or self.truncated:
            raise ValueError("Non ending transition cannot be terminated or truncated")

    #simplifies results
    @property
    def successful_goal(self) -> bool:
        return self.end_reason == "goal"

    @property
    def death(self) -> bool:
        return self.end_reason == "death"

    @property
    def timeout(self) -> bool:
        return self.end_reason == "timestep_limit"

    @property
    def true_terminal(self) -> bool:
        return self.terminated

