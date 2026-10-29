from dataclasses import FrozenInstanceError

import pytest

from src.environment.step_events import StepEvents
from src.rewards.transition_context import (
    StepEventsSnapshot,
    TransitionContext,
    TransitionState,
)


#creates valid transition state
def make_state(
    position=(0, 0),
    hp=100.0,
    distance_to_goal=5.0,
) -> TransitionState:
    return TransitionState(
        position=position,
        hp=hp,
        distance_to_goal=distance_to_goal,
    )


#creates valid transition context
def make_context(
    end_reason=None,
    terminated=False,
    truncated=False,
) -> TransitionContext:
    events = StepEventsSnapshot.from_events(
        StepEvents(step_taken=True)
    )

    return TransitionContext(
        previous=make_state(),
        next=make_state(
            position=(1, 0),
            distance_to_goal=4.0,
        ),
        events=events,
        step_number=1,
        max_episode_steps=100,
        terminated=terminated,
        truncated=truncated,
        end_reason=end_reason,
    )


#ensures valid transition state values are stored properly
def test_transition_state_stores_values():
    state = make_state(
        position=(2, 3),
        hp=75.0,
        distance_to_goal=4.0,
    )

    assert state.position == (2, 3)
    assert state.hp == 75.0
    assert state.distance_to_goal == 4.0


#ensures that invalid state values will be rejected
@pytest.mark.parametrize(
    "position,hp,distance",
    [
        ((0,), 100.0, 5.0),
        ((0, "1"), 100.0, 5.0),
        ((0, 0), -1.0, 5.0),
        ((0, 0), 100.0, -1.0),
    ],
)
def test_transition_state_rejects_invalid_values(
    position,
    hp,
    distance,
):
    with pytest.raises(ValueError):
        TransitionState(
            position=position,
            hp=hp,
            distance_to_goal=distance,
        )


#ensures StepEvents get copied into immutable snapshot
def test_step_events_snapshot_copies_events():
    events = StepEvents(
        reached_goal=True,
        damage_taken=10.0,
        fall_damage=2.0,
        hazard_damage=3.0,
        enemy_damage=5.0,
        hazard_contact=True,
        enemy_contact=True,
        invalid_move=False,
        waited=False,
        step_taken=True,
        distance_change=1.0,
    )

    snapshot = StepEventsSnapshot.from_events(events)

    assert snapshot.reached_goal is True
    assert snapshot.damage_taken == 10.0
    assert snapshot.fall_damage == 2.0
    assert snapshot.hazard_damage == 3.0
    assert snapshot.enemy_damage == 5.0
    assert snapshot.hazard_contact is True
    assert snapshot.enemy_contact is True
    assert snapshot.step_taken is True
    assert snapshot.distance_change == 1.0


#ensures changing original StepEvents will not change snapshot
def test_step_events_snapshot_does_not_share_mutable_state():
    events = StepEvents(
        damage_taken=5.0,
        step_taken=True,
    )

    snapshot = StepEventsSnapshot.from_events(events)

    events.damage_taken = 50.0

    assert snapshot.damage_taken == 5.0


#ensures transition state won't be changed after creation
def test_transition_state_is_immutable():
    state = make_state()

    with pytest.raises(FrozenInstanceError):
        state.hp = 50.0


#ensures event snapshot cannot be changed after creation
def test_step_events_snapshot_is_immutable():
    snapshot = StepEventsSnapshot.from_events(
        StepEvents(step_taken=True)
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.damage_taken = 50.0


#ensures transition context won't be changed after creation
def test_transition_context_is_immutable():
    context = make_context()

    with pytest.raises(FrozenInstanceError):
        context.step_number = 2


#ensures goal outcome is shown correctly
def test_goal_transition():
    context = make_context(
        end_reason="goal",
        terminated=True,
        truncated=False,
    )

    assert context.successful_goal is True
    assert context.death is False
    assert context.timeout is False


#ensures death outcome is shown correctly
def test_death_transition():
    context = make_context(
        end_reason="death",
        terminated=True,
        truncated=False,
    )

    assert context.successful_goal is False
    assert context.death is True
    assert context.timeout is False


#ensures timestep limit is truncation not termination
def test_timeout_transition():
    context = make_context(
        end_reason="timestep_limit",
        terminated=False,
        truncated=True,
    )

    assert context.successful_goal is False
    assert context.death is False
    assert context.timeout is True


#ensures conflicting outcome flags get rejected
@pytest.mark.parametrize(
    "end_reason,terminated,truncated",
    [
        ("goal", False, False),
        ("goal", True, True),
        ("death", False, False),
        ("timestep_limit", True, False),
        (None, True, False),
        (None, False, True),
    ],
)
def test_transition_context_rejects_invalid_outcomes(
    end_reason,
    terminated,
    truncated,
):
    with pytest.raises(ValueError):
        make_context(
            end_reason=end_reason,
            terminated=terminated,
            truncated=truncated,
        )


#goal/death should be true terminal outcomes
def test_goal_is_true_terminal():
    context = make_context(
        end_reason="goal",
        terminated=True,
        truncated=False,
    )

    assert context.true_terminal is True

#time limit is truncation
def test_timeout_is_not_true_terminal():
    context = make_context(
        end_reason="timestep_limit",
        terminated=False,
        truncated=True,
    )

    assert context.true_terminal is False