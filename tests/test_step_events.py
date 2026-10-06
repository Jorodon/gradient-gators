from src.environment.step_events import StepEvents

def test_step_events_defaults():
    events = StepEvents()

    assert events.reached_goal is False
    assert events.damage_taken == 0.0
    assert events.fall_damage == 0.0
    assert events.distance_change == 0.0
    assert events.hazard_contact is False
    assert events.enemy_contact is False
    assert events.invalid_move is False
    assert events.step_taken is False

def test_step_events_custom_values():
    events = StepEvents(
        reached_goal=True,
        damage_taken=10.0,
        fall_damage=5.0,
        distance_change=1.0,
        hazard_contact=True,
        enemy_contact=True,
        invalid_move=False,
        step_taken=True,
    )

    assert events.reached_goal is True
    assert events.damage_taken == 10.0
    assert events.fall_damage == 5.0
    assert events.distance_change == 1.0
    assert events.hazard_contact is True
    assert events.enemy_contact is True
    assert events.invalid_move is False
    assert events.step_taken is True
