from src.environment.action_space import Action


def test_action_space_contains_all_movement_actions(action_space):
    assert action_space.contains(Action.UP)
    assert action_space.contains(Action.RIGHT)
    assert action_space.contains(Action.DOWN)
    assert action_space.contains(Action.LEFT)


def test_action_space_rejects_unknown_action(action_space):
    assert action_space.contains(Action.WAIT)
    assert not action_space.contains(5) #not in action space
