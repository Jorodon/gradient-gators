from src.environment.action_space import ACTION_SPACE, Action


def test_action_space():
    assert ACTION_SPACE.contains(Action.UP)
    assert ACTION_SPACE.contains(Action.RIGHT)
    assert ACTION_SPACE.contains(Action.DOWN)
    assert ACTION_SPACE.contains(Action.LEFT)
    assert not ACTION_SPACE.contains(4) #not in action space