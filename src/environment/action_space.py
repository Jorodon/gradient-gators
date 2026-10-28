from enum import IntEnum
from gymnasium import spaces

# 4 direction movement action space and wait (no movement)
class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4

# Change in (x, y) for each action (x increases to right and y increases downwards) tiles[y][x]
ACTION_DELTAS = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, 1),
    Action.WAIT: (0, 0)
}

ACTION_SPACE = spaces.Discrete(len(Action))