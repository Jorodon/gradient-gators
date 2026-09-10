from enum import IntEnum
from gymnasium import spaces

# 4 direction movement action space
class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

ACTION_SPACE = spaces.Discrete(4)