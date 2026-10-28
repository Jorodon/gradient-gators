from collections.abc import Iterable

from src.environment.map import GameMap
from src.environment.action_space import Action, ACTION_DELTAS

Position = tuple[int, int]

# Checks if movement is legal on GameMap
def can_step_on_map(game_map: GameMap, src: Position, dst: Position) -> bool:
    # x and y pos for start and destination
    sx, sy = src
    dx, dy = dst

    # Ensures both positions are within bounds of map
    if not (
        0 <= sx < game_map.width
        and 0 <= sy < game_map.height
        and 0 <= dx < game_map.width
        and 0 <= dy < game_map.height
    ):
        return False

    # Ensures that movement is only one tile
    if abs(dx - sx) + abs(dy - sy) != 1:
        return False

    source_tile = game_map.get_tile(sx, sy)
    dest_tile = game_map.get_tile(dx, dy)

    # Ensures specified tiles are not obstacles
    if source_tile.obstacle or dest_tile.obstacle:
        return False
    
    # Ensures that the destination is lower or target is special traversal
    if dest_tile.elevation > source_tile.elevation and not dest_tile.special_traversal:
        return False

    return True

def legal_moves(game_map: GameMap, pos: Position) -> Iterable[tuple[Action, Position]]:
    x, y = pos

    # For each action and direction, update destination
    for action, (dx, dy) in ACTION_DELTAS.items():
        if action == Action.WAIT:
            continue

        dst = (x + dx, y + dy)

        # If can step from pos to dst, pauses function and returns value
        if can_step_on_map(game_map, pos, dst):
            yield action, dst