"""
GatorEnv - the Gymnasium environment for the Gradient Gators project.

This class is the "world" the agent acts in. It owns the agent's position and HP,
applies the movement / elevation / fall-damage rules, and reports what happened
on each step. It deliberately does NOT decide reward values itself: it records
what happened into a StepEvents object and hands that to the reward module, so
comparing reward structures never means editing this file.

Modules this file depends on:
    src/configs/environment_config.py     EnvironmentConfig (max_steps, max_hp,
                                          safe_fall_height, fall_damage_scale)
                                          and MapConfig (max_elevation, etc.)
    src/configs/training_config.py        RewardConfig - the reward weights
    src/environment/action_space.py       Action enum, ACTION_DELTAS, ACTION_SPACE
    src/environment/map.py                GameMap and Tile (elevation, obstacle,
                                          hazard, special_traversal)
    src/environment/observation_space.py  declares the observation space
    src/environment/sparse_reward.py      turns StepEvents into a reward number
    src/environment/step_events.py        StepEvents - the per-step record

Coordinates: positions are (x, y). x indexes width and increases to the right;
y indexes height and increases DOWNWARDS. The map stores tiles[y][x], which is
why game_map.get_tile(pos[0], pos[1]) passes x first. Worth flagging, since grid
code often uses (row, col) - the opposite order.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.configs.environment_config import EnvironmentConfig, MapConfig
from src.configs.training_config import RewardConfig
from src.environment.action_space import ACTION_DELTAS, ACTION_SPACE, Action
from src.environment.map import GameMap, load_map_from_file
from src.environment.observation_space import create_observation_space
from src.environment.sparse_reward import calculate_sparse_reward
from src.environment.step_events import StepEvents

# Map loaded when no game_map is passed in - the fixed 7x7 MVP map for
# Milestone 1. The other maps in maps/ (fall, elevation, hazard) are for
# exercising one mechanic at a time in tests.
DEFAULT_MAP_PATH = (Path(__file__).parent / "maps" / "mvp_map.json")

class GatorEnv(gym.Env):
    # Empty because no rendering is implemented yet; Gymnasium still expects
    # the attribute to exist.
    metadata = {"render_modes": []}

    def __init__(
            self,
            game_map: GameMap | None = None,
            environment_config: EnvironmentConfig | None = None,
            map_config: MapConfig | None = None,
            reward_config: RewardConfig | None = None
    ):
        super().__init__()

        # Every argument is optional and falls back to a default, so GatorEnv()
        # works with no arguments. Passing configs in is how an experiment
        # changes world rules or reward weights without editing code.
        self.environment_config = environment_config or EnvironmentConfig()
        self.map_config = map_config or MapConfig()
        self.reward_config = reward_config or RewardConfig()

        # Fixed-map JSON
        # Loaded from disk rather than hard-coded, so a map can be swapped by
        # editing JSON or by passing game_map in. Procedural generation
        # replaces this in Milestone 2.
        self.game_map = (game_map if game_map is not None
                         else load_map_from_file(DEFAULT_MAP_PATH))

        # Uses the implemented space definitions
        # deepcopy so each env instance owns its action space instead of sharing
        # the module-level object - each needs its own RNG state once several
        # envs are sampling in parallel.
        self.action_space = deepcopy(ACTION_SPACE)

        # The observation space is full-map: one (height, width) plane per
        # feature - elevation, obstacles, hazards, special_traversal, goal,
        # enemy_occupancy, agent_occupancy - plus a 1-element agent_hp box.
        # It depends on map size, so it is built here rather than being a
        # module-level constant.
        self.observation_space = create_observation_space(
            width = self.game_map.width,
            height = self.game_map.height,
            map_config = self.map_config,
            environment_config=self.environment_config
        )

        # Per-episode mutable state. reset() sets these properly; assigning them
        # here means the attributes always exist even before the first reset.
        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0

    # Resets env to initial state
    # Called at the start of every episode. Returns (observation, info) - two
    # values, where step() returns five.
    def reset(self, *, seed=None, options=None):
        # Seeds Gymnasium's RNG (self.np_random). Skipping this is the usual
        # cause of runs that cannot be reproduced.
        super().reset(seed=seed)
        self.agent_position = self.game_map.start
        self.agent_hp = self.environment_config.max_hp
        self.steps = 0

        return self._get_obs(), self._get_info()

    # Advance env one timestep
    def step(self, action):
        # Reject anything outside the action space here, rather than letting a
        # bad index fail somewhere less obvious further down.
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Incremented before the move is attempted, so an invalid move still
        # costs a turn (as documented on StepEvents.step_taken).
        self.steps += 1

        # StepEvents is the record of what happened this step. Everything the
        # reward needs gets written here, and it is the only thing handed to the
        # reward module - so reward code never reads env state directly.
        events = StepEvents(step_taken=True)

        # Snapshot taken before the move, so the distance change below has
        # something to compare against.
        old_position = self.agent_position
        old_distance = self._distance_to_goal(old_position)

        # Based on action, determine change in x and y
        # Action(action) turns the raw integer the policy produced into the
        # named enum, so the delta lookup reads by meaning rather than number.
        action = Action(action)
        dx, dy = ACTION_DELTAS[action]

        target_position = (old_position[0] + dx, old_position[1] + dy)

        # If action is valid
        if self._can_step(old_position, target_position):
            # Damage is worked out from the two tiles before the position is
            # committed. Both tiles are passed explicitly, so this does not
            # depend on the assignment order below.
            fall_damage = self._calculate_fall_damage(old_position, target_position)
            self.agent_position = target_position

            if fall_damage > 0:
                # HP floored at 0 so it never goes negative; the termination
                # check below simply tests for <= 0.
                self.agent_hp = max(0.0, self.agent_hp - fall_damage)
                events.fall_damage = fall_damage
                # damage_taken is the running total across all damage sources
                # for this step. Falling is currently the only source that
                # writes to it (see review note 1 at the bottom of the file).
                events.damage_taken += fall_damage
        else:
            # Illegal move: the agent does not move, but the turn is still spent
            # and the attempt is recorded so a reward structure can penalise it.
            events.invalid_move = True


        # Positive when the agent ended the step closer to the goal. This is the
        # (d_{t-1} - d_t) progress term from the proposal, recorded for the
        # progress-shaped reward structure that is not written yet.
        new_distance = self._distance_to_goal(self.agent_position)
        events.distance_change = old_distance - new_distance

        if self.agent_position == self.game_map.goal:
            events.reached_goal = True

        # Calculates reward and determines if environment is terminated or truncated
        # Note calculate_sparse_reward currently reads only events.reached_goal -
        # the damage, invalid-move and distance fields are recorded for the other
        # four reward structures in the proposal, which are still to be built.
        #
        # terminated = the episode genuinely ended (reached the goal, or died).
        # truncated  = we cut it off at the step limit instead.
        # They are separate because the training algorithm bootstraps a value
        # estimate on truncation but not on termination; "and not terminated"
        # stops both being true on the same step.
        reward = calculate_sparse_reward(events, self.reward_config)
        terminated = events.reached_goal or self.agent_hp <= 0
        truncated = self.steps >= self.environment_config.max_steps and not terminated

        # Sets specific end_reason
        # Human-readable label for logging and evaluation. Stays None on every
        # step that is not the last one of an episode.
        end_reason = None
        if events.reached_goal: end_reason = "goal"
        elif self.agent_hp <= 0: end_reason = "death"
        elif truncated: end_reason = "timestep_limit"

        observation = self._get_obs()
        info = self._get_info(events=events, end_reason=end_reason)

        # The five-tuple Gymnasium requires, in this exact order.
        return(
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # return whether movement from source to target is allowed
    # All movement legality lives here, so anything needing to know what is
    # reachable can call this instead of re-implementing the checks.
    def _can_step(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        # Bound checking w/ helper function
        # Checked first: a negative index is valid Python and would silently
        # read a tile from the far edge of the map rather than failing.
        if not self._in_bounds(dst):
            return False

        # Gets target and source tile date
        target_tile = self.game_map.get_tile(dst[0], dst[1])
        source_tile = self.game_map.get_tile(src[0], src[1])

        if target_tile.obstacle:
            return False

        # Finds change in elevation (change < 0 means it's a valid move but is falling)
        # <= 0 covers flat ground and any drop, however deep. Deep drops are
        # allowed but costly - that is fall damage's job, not this check's.
        elevation_change = (target_tile.elevation - source_tile.elevation)
        if elevation_change <= 0:
            return True

        # Any other movement is trying to move up elevation so we return if target tile is special traversal tile to allow upward traversal
        # So going up is possible only onto a ladder-type tile. Note this permits
        # a climb of ANY height onto such a tile, not just one level.
        return target_tile.special_traversal

    # returns fall damage taken from step
    # Implements the formula from the proposal:
    #     D_fall = k * max(0, delta_h - h_safe)
    # with k = fall_damage_scale and h_safe = safe_fall_height, both from
    # EnvironmentConfig (defaults 1.0 and 1).
    def _calculate_fall_damage(self, src: tuple[int, int], dst: tuple[int, int]) -> float:
        src_elevation = self.game_map.get_tile(src[0], src[1]).elevation
        dst_elevation = self.game_map.get_tile(dst[0], dst[1]).elevation

        # Positive means moving downhill.
        drop = src_elevation - dst_elevation

        # Climbing or staying level can never cause fall damage.
        if drop <= 0:
            return 0.0

        # Finds value of unsafe_drop before multiplying fall damage weight and returning
        # Only the part of the drop beyond the safe height is charged, so with
        # safe_fall_height = 1 a one-level drop is free.
        unsafe_drop = max(0, drop - self.environment_config.safe_fall_height)
        return self.environment_config.fall_damage_scale * unsafe_drop

    # Determines if a tile position is within map bounds
    # x is compared against width and y against height - see the coordinate
    # note at the top of the file.
    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos

        inBounds = 0 <= x < self.game_map.width and 0 <= y < self.game_map.height
        return(inBounds)

    # Manhattan (grid) distance from pos to the goal, ignoring obstacles and
    # elevation. See review note 2 at the bottom of the file.
    def _distance_to_goal(self, pos: tuple[int, int]) -> float:
        x, y = pos
        goal_x, goal_y = self.game_map.goal

        # Finds distance to goal (absolute value since change in distance to goal is what reward structure uses)
        dist = float(abs(goal_x - x) + abs(goal_y - y))
        return dist

    # Returns env diagnostic info
    # This dict is for us, not for the agent - the policy never sees info, so
    # anything useful for logging or debugging can safely go in here.
    def _get_info(
            self,
            events: StepEvents | None = None,
            end_reason: str | None = None) -> dict:

        info = {
            "agent_position": self.agent_position,
            "agent_hp": self.agent_hp,
            "steps": self.steps
        }

        # If events exists, convert to a dictionary to display through info
        # events is None when called from reset(), since no step has happened.
        # asdict() flattens the dataclass so it can be logged or serialised.
        if events is not None:
            info["events"] = asdict(events)

        # Creates "episode_end_reason" dictionary key initialized to the provided end_reason string
        # Present only on the final step of an episode.
        if end_reason is not None:
            info["episode_end_reason"] = end_reason

        return info

    """
    TEMPORARY GYMNASIUM-VALID OBSERVATION

    Replace this when the observation builder is implemented. DO NOT USE FOR ACTUAL RL TRAINING! Message @Jordon if you have questions.
    """
    # Returns an all-zero observation of the right shape and dtype for every key
    # in the declared space. That keeps the env valid against its own
    # observation_space (check_env and SB3 both accept it) while carrying no
    # information about the world - hence the warning above. Only agent_hp holds
    # a real value, so the agent cannot currently see the map, the goal, or even
    # where it is standing. See review note 3 at the bottom of the file.
    def _get_obs(self) -> dict:
        # Creates observation dictionary
        observation = {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }

        observation["agent_hp"] = np.array([self.agent_hp], dtype=np.float32)

        return observation


# TODO for the rest of Milestone 1:
#   - decide whether the observation should be a local view instead of coordinates (This project uses full-map observation)
#   - add an enemy (M2, shared with Mattias)
#   - richer episode logging for the evaluation stage

# ---------------------------------------------------------------------------
# REVIEW NOTES (Guoqing, review/environment) - raising these rather than
# changing anything, since this branch is meant to be comments only.
#
# 1. Hazard tiles are never read by this file. map.Tile has a `hazard` field,
#    StepEvents has `hazard_contact` and a `damage_taken` accumulator, and
#    maps/hazard_map.json contains a tile with hazard "damage" - but nothing
#    here checks tile.hazard, so an agent can stand on that tile with HP and
#    hazard_contact both unchanged. Fall damage is currently the only thing
#    that writes to damage_taken. Is hazard damage assigned to someone yet?
#
# 2. _distance_to_goal is Manhattan, so it ignores obstacles and the
#    elevation rules. On all four maps in maps/ it happens to equal the true
#    reachable distance, so nothing is wrong today. It is an assumption worth
#    recording though: once maps force a detour - going away from the goal to
#    reach the only special_traversal tile - distance_change can go negative
#    while the agent is making real progress, and that value feeds the
#    progress-shaped reward. Likely to bite when Milestone 2 generates maps.
#
# 3. observation_builder.build_observation() already appears to be finished,
#    and its output validates against this env's observation_space with the
#    same keys. If that is right then the blocker described in the TEMPORARY
#    comment above is already cleared and _get_obs could call it. Flagging
#    rather than doing it: swapping the observation is a behaviour change and
#    belongs in its own PR, not a comments branch. @Jordon - is that the
#    intended next step, and is it assigned?
# ---------------------------------------------------------------------------
