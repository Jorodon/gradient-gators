"""Record a+ persist real env trajectories for later playback."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.environment.action_space import Action
from src.environment.generated_scenario import GeneratedScenario
from src.environment.map import GameMap
from src.environment.patrol_generator import EnemySpec
from src.environment.scenario_serialization import scenario_to_dict
from src.environment.step_events import StepEvents


TRAJECTORY_SCHEMA_VERSION = 1
Position = tuple[int, int]
_EVENT_FIELDS = tuple(field.name for field in fields(StepEvents))


def _position(value: object, label: str) -> Position:
    """Validates /normalizes two dimensional map position."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise ValueError(f"Invalid {label}")
    return (value[0], value[1])


def _map_dict(game_map: GameMap) -> dict[str, Any]:
    """Converts map to JSON compatible static playback data."""
    return {
        "width": game_map.width,
        "height": game_map.height,
        "tiles": [
            [
                {
                    "elevation": tile.elevation,
                    "obstacle": tile.obstacle,
                    "hazard": tile.hazard.value if tile.hazard is not None else None,
                    "special_traversal": tile.special_traversal,
                }
                for tile in row
            ]
            for row in game_map.tiles
        ],
        "start": list(game_map.start),
        "goal": list(game_map.goal),
    }


def _enemy_dict(enemy_spec: EnemySpec | None) -> dict[str, Any] | None:
    """Converts an optional enemy specification to JSON-compatible data."""
    if enemy_spec is None:
        return None
    return {
        "patrol_route": [list(position) for position in enemy_spec.patrol_route],
        "initial_patrol_index": enemy_spec.initial_patrol_index,
    }


def build_scenario_manifest(
    game_map: GameMap,
    *,
    scenario_id: str,
    enemy_spec: EnemySpec | None = None,
    environment_config: Any | None = None,
    scenario_checksum: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds static scenario data stored once in trajectory.

    Args:
        game_map: Actual map used by the env.
        scenario_id: Stable identifier for the evaluated scenario.
        enemy_spec: Optional route and initial patrol phase for the enemy.
        environment_config: Optional environment config used for the run.
        scenario_checksum: Checksum of saved scenario snapshot (when known).
        provenance: Optional collection and generation metadata.

    Returns:
        dict[str, Any]: Versioned JSON compatible scenario manifest.

    Raises:
        ValueError: If the scenario identifier is empty.
    """
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non empty string")

    map_data = _map_dict(game_map)
    enemy_data = _enemy_dict(enemy_spec)
    config_data = (
        deepcopy(asdict(environment_config))
        if environment_config is not None
        else {}
    )
    provenance_data = deepcopy(dict(provenance or {}))

    if not scenario_checksum:
        scenario_checksum = hashlib.sha256(
            json.dumps(
                {
                    "scenario_id": scenario_id,
                    "map": map_data,
                    "enemy_spec": enemy_data,
                    "environment_config": config_data,
                    "provenance": provenance_data,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_checksum": scenario_checksum,
        "map_id": "map_" + _map_digest(map_data),
        "map": map_data,
        "enemy_spec": enemy_data,
        "environment_config": config_data,
        "provenance": provenance_data,
    }


def scenario_manifest_from_scenario(scenario: GeneratedScenario) -> dict[str, Any]:
    """Create playback manifest from saved generated scenario.

    Args:
        scenario: Generated scenario containing the exact map and enemy setup.

    Returns:
        dict[str, Any]: Manifest linked to the scenario snapshot identity.
    """
    record = scenario_to_dict(scenario)
    manifest = build_scenario_manifest(
        scenario.game_map,
        scenario_id=record["scenario_id"],
        enemy_spec=scenario.enemy_spec,
        environment_config=scenario.environment_config,
        scenario_checksum=record["checksum"],
        provenance=scenario.provenance,
    )
    manifest["scenario_schema_version"] = record["schema_version"]
    return manifest


def _map_digest(map_data: Mapping[str, Any]) -> str:
    """Returns stable digest for a static map representation."""
    canonical = json.dumps(
        map_data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _events_dict(events: StepEvents | Mapping[str, Any]) -> dict[str, Any]:
    """Validates + copies event data returned by the environment."""
    if isinstance(events, StepEvents):
        values = asdict(events)
    elif isinstance(events, Mapping):
        values = dict(events)
    else:
        raise TypeError("info['events'] must be a StepEvents or mapping")

    if set(values) != set(_EVENT_FIELDS):
        raise ValueError("info['events'] does not match the StepEvents schema")
    return {field: values[field] for field in _EVENT_FIELDS}


def _state_from_info(
    info: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Position, float, Position | None, int | None]:
    """Extracts real environment state from an info mapping."""
    if "agent_position" not in info or "agent_hp" not in info:
        raise ValueError("Env info must include agent_position and agent_hp")

    game_map = manifest["map"]
    width, height = game_map["width"], game_map["height"]
    agent_position = _position(info["agent_position"], "agent position")
    if not (0 <= agent_position[0] < width and 0 <= agent_position[1] < height):
        raise ValueError("Agent position is outside the scenario map")

    agent_hp = float(info["agent_hp"])
    if not math.isfinite(agent_hp) or agent_hp < 0:
        raise ValueError("Agent HP must be finite nonnegative num")

    enemy_position = None
    if info.get("enemy_position") is not None:
        enemy_position = _position(info["enemy_position"], "enemy position")
        if not (0 <= enemy_position[0] < width and 0 <= enemy_position[1] < height):
            raise ValueError("Enemy position is outside the scenario map")

    enemy_patrol_index = info.get("enemy_patrol_index")
    if enemy_patrol_index is not None and type(enemy_patrol_index) is not int:
        raise ValueError("Enemy patrol index must be an int")

    return agent_position, agent_hp, enemy_position, enemy_patrol_index


def _validate_runtime_metadata(
    info: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Ensure optional environment scenario metadata matches the trace."""
    if (
        info.get("scenario_id") is not None
        and info["scenario_id"] != manifest["scenario_id"]
    ):
        raise ValueError("Environment scenario_id does not match trajectory")
    if (
        info.get("snapshot_checksum") is not None
        and info["snapshot_checksum"] != manifest["scenario_checksum"]
    ):
        raise ValueError("Environment snapshot checksum does not match trajectory")


@dataclass(frozen=True)
class TrajectoryFrame:
    """Represents the state at reset OR after one executed environment step.

    Args:
        step: Zero for the reset state otherwise the executed timestep number.
        action: Action selected for this transition, or None at reset.
        agent_position: Agent coordinates in the real map coordinate system.
        agent_hp: Agent HP after the transition's damage resolution.
        enemy_position: Enemy coordinates after enemy movement (when enabled).
        enemy_patrol_index: Enemy route index after movement (when enabled).
        reward: Reward returned for the transition or None at reset.
        events: StepEvents returned for the transition or None at reset.
        terminated: Whether this transition ended the episode terminally.
        truncated: Whether this transition ended by timestep truncation.
        end_reason: Environment outcome when the episode ended.
    """

    step: int
    action: Action | None
    agent_position: Position
    agent_hp: float
    enemy_position: Position | None
    enemy_patrol_index: int | None
    reward: float | None
    events: StepEvents | None
    terminated: bool
    truncated: bool
    end_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Converts the frame to JSON compatible data."""
        return {
            "step": self.step,
            "action": None if self.action is None else int(self.action),
            "agent_position": list(self.agent_position),
            "agent_hp": self.agent_hp,
            "enemy_position": (
                None if self.enemy_position is None else list(self.enemy_position)
            ),
            "enemy_patrol_index": self.enemy_patrol_index,
            "reward": self.reward,
            "events": None if self.events is None else asdict(self.events),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "end_reason": self.end_reason,
        }


@dataclass(frozen=True)
class TrajectoryTrace:
    """Represents completed + versioned episode trajectory."""

    trace_id: str
    episode_id: str
    episode_record_id: str
    model_id: str
    reward_type: str
    distribution_label: str
    scenario_id: str
    scenario_checksum: str
    scenario_manifest: dict[str, Any]
    frames: tuple[TrajectoryFrame, ...]
    outcome: dict[str, Any]
    episode_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Converts the trace to its versioned JSON representation."""
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "episode_record_id": self.episode_record_id,
            "model_id": self.model_id,
            "reward_type": self.reward_type,
            "distribution_label": self.distribution_label,
            "scenario_id": self.scenario_id,
            "scenario_checksum": self.scenario_checksum,
            "scenario_manifest": deepcopy(self.scenario_manifest),
            "frames": [frame.to_dict() for frame in self.frames],
            "outcome": deepcopy(self.outcome),
            "episode_summary": deepcopy(self.episode_summary),
        }


class TrajectoryRecorder:
    """Observes one environment episode and optionally retain its trajectory.

    Args:
        episode_id: Stable identifier for this evaluation episode.
        model_id: Identifier for model or policy being evaluated.
        reward_type: Reward configuration label used for the episode.
        scenario_manifest: Static map and scenario identity for playback.
        distribution_label: Optional ID/OOD label.
        trace_id: Optional stable trace identifier. Generated when omitted.
        enabled: Set false for non selected episodes to avoid recording frames.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        model_id: str,
        reward_type: str,
        scenario_manifest: Mapping[str, Any],
        distribution_label: str = "",
        trace_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        if not episode_id or not model_id or not reward_type:
            raise ValueError("episode_id, model_id, and reward_type are required")

        manifest = _validate_manifest(scenario_manifest)
        self.episode_id = episode_id
        self.episode_record_id = episode_id
        self.model_id = model_id
        self.reward_type = reward_type
        self.distribution_label = distribution_label
        self.scenario_manifest = deepcopy(manifest)
        self.scenario_id = manifest["scenario_id"]
        self.scenario_checksum = manifest["scenario_checksum"]
        self.trace_id = trace_id or _trace_id(
            episode_id, model_id, reward_type, self.scenario_id
        )
        self.enabled = enabled
        self._frames: list[TrajectoryFrame] = []
        self._finished = False

    @classmethod
    def from_scenario(
        cls,
        scenario: GeneratedScenario,
        *,
        episode_id: str,
        model_id: str,
        reward_type: str,
        distribution_label: str = "",
        trace_id: str | None = None,
        enabled: bool = True,
    ) -> "TrajectoryRecorder":
        """Creates a recorder linked to a ``GeneratedScenario`` snapshot.

        Args:
            scenario: Exact generated scenario used for the episode.
            episode_id: Stable identifier for this evaluation episode.
            model_id: Identifier for the model or policy being evaluated.
            reward_type: Reward configuration label used for the episode.
            distribution_label: Optional ID/OOD label.
            trace_id: Optional stable trace identifier.
            enabled: Set false for non selected episodes.

        Returns:
            TrajectoryRecorder: Recorder using the scenario's snapshot identity.
        """
        return cls(
            episode_id=episode_id,
            model_id=model_id,
            reward_type=reward_type,
            scenario_manifest=scenario_manifest_from_scenario(scenario),
            distribution_label=distribution_label,
            trace_id=trace_id,
            enabled=enabled,
        )

    @property
    def frames(self) -> tuple[TrajectoryFrame, ...]:
        """Returns recorded frames w/o exposing mutable recorder state."""
        return tuple(self._frames)

    def record_reset(self, info: Mapping[str, Any]) -> TrajectoryFrame | None:
        """Records initial state returned by ``env.reset()``.

        Args:
            info: Reset info mapping returned by the env.

        Returns:
            TrajectoryFrame | None: Initial frame or None when recording disabled.

        Raises:
            RuntimeError: If reset was already recorded or the trace was finished.
        """
        if not self.enabled:
            return None
        if self._frames or self._finished:
            raise RuntimeError("The trajectory reset has already been recorded")

        _validate_runtime_metadata(info, self.scenario_manifest)
        agent_position, agent_hp, enemy_position, patrol_index = _state_from_info(
            info, self.scenario_manifest
        )
        frame = TrajectoryFrame(
            step=0,
            action=None,
            agent_position=agent_position,
            agent_hp=agent_hp,
            enemy_position=enemy_position,
            enemy_patrol_index=patrol_index,
            reward=None,
            events=None,
            terminated=False,
            truncated=False,
            end_reason=None,
        )
        self._frames.append(frame)
        return frame

    def record_step(
        self,
        action: Action | int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> TrajectoryFrame | None:
        """Records one actual ``env.step()`` result.

        Args:
            action: Action selected by the policy.
            reward: Numerical reward returned by the environment.
            terminated: Gymnasium terminal flag returned by the environment.
            truncated: Gymnasium truncation flag returned by the environment.
            info: Step info mapping including actual state and StepEvents.

        Returns:
            TrajectoryFrame | None: Recorded transition, or None when disabled.

        Raises:
            RuntimeError: If reset was not recorded or the episode is complete.
            ValueError: If action, reward, state, or events are invalid.
        """
        if not self.enabled:
            return None
        if not self._frames:
            raise RuntimeError("Call record_reset before record_step")
        if self._finished or self._frames[-1].terminated or self._frames[-1].truncated:
            raise RuntimeError("Can't record a step after the episode ended")

        try:
            action_value = Action(action)
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("Reward must be finite")
        if not isinstance(info, Mapping):
            raise TypeError("info must be a mapping")

        _validate_runtime_metadata(info, self.scenario_manifest)
        agent_position, agent_hp, enemy_position, patrol_index = _state_from_info(
            info, self.scenario_manifest
        )
        events = _events_dict(info.get("events", {}))
        end_reason = info.get("episode_end_reason")
        if end_reason is None:
            if terminated and agent_hp <= 0:
                end_reason = "death"
            elif events["reached_goal"]:
                end_reason = "goal"
            elif truncated:
                end_reason = "timestep_limit"

        frame = TrajectoryFrame(
            step=len(self._frames),
            action=action_value,
            agent_position=agent_position,
            agent_hp=agent_hp,
            enemy_position=enemy_position,
            enemy_patrol_index=patrol_index,
            reward=reward_value,
            events=StepEvents(**events),
            terminated=bool(terminated),
            truncated=bool(truncated),
            end_reason=end_reason,
        )
        self._frames.append(frame)
        return frame

    def finish(
        self,
        *,
        episode_record: Any | Mapping[str, Any] | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        success: bool | None = None,
        end_reason: str | None = None,
    ) -> TrajectoryTrace | None:
        """Finalizes the trace and optionally attach its exact episode summary.

        Args:
            episode_record: ``EpisodeRecord`` or its dictionary representation.
            terminated: Terminal flag when no episode record is supplied.
            truncated: Truncation flag when no episode record is supplied.
            success: Optional success override when no episode record is supplied.
            end_reason: Optional outcome override when no episode record is supplied.

        Returns:
            TrajectoryTrace | None: Completed trace, or None when disabled.

        Raises:
            RuntimeError: If no reset was recorded or the trace was already finished.
            ValueError: If outcome metadata conflicts with the recorded trace.
        """
        if not self.enabled:
            return None
        if not self._frames:
            raise RuntimeError("Call record_reset before finish")
        if self._finished:
            raise RuntimeError("The trajectory has already been finished")

        summary = _summary_dict(episode_record)
        if summary is not None:
            if summary.get("scenario_id") and summary["scenario_id"] != self.scenario_id:
                raise ValueError("Episode summary scenario_id does not match trace")
            if summary.get("reward_type") != self.reward_type:
                raise ValueError("Episode summary reward_type does not match trace")
            terminated = bool(summary["terminated"])
            truncated = bool(summary["truncated"])
            success = bool(summary["success"])
            end_reason = summary.get("end_reason")
        elif terminated is None or truncated is None:
            raise ValueError("terminated and truncated are required w/o a summary")

        final_frame = self._frames[-1]
        final_reason = end_reason or final_frame.end_reason
        if final_reason is None:
            if success:
                final_reason = "goal"
            elif truncated:
                final_reason = "timestep_limit"
            elif terminated and final_frame.agent_hp <= 0:
                final_reason = "death"

        if success is None:
            success = final_reason == "goal"

        outcome = {
            "success": bool(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_reason": final_reason,
            "final_hp": final_frame.agent_hp,
            "final_position": list(final_frame.agent_position),
        }
        self._finished = True
        return TrajectoryTrace(
            trace_id=self.trace_id,
            episode_id=self.episode_id,
            episode_record_id=self.episode_record_id,
            model_id=self.model_id,
            reward_type=self.reward_type,
            distribution_label=self.distribution_label,
            scenario_id=self.scenario_id,
            scenario_checksum=self.scenario_checksum,
            scenario_manifest=deepcopy(self.scenario_manifest),
            frames=tuple(self._frames),
            outcome=outcome,
            episode_summary=summary,
        )


def _trace_id(episode_id: str, model_id: str, reward_type: str, scenario_id: str) -> str:
    """Generates a stable identifier from trace's identity fields."""
    value = "|".join((episode_id, model_id, reward_type, scenario_id))
    return "trace_" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _summary_dict(summary: Any | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalizes an EpisodeRecord or mapping without mutating it."""
    if summary is None:
        return None
    if hasattr(summary, "to_dict"):
        values = summary.to_dict()
    elif isinstance(summary, Mapping):
        values = dict(summary)
    else:
        raise TypeError("episode_record must be EpisodeRecord or mapping")
    return deepcopy(values)


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validates trajectory scenario manifest and return  a defensive copy."""
    if not isinstance(value, Mapping):
        raise TypeError("scenario_manifest must be a mapping")
    manifest = deepcopy(dict(value))
    required = {
        "schema_version",
        "scenario_id",
        "scenario_checksum",
        "map_id",
        "map",
        "enemy_spec",
        "environment_config",
        "provenance",
    }
    if not required <= manifest.keys():
        raise ValueError("Incomplete scenario manifest")
    if manifest["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory scenario manifest version")
    if not all(isinstance(manifest[key], str) and manifest[key] for key in (
        "scenario_id", "scenario_checksum", "map_id"
    )):
        raise ValueError("Invalid scenario manifest identifiers")

    map_data = manifest["map"]
    if not isinstance(map_data, Mapping):
        raise ValueError("Invalid scenario map")
    if set(map_data) != {"width", "height", "tiles", "start", "goal"}:
        raise ValueError("Incomplete scenario map")
    width, height = map_data["width"], map_data["height"]
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Invalid scenario map dimensions")
    tiles = map_data["tiles"]
    if not isinstance(tiles, list) or len(tiles) != height:
        raise ValueError("Invalid scenario tile rows")
    for row in tiles:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("Invalid scenario tile columns")
        for tile in row:
            if not isinstance(tile, Mapping) or set(tile) != {
                "elevation", "obstacle", "hazard", "special_traversal"
            }:
                raise ValueError("Invalid scenario tile")
            if (
                type(tile["elevation"]) is not int
                or type(tile["obstacle"]) is not bool
                or type(tile["special_traversal"]) is not bool
                or tile["hazard"] not in (None, "damage")
            ):
                raise ValueError("Invalid scenario tile fields")

    start = _position(map_data["start"], "scenario start")
    goal = _position(map_data["goal"], "scenario goal")
    if start == goal or any(
        not (0 <= position[0] < width and 0 <= position[1] < height)
        for position in (start, goal)
    ):
        raise ValueError("Invalid scenario start or goal")

    enemy = manifest["enemy_spec"]
    if enemy is not None:
        if not isinstance(enemy, Mapping) or set(enemy) != {
            "patrol_route", "initial_patrol_index"
        }:
            raise ValueError("Invalid scenario enemy specification")
        route = enemy["patrol_route"]
        phase = enemy["initial_patrol_index"]
        if not isinstance(route, list) or not route or type(phase) is not int:
            raise ValueError("Invalid scenario enemy route")
        if not 0 <= phase < len(route):
            raise ValueError("Invalid scenario enemy patrol phase")
        for waypoint in route:
            waypoint = _position(waypoint, "patrol waypoint")
            if not (0 <= waypoint[0] < width and 0 <= waypoint[1] < height):
                raise ValueError("Patrol waypoint is outside the scenario map")

    if not isinstance(manifest["environment_config"], Mapping):
        raise ValueError("Invalid scenario environment configuration")
    if not isinstance(manifest["provenance"], Mapping):
        raise ValueError("Invalid scenario provenance")
    manifest["map"] = dict(map_data)
    return manifest


def _frame_from_dict(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_step: int,
) -> TrajectoryFrame:
    """Validates + restores one serialized trajectory frame."""
    required = {
        "step", "action", "agent_position", "agent_hp", "enemy_position",
        "enemy_patrol_index", "reward", "events", "terminated", "truncated",
        "end_reason",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Invalid trajectory frame fields")
    if value["step"] != expected_step:
        raise ValueError("Trajectory frames are not ordered")

    action = None if value["action"] is None else Action(value["action"])
    agent_position, agent_hp, enemy_position, patrol_index = _state_from_info(
        {
            "agent_position": value["agent_position"],
            "agent_hp": value["agent_hp"],
            "enemy_position": value["enemy_position"],
            "enemy_patrol_index": value["enemy_patrol_index"],
        },
        manifest,
    )
    reward = value["reward"]
    if reward is not None:
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Invalid trajectory reward")
    events = None if value["events"] is None else StepEvents(**_events_dict(value["events"]))
    if expected_step == 0 and (action is not None or reward is not None or events is not None):
        raise ValueError("Step zero must be the initial state")
    if expected_step > 0 and (action is None or reward is None or events is None):
        raise ValueError("Transition frames require action, reward, and events")
    return TrajectoryFrame(
        step=expected_step,
        action=action,
        agent_position=agent_position,
        agent_hp=agent_hp,
        enemy_position=enemy_position,
        enemy_patrol_index=patrol_index,
        reward=reward,
        events=events,
        terminated=bool(value["terminated"]),
        truncated=bool(value["truncated"]),
        end_reason=value["end_reason"],
    )


def trajectory_from_dict(data: Mapping[str, Any]) -> TrajectoryTrace:
    """Validates + restores a trajectory from decoded JSON data.

    Args:
        data: Decoded versioned trajectory record.

    Returns:
        TrajectoryTrace: Validated trajectory suitable for playback.

    Raises:
        ValueError: If the record is incomplete, inconsistent, or corrupted.
    """
    required = {
        "schema_version", "trace_id", "episode_id", "episode_record_id",
        "model_id", "reward_type", "distribution_label", "scenario_id",
        "scenario_checksum", "scenario_manifest", "frames", "outcome",
        "episode_summary",
    }
    if not isinstance(data, Mapping) or set(data) != required:
        raise ValueError("Invalid trajectory fields")
    if data["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema version")
    manifest = _validate_manifest(data["scenario_manifest"])
    if (
        data["scenario_id"] != manifest["scenario_id"]
        or data["scenario_checksum"] != manifest["scenario_checksum"]
    ):
        raise ValueError("Trajectory scenario identity mismatch")
    if not all(isinstance(data[key], str) and data[key] for key in (
        "trace_id", "episode_id", "episode_record_id", "model_id", "reward_type",
    )):
        raise ValueError("Invalid trajectory identifiers")

    raw_frames = data["frames"]
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("Trajectory must contain an initial frame")
    frames = tuple(
        _frame_from_dict(frame, manifest, index)
        for index, frame in enumerate(raw_frames)
    )
    if frames[0].step != 0 or frames[0].action is not None:
        raise ValueError("Trajectory must begin with step zero")

    outcome = data["outcome"]
    if not isinstance(outcome, Mapping) or set(outcome) != {
        "success", "terminated", "truncated", "end_reason", "final_hp", "final_position"
    }:
        raise ValueError("Invalid trajectory outcome")
    final_position = _position(outcome["final_position"], "final position")
    if final_position != frames[-1].agent_position:
        raise ValueError("Trajectory final position doesn't match its last frame")
    if float(outcome["final_hp"]) != frames[-1].agent_hp:
        raise ValueError("Trajectory final HP doesn't match its last frame")

    summary = data["episode_summary"]
    if summary is not None and not isinstance(summary, Mapping):
        raise ValueError("Invalid episode summary")
    return TrajectoryTrace(
        trace_id=data["trace_id"],
        episode_id=data["episode_id"],
        episode_record_id=data["episode_record_id"],
        model_id=data["model_id"],
        reward_type=data["reward_type"],
        distribution_label=data["distribution_label"],
        scenario_id=data["scenario_id"],
        scenario_checksum=data["scenario_checksum"],
        scenario_manifest=manifest,
        frames=frames,
        outcome=dict(outcome),
        episode_summary=None if summary is None else dict(summary),
    )


def save_trajectory(trace: TrajectoryTrace, path: str | Path) -> Path:
    """Writes a validated trajectory to a readable JSON file.

    Args:
        trace: Completed trajectory to save.
        path: Destination JSON path.

    Returns:
        Path: Destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def load_trajectory(path: str | Path) -> TrajectoryTrace:
    """Load + validate a trajectory JSON file.

    Args:
        path: JSON file containing a saved trajectory.

    Returns:
        TrajectoryTrace: Validated playback trace.
    """
    with Path(path).open(encoding="utf-8") as file:
        return trajectory_from_dict(json.load(file))


def reconstruct_path(trace: TrajectoryTrace) -> tuple[Position, ...]:
    """Returns ordered real-map coordinates for visualization.

    Args:
        trace: Loaded or newly completed trajectory.

    Returns:
        tuple[Position, ...]: Initial position followed by every transition state.
    """
    return tuple(frame.agent_position for frame in trace.frames)
