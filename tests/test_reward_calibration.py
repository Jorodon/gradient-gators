import json
import pytest
from pathlib import Path

from src.configs.training_config import PPOSettings, RewardConfig
from src.rewards.reward_calculators import calculate_reward_components
from src.rewards.transition_context import (
    StepEventsSnapshot,
    TransitionContext,
    TransitionState,
)


CANDIDATES_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "reward_calibration"
    / "pilot_candidates.json"
)

#verifies progress shaping and PPO use same discount factor
def test_pilot_progress_gamma_matches_ppo_gamma():
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    ppo_gamma = PPOSettings().gamma

    for candidate in candidates.values():
        assert candidate["progress_gamma"] == ppo_gamma

#makes sure step penalties can't outweigh successful goal
def test_goal_reward_exceeds_maximum_step_penalty():
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    max_episode_steps = 400

    for candidate in candidates.values():
        total_step_penalty = candidate["step_penalty"] * max_episode_steps
        assert candidate["goal_reward"] > total_step_penalty

#makes sure successful episode has positive total reward
def test_representative_success_stays_positive():
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    steps = 100
    damage_taken = 10.0
    hazard_contacts = 1
    enemy_contacts = 1

    for candidate in candidates.values():
        reward = (
            candidate["goal_reward"] - candidate["step_penalty"] * steps - candidate["hazard_penalty"] * hazard_contacts - candidate["enemy_penalty"] * enemy_contacts - candidate["damage_penalty"] * damage_taken
        )
        assert reward > 0

#ensures progress reward is recorded by WAIT
def test_wait_progress_component_is_audited():
    context = TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        next=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        events=StepEventsSnapshot(
            reached_goal=False,
            damage_taken=0.0,
            fall_damage=0.0,
            hazard_damage=0.0,
            enemy_damage=0.0,
            hazard_contact=False,
            enemy_contact=False,
            invalid_move=False,
            waited=True,
            step_taken=True,
            distance_change=0.0,
        ),
        step_number=1,
        max_episode_steps=100,
        terminated=False,
        truncated=False,
        end_reason=None,
    )

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    components = calculate_reward_components(context, config)

    assert components["progress"] == pytest.approx(0.05)

#records progress reward from invalid action
def test_invalid_action_progress_component_is_audited():
    context = TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        next=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        events=StepEventsSnapshot(
            reached_goal=False,
            damage_taken=0.0,
            fall_damage=0.0,
            hazard_damage=0.0,
            enemy_damage=0.0,
            hazard_contact=False,
            enemy_contact=False,
            invalid_move=True,
            waited=False,
            step_taken=True,
            distance_change=0.0,
        ),
        step_number=1,
        max_episode_steps=100,
        terminated=False,
        truncated=False,
        end_reason=None,
    )

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    components = calculate_reward_components(context, config)

    assert components["progress"] == pytest.approx(0.05)

#records progress contribution of forward or backward movement loop
def test_progress_movement_loop_is_not_audited():
    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    forward_context = TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        next=TransitionState(
            position=(1, 0),
            hp=100.0,
            distance_to_goal=4.0,
        ),
        events=StepEventsSnapshot(
            reached_goal=False,
            damage_taken=0.0,
            fall_damage=0.0,
            hazard_damage=0.0,
            enemy_damage=0.0,
            hazard_contact=False,
            enemy_contact=False,
            invalid_move=False,
            waited=False,
            step_taken=True,
            distance_change=1.0,
        ),
        step_number=1,
        max_episode_steps=100,
        terminated=False,
        truncated=False,
        end_reason=None,
    )

    backward_context = TransitionContext(
        previous=TransitionState(
            position=(1, 0),
            hp=100.0,
            distance_to_goal=4.0,
        ),
        next=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        events=StepEventsSnapshot(
            reached_goal=False,
            damage_taken=0.0,
            fall_damage=0.0,
            hazard_damage=0.0,
            enemy_damage=0.0,
            hazard_contact=False,
            enemy_contact=False,
            invalid_move=False,
            waited=False,
            step_taken=True,
            distance_change=-1.0,
        ),
        step_number=2,
        max_episode_steps=100,
        terminated=False,
        truncated=False,
        end_reason=None,
    )

    forward = calculate_reward_components(forward_context, config)["progress"]
    backward = calculate_reward_components(backward_context, config)["progress"]

    assert forward == pytest.approx(1.04)
    assert backward == pytest.approx(-0.95)
    assert forward + backward == pytest.approx(0.09)

#record hazard, enemy, damage contributions for each pilot candidate
def test_safety_components_are_audited_for_pilot_candidates():
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    context = TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=5.0,
        ),
        next=TransitionState(
            position=(1, 0),
            hp=90.0,
            distance_to_goal=4.0,
        ),
        events=StepEventsSnapshot(
            reached_goal=False,
            damage_taken=10.0,
            fall_damage=0.0,
            hazard_damage=5.0,
            enemy_damage=5.0,
            hazard_contact=True,
            enemy_contact=True,
            invalid_move=False,
            waited=False,
            step_taken=True,
            distance_change=1.0,
        ),
        step_number=1,
        max_episode_steps=100,
        terminated=False,
        truncated=False,
        end_reason=None,
    )

    expected = {
        "candidate_a": (-1.0, -1.0, -2.5),
        "candidate_b": (-2.0, -2.0, -5.0),
        "candidate_c": (-4.0, -4.0, -10.0),
    }

    for name, candidate in candidates.items():
        config = RewardConfig(
            variant="safety",
            goal_reward=candidate["goal_reward"],
            hazard_penalty=candidate["hazard_penalty"],
            enemy_penalty=candidate["enemy_penalty"],
            damage_penalty=candidate["damage_penalty"],
        )

        components = calculate_reward_components(context, config)

        hazard, enemy, damage = expected[name]

        assert components["hazard"] == pytest.approx(hazard)
        assert components["enemy"] == pytest.approx(enemy)
        assert components["damage"] == pytest.approx(damage)

#verifies PPO rollout size is exactly 25,000 step checkpoints
def test_default_ppo_rollout_divides_checkpoint_exactly():
    ppo = PPOSettings()

    rollout_size = ppo.n_steps * ppo.n_envs

    assert 25000 % rollout_size == 0

#verifies every PPO candidate is on 25,000 step checkpoints
def test_ppo_candidates_divide_checkpoint_exactly():
    ppo_candidates_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "reward_calibration"
        / "ppo_candidates.json"
    )

    candidates = json.loads(ppo_candidates_path.read_text(encoding="utf-8"))

    for candidate in candidates.values():
        rollout_size = candidate["n_steps"]
        assert 25000 % rollout_size == 0