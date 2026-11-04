import pytest
import math
from src.configs.training_config import RewardConfig
from src.environment.step_events import StepEvents
from src.rewards.reward_calculators import (
    calculate_reward_components,
    calculate_efficiency_reward,
    calculate_safety_reward,
    calculate_balanced_reward,
    calculate_progress_reward,
)
from src.rewards.transition_context import (
    StepEventsSnapshot,
    TransitionContext,
    TransitionState,
)

#creates transition context for tests
def make_context(
    *,
    previous_distance=5.0,
    next_distance=4.0,
    damage_taken=0.0,
    hazard_contact=False,
    enemy_contact=False,
    end_reason=None,
    terminated=False,
    truncated=False,
):
    events = StepEvents(
        reached_goal=end_reason == "goal",
        damage_taken=damage_taken,
        hazard_contact=hazard_contact,
        enemy_contact=enemy_contact,
        step_taken=True,
        distance_change=previous_distance - next_distance,
    )

    return TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=previous_distance,
        ),
        next=TransitionState(
            position=(1, 0),
            hp=100.0 - damage_taken,
            distance_to_goal=next_distance,
        ),
        events=StepEventsSnapshot.from_events(events),
        step_number=1,
        max_episode_steps=100,
        terminated=terminated,
        truncated=truncated,
        end_reason=end_reason,
    )

#verifies efficiency reward actually subtracts timestep penalty
def test_efficiency_reward_applies_step_penalty():
    context = make_context()

    config = RewardConfig(
        variant="efficiency",
        goal_reward=100.0,
        step_penalty=2.0,
    )

    reward = calculate_efficiency_reward(context, config)

    assert reward == -2.0

#verifies efficiency reward also accounts for goal reward when successful
def test_efficiency_reward_includes_successful_goal():
    context = make_context(
        next_distance=0.0,
        end_reason="goal",
        terminated=True,
    )

    config = RewardConfig(
        variant="efficiency",
        goal_reward=100.0,
        step_penalty=2.0,
    )

    reward = calculate_efficiency_reward(context, config)

    assert reward == 98.0

#verifies that safety reward take all penalties into account
def test_safety_reward_penalizes_unsafe_events():
    context = make_context(
        damage_taken=5.0,
        hazard_contact=True,
        enemy_contact=True,
    )

    config = RewardConfig(
        variant="safety",
        hazard_penalty=3.0,
        enemy_penalty=4.0,
        damage_penalty=2.0,
    )

    reward = calculate_safety_reward(context, config)

    #0 goal - 3 hazard - 4 enemy - (2 * 5 damage)
    assert reward == -17.0

#verifies balanced reward accounts for efficiency and safety penalties
def test_balanced_reward_combines_components():
    context = make_context(
        damage_taken=5.0,
        hazard_contact=True,
        enemy_contact=True,
    )

    config = RewardConfig(
        variant="balanced",
        step_penalty=1.0,
        hazard_penalty=3.0,
        enemy_penalty=4.0,
        damage_penalty=2.0,
    )

    reward = calculate_balanced_reward(context, config)

    #0 goal - 1 step - 3 hazard - 4 enemy - 10 damage
    assert reward == -18.0

#verifies progress reward is positive when agent moves towards goal
def test_progress_reward_for_moving_closer():
    context = make_context(
        previous_distance=5.0,
        next_distance=4.0,
    )

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    reward = calculate_progress_reward(context, config)

    #phi(previous) = -5
    #phi(next) = -4
    #f = 0.99(-4) - (-5) = 1.04
    assert reward == pytest.approx(1.04)

#negative when moving away from goal
def test_progress_reward_for_moving_farther():
    context = make_context(
        previous_distance=4.0,
        next_distance=5.0,
    )

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    reward = calculate_progress_reward(context, config)

    #0.99(-5) - (-4) = -0.95
    assert reward == pytest.approx(-0.95)

#verifies true terminal states are using zero next state potential
def test_progress_true_terminal_uses_zero_next_potential():
    context = make_context(
        previous_distance=1.0,
        next_distance=0.0,
        end_reason="goal",
        terminated=True,
    )

    config = RewardConfig(
        variant="progress",
        goal_reward=100.0,
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    reward = calculate_progress_reward(context, config)

    #goal = 100
    #next potential = 0 (true terminal)
    #shaping = 0 - (-1) = 1
    assert reward == pytest.approx(101.0)

#verifies timestep truncation keeps next state potential
def test_progress_timeout_keeps_next_potential():
    context = make_context(
        previous_distance=5.0,
        next_distance=5.0,
        end_reason="timestep_limit",
        terminated=False,
        truncated=True,
    )

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    reward = calculate_progress_reward(context, config)

    #timeout is truncation not true terminal
    #0.99(-5) - (-5) = 0.05
    assert reward == pytest.approx(0.05)

#verifies diagnostic reports each reward component
def test_reward_components_report_expected_values():
    context = make_context(
        damage_taken=5.0,
        hazard_contact=True,
        enemy_contact=True,
    )
    config = RewardConfig(
        variant="balanced",
        goal_reward=100.0,
        step_penalty=1.0,
        hazard_penalty=3.0,
        enemy_penalty=4.0,
        damage_penalty=2.0,
        progress_scale=5.0,
        progress_gamma=0.99,
    )

    components = calculate_reward_components(context, config)

    assert components["goal"] == 0.0
    assert components["step"] == -1.0
    assert components["hazard"] == -3.0
    assert components["enemy"] == -4.0
    assert components["damage"] == -10.0
    assert components["progress"] == pytest.approx(5.2)

#verifies progress diagnostics respect the true terminal potential handling
def test_reward_components_zero_next_potential_on_true_terminal():
    context = make_context(
        previous_distance=1.0,
        next_distance=0.0,
        end_reason="goal",
        terminated=True,
    )
    config = RewardConfig(
        variant="progress",
        goal_reward=100.0,
        progress_scale=1.0,
        progress_gamma=0.99,
    )
    components = calculate_reward_components(context, config)

    assert components["goal"] == 100.0
    assert components["progress"] == pytest.approx(1.0)

#verifies that dying at the goal doesnt receive goal reward
def test_fatal_goal_gets_no_goal_reward():
    context = make_context(
        previous_distance=1.0,
        next_distance=0.0,
        damage_taken=100.0,
        end_reason="death",
        terminated=True,
    )

    config = RewardConfig(
        variant="balanced",
        goal_reward=100.0,
        damage_penalty=1.0,
    )

    reward = calculate_balanced_reward(context, config)

    assert reward == -100.0


#verifies WAIT still receives the efficiency timestep cost
def test_wait_still_gets_efficiency_penalty():
    context = make_context(
        previous_distance=5.0,
        next_distance=5.0,
    )

    config = RewardConfig(
        variant="efficiency",
        step_penalty=1.0,
    )

    reward = calculate_efficiency_reward(context, config)

    assert reward == -1.0


#verifies reward calculation does not change TransitionContext
def test_reward_calculation_does_not_change_context():
    context = make_context(
        previous_distance=5.0,
        next_distance=4.0,
    )

    before = context

    config = RewardConfig(
        variant="progress",
        progress_scale=1.0,
        progress_gamma=0.99,
    )

    calculate_progress_reward(context, config)

    assert context == before

#verifies reward outputs remain finite
def test_reward_outputs_are_finite():

    context = make_context(
        damage_taken=5.0,
        hazard_contact=True,
        enemy_contact=True,
    )

    config = RewardConfig(
        variant="balanced",
        goal_reward=100.0,
        step_penalty=1.0,
        hazard_penalty=2.0,
        enemy_penalty=3.0,
        damage_penalty=4.0,
    )

    reward = calculate_balanced_reward(context, config)

    assert math.isfinite(reward)