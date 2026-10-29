from src.configs.training_config import RewardConfig
from src.environment.step_events import StepEvents
from src.environment.sparse_reward import calculate_sparse_reward
from src.rewards.transition_context import (
    StepEventsSnapshot,
    TransitionContext,
    TransitionState,
)


#creates transition context for sparse reward tests
def make_context(
    *,
    end_reason=None,
    terminated=False,
    truncated=False,
):
    events = StepEvents(
        reached_goal=end_reason == "goal",
        step_taken=True,
    )

    return TransitionContext(
        previous=TransitionState(
            position=(0, 0),
            hp=100.0,
            distance_to_goal=1.0,
        ),
        next=TransitionState(
            position=(1, 0),
            hp=100.0,
            distance_to_goal=0.0,
        ),
        events=StepEventsSnapshot.from_events(events),
        step_number=1,
        max_episode_steps=100,
        terminated=terminated,
        truncated=truncated,
        end_reason=end_reason,
    )


#sparse reward rewards when goal is reached successfully
def test_sparse_reward_when_goal_reached():
    context = make_context(
        end_reason="goal",
        terminated=True,
    )

    config = RewardConfig(goal_reward=100.0)

    reward = calculate_sparse_reward(
        context,
        config,
    )

    assert reward == 100.0


#sparse reward is zero when goal is not reached
def test_sparse_reward_when_goal_not_reached():
    context = make_context()

    config = RewardConfig(goal_reward=100.0)

    reward = calculate_sparse_reward(
        context,
        config,
    )

    assert reward == 0.0

#sparse reward must not modify transition context
def test_sparse_reward_does_not_modify_context():
    context = make_context(
        end_reason="goal",
        terminated=True,
    )

    config = RewardConfig(goal_reward=100.0)

    previous_state = context.previous
    next_state = context.next
    events = context.events

    calculate_sparse_reward(context, config)

    assert context.previous == previous_state
    assert context.next == next_state
    assert context.events == events