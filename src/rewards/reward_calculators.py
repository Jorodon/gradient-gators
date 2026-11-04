from src.configs.training_config import RewardConfig
from src.rewards.transition_context import TransitionContext

#sparse: only cares about the goal
def _goal_component(
        context: TransitionContext,
        config: RewardConfig,
) -> float:
    """Goal reward is only given for a living successful goal."""
    return config.goal_reward if context.successful_goal else 0.0

def calculate_reward_components(
        context: TransitionContext,
        config: RewardConfig,
) -> dict[str, float]:
    """Returns individual reward components (diagnostics)"""

    goal = _goal_component(context, config)
    step = -config.step_penalty
    hazard = (
        -config.hazard_penalty
        if context.events.hazard_contact
        else 0.0
    )
    enemy = (
        -config.enemy_penalty
        if context.events.enemy_contact
        else 0.0
    )
    damage = (
        -config.damage_penalty * context.events.damage_taken
    )
    previous_potential = -context.previous.distance_to_goal

    if context.true_terminal:
        next_potential = 0.0
    else:
        next_potential = -context.next.distance_to_goal

    progress = config.progress_scale * (
            config.progress_gamma * next_potential - previous_potential
    )

    return {
        "goal": float(goal),
        "hazard": float(hazard),
        "step": float(step),
        "progress": float(progress),
        "enemy": float(enemy),
        "damage": float(damage),
    }

#efficiency: goal and timestep cost
def calculate_efficiency_reward(
        context: TransitionContext,
        config: RewardConfig,
) -> float:
        """Base success reward but with a penalty for each timestep used."""
        components = calculate_reward_components(context, config)

        return float(
            components["goal"] + components["step"]
        )

#safety: goal, hazard contact, enemy contact, total damage
def calculate_safety_reward(
        context: TransitionContext,
        config: RewardConfig,
) -> float:
        """Reward success while penalizing unsafe events"""
        components = calculate_reward_components(context, config)

        return float(
            components["goal"] + components["hazard"]
            + components["enemy"] + components["damage"]
        )

#balanced: goal, timestep cost, hazard contact, enemy contact, total damage
def calculate_balanced_reward(
        context: TransitionContext,
        config: RewardConfig,
) -> float:
        """Consider success, efficiency, and safety."""
        components = calculate_reward_components(context, config)

        return float(
            components["goal"] + components["hazard"] + components["step"]
            + components["enemy"] + components["damage"]
        )

#progress: goal and progress made towards goal (potential based progress shaping)
def calculate_progress_reward(
        context: TransitionContext,
        config: RewardConfig,
) -> float:
    """Base reward with potential based Manhattan progress shaping."""
    components = calculate_reward_components(context, config)

    return float(
        components["goal"] + components["progress"]
    )
