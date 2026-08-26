# Gradient Gators

## Reward Design and Robustness in RL Agents

This repo is for a senior research project focused on studying how different reward structures affect the policies learned by a reinforcement learning (RL) agent.

The project will compare a sparse reward baseline against shaped reward structures designed to emphasize priorities such as efficiency, safety, and balanced performance. The trained policies will then be evaluated under changes to the environment to study how well they generalize.

## Research Question

**How do different reward structures affect the policies learned by a reinforcement learning agent, and which reward structures produce the most robust behavior under changes in hazards, enemy behavior, and map layout?**

The project will also examine:

- How sparse and dense reward structures compare in learning efficiency and final performance.
- Which reward structures best balance task completion, safety, and robustness.
- How well trained policies generalize when the environment changes.

## Proposed Environment

The project will use a controlled 2.5D simulated environment designed for repeated RL experiments.

The environment is planned to include:

- A single RL-controlled agent
- A goal or target area
- Static obstacles
- Configurable hazards
- An enemy with adjustable behavior
- Multiple map layouts or environmental configurations

The agent's primary objective will be to reach the goal. Depending on the reward structure being tested, it may also be encouraged to avoid hazards, minimize unsafe interactions, reduce the number of steps taken, and/or adapt to environmental changes.

## Reward Structures

The initial planned comparison includes:

- **Sparse baseline** — reward for reaching the goal.
- **Efficiency focused** — adds a per-step penalty to encourage shorter paths.
- **Safety focused** — penalizes hazards, damage, and unsafe enemy interactions.
- **Balanced** — combines efficiency and safety incentives.
- **Progress focused** — may be explored later using progress toward the goal as an additional reward signal.

These reward structures may be adjusted as development and testing reveal useful behaviors or issues.

## Tech Stack

- **Python** — environment, training, evaluation, and analysis
- **Gymnasium** — custom RL environment API
- **Stable-Baselines3** — RL training and evaluation
- **PyTorch** — neural-network backend
- **NumPy** — environment and observation data
- **pandas** — experiment logging and analysis
- **Matplotlib** — result visualization

Additional visualization tools may be added later if needed.


## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Current Status

The project is currently in the initial planning and environment-development phase.
