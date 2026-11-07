# Reward Calibration Record

## Issue
#107 - Pilot Calibrate and Freeze Reward Coefficients

## Candidate Reward Sets

### Candidate A
- goal_reward: 100.0
- step_penalty: 0.05
- hazard_penalty: 1.0
- damage_penalty: 0.25
- enemy_penalty: 1.0
- progress_scale: 0.5
- progress_gamma: 0.99

### Candidate B
- goal_reward: 100.0
- step_penalty: 0.10
- hazard_penalty: 2.0
- damage_penalty: 0.5
- enemy_penalty: 2.0
- progress_scale: 1.0
- progress_gamma: 0.99

### Candidate C
- goal_reward: 100.0
- step_penalty: 0.20
- hazard_penalty: 4.0
- damage_penalty: 1.0
- enemy_penalty: 4.0
- progress_scale: 2.0
- progress_gamma: 0.99

## Calibration Data

Final calibration used M3 Frozen Research Datasets V2.

- TRAIN collection: `research_train_v1`
- TRAIN checksum: `11632493f54b3659008af1d0040d6678db306b9f11132c4469e014fbe364772f`
- CHECKPOINT_VALIDATION collection: `research_checkpoint_validation_v1`
- CHECKPOINT_VALIDATION checksum: `e9b9451df4c358714c713aab4742ef27a58631beaddb17c3d0360154134c40c5`
- Training schedule checksum: `2f3c38345c2e443131d4990f7e184bcda9ef7e4475dc30b6605e4973d6365243`
- Model seed slot: 0
- Model seed: 2278348526


## Hand-Audited Reward Checks

- `progress_gamma` matches PPO gamma at 0.99.
- Maximum research episode length is 400 steps.
- Step penalties alone do not outweigh the +100 goal reward for any candidate.
- Representative successful trajectories remain positive.
- WAIT and invalid actions can receive a small positive raw progress reward when distance does not change.
- A simple forward/backward movement loop can also produce a small positive raw progress total.
- Safety penalties increase consistently from Candidate A through Candidate C.

## PPO Benchmark

All PPO candidates tested at 25,000 training interactions.

### Candidate A
- n_steps: 125
- wall time: 7.437 seconds
- throughput: 3361.54 interactions/second
- reload verification: passed

### Candidate B
- n_steps: 250
- wall time: 7.091 seconds
- throughput: 3525.67 interactions/second
- reload verification: passed

### Candidate C
- n_steps: 500
- wall time: 6.986 seconds
- throughput: 3578.77 interactions/second
- reload verification: passed

## PPO Selection

Selected Candidate B as the shared PPO configuration.

Candidate C was slightly faster, but Candidate B was very close in performance and keeps the existing `n_steps = 250` configuration.

Frozen PPO settings:
- n_steps: 250
- n_envs: 1
- batch_size: 125
- learning_rate: 0.0003
- n_epochs: 2
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.0
- vf_coef: 0.5
- device: cpu

Stable-Baselines3 2.9.2: time limit truncations are bootstrapped instead of treated as true terminal states.

## Reward Pilot Results

### Efficiency

Candidate A:
- step_penalty: 0.05
- invalid moves: 136
- episode return: -7.0

Candidate B:
- step_penalty: 0.10
- invalid moves: 140
- episode return: -14.0

Candidate C:
- step_penalty: 0.20
- invalid moves: 140
- episode return: -28.0

Selected Candidate B. Candidate A gave a weaker step penalty, while Candidate C increased the penalty without improving behavior.

### Safety

Candidate A:
- damage: 0
- enemy contacts: 0
- episode return: 0.0

Candidate B:
- damage: 0
- enemy contacts: 0
- episode return: 0.0

Candidate C:
- damage: 100
- enemy contacts: 10
- episode return: -140.0

Selected Candidate B. Candidate C greatly increased the penalty without preventing unsafe behavior, while Candidate B kept a moderate safety penalty.

### Balanced

Candidate A:
- invalid moves: 140
- damage: 0
- episode return: -7.0

Candidate B:
- invalid moves: 0
- WAIT actions: 140
- damage: 0
- episode return: -14.0

Candidate C:
- invalid moves: 39
- damage: 100
- enemy contacts: 10
- episode return: -148.2

Selected Candidate B since it avoided damage and invalid actions; however, it chose WAIT for the entire episode, reward could be encouraging agent to be too inactive.

### Progress

Candidate A:
- progress_scale: 0.5
- invalid moves: 140
- net distance change: 0
- episode return: +4.2

Candidate B:
- progress_scale: 1.0
- invalid moves: 140
- net distance change: 0
- episode return: +8.4

Candidate C:
- progress_scale: 2.0
- invalid moves: 140
- net distance change: 0
- episode return: +16.8

Selected Candidate A since it gives the smallest unintended positive raw shaping reward while still using potential based shaping approach.

## Frozen Reward Coefficients

### Sparse
- goal_reward: 100.0

### Efficiency
- goal_reward: 100.0
- step_penalty: 0.10

### Safety
- goal_reward: 100.0
- hazard_penalty: 2.0
- enemy_penalty: 2.0
- damage_penalty: 0.5

### Balanced
- goal_reward: 100.0
- step_penalty: 0.10
- hazard_penalty: 2.0
- enemy_penalty: 2.0
- damage_penalty: 0.5

### Progress
- goal_reward: 100.0
- progress_scale: 0.5
- progress_gamma: 0.99

## Remaining Concerns

Some issues are still present in the pilot results:

- Several configurations still showed repeated invalid actions.
- Balanced Candidate B chose WAIT for the entire episode, hence the reward may be encouraging the agent to be too inactive.
- Progress shaping gave a small positive reward even when agent made no net progress toward goal.
- Issues documented before freezing experiment.