# Reward Calibration Record

## Issue
#107 - Pilot Calibrate and Freeze Reward Coefficients

## Pilot Candidate Sets

Three coefficient sets defined for pilot comparison:

### Candidate A
- goal_reward: 100.0
- step_penalty: 0.05
- hazard_penalty: 1.0
- enemy_penalty: 1.0
- damage_penalty: 0.25
- progress_scale: 0.5
- progress_gamma: 0.99

### Candidate B
- goal_reward: 100.0
- step_penalty: 0.10
- hazard_penalty: 2.0
- enemy_penalty: 2.0
- damage_penalty: 0.50
- progress_scale: 1.0
- progress_gamma: 0.99

### Candidate C
- goal_reward: 100.0
- step_penalty: 0.20
- hazard_penalty: 4.0
- enemy_penalty: 4.0
- damage_penalty: 1.0
- progress_scale: 2.0
- progress_gamma: 0.99

## Hand Audited Checks

- Progress gamma matches PPO gamma: 0.99.
- Max research episode length is 400 steps.
- Step penalties alone don't outweigh the +100 goal reward for any pilot candidate.
- A representative successful trajectory remains positive with all three candidates.
- WAIT actions at distance 5 produce raw progress component of +0.05 when progress_scale = 1.0.
- Invalid actions with no movement produce the same raw progress component of +0.05 when progress_scale = 1.0.
- A one-step forward/backward movement loop produces:
  - forward progress component: +1.04
  - backward progress component: -0.95
  - net raw progress component: +0.09
- Hazard, enemy-contact, and damage penalties increase consistently from Candidate A through Candidate C.

## Open Calibration Concern

Potential based progress shaping with gamma = 0.99 can produce a small positive raw shaping contribution for unchanged distance steps/simple movement loops. This need to be considered when selecting final progress_scale.

## Dataset Restriction

Pilot coefficient selection must only use separately identified pilot/TRAIN as well as checkpoint validation scenarios. Final held out ID and OOD scenarios cannot be used for coefficient selection.

## Final Selection

Not frozen yet. PPO rollout benchmarking/pilot coefficient comparison still required.

## PPO / SB3 Verification

- Stable-Baselines3 2.9.2 explicitly detects `TimeLimit.truncated`.
- On truncation, SB3 evaluates the terminal observation and adds `gamma * terminal_value` before storing rollout transition.
- Time limit truncations are bootstrapped rather than treated as true terminal states.

## PPO Rollout Benchmark

### Candidate A
- n_steps: 125
- n_envs: 1
- batch_size: 125
- training interactions: 25,000
- checkpoint boundary: 25,000
- wall time: 7.193 seconds
- throughput: 3475.76 interactions/second
- checkpoint validation interactions: 100
- all 3 pilot TRAIN scenarios used
- save/reload/predict verification passed
- status: completed

### Candidate B
- n_steps: 250
- n_envs: 1
- batch_size: 125
- training interactions: 25,000
- checkpoint boundary: 25,000
- wall time: 7.148 seconds
- throughput: 3497.41 interactions/second
- checkpoint validation interactions: 100
- all 3 pilot TRAIN scenarios used
- save/reload/predict verification passed
- status: completed

### Candidate C
- n_steps: 500
- n_envs: 1
- batch_size: 125
- training interactions: 25,000
- checkpoint boundary: 25,000
- wall time: 7.425 seconds
- throughput: 3366.95 interactions/second
- checkpoint validation interactions: 29
- all 3 pilot TRAIN scenarios used
- save/reload/predict verification passed
- status: completed

### PPO Selection
Candidate B selected as the shared frozen PPO rollout configuration.

Reasons:
- Highest measured throughput of the three candidates.
- Completed exactly 25,000 training interactions.
- Produced the required 25,000-step checkpoint.
- Save/reload/predict verification passed.
- Uses the value of n_steps = 250

Frozen PPO rollout settings:
- n_steps: 250
- n_envs: 1
- batch_size: 125
- gamma: 0.99

### Efficiency - Candidate A

- step_penalty: 0.05
- validation success: false
- episode length: 100
- episode return: -5.0
- end reason: timestep limit
- invalid moves: 98 / 100
- wait actions: 0
- total distance change: +2.0
- damage taken: 10
- hazard contacts: 1

Behavior summary:
The policy selected RIGHT repeatedly and produced 98 invalid moves before reaching timestep limit. The 0.05 step penalty did not prevent this behavior. 

### Efficiency - Candidate B

- step_penalty: 0.10
- validation success: false
- episode length: 100
- episode return: -10.0
- invalid moves: 94 / 100
- total distance change: +2.0

Behavior summary:
The policy timed out with 94 invalid moves and no goal reached. Candidate B reduced invalid moves barely compared to Candidate A, but repeated invalid action behavior remained.

### Efficiency - Candidate C

- step_penalty: 0.20
- validation success: false
- episode length: 100
- episode return: -20.0
- invalid moves: 98 / 100
- total distance change: +2.0

Behavior summary:
The policy timed out with 98 invalid moves and goal not reached. Increasing the step penalty to 0.20 did not improve the repeated invalid action behavior.

Efficiency selection:
Candidate B (step_penalty = 0.10) is preferred from the pilot comparison since it produced fewest invalid moves, while avoiding stronger 0.20 per step penalty. Repeated invalid action behavior remains.

### Safety - Candidate A

- validation success: false
- damage taken: 10
- hazard contacts: 1
- enemy contacts: 0
- episode return: -3.5

Behavior summary:
The policy timed out after one hazard contact and 10 damage. Candidate A had a total safety penalty of -3.5.

### Safety - Candidate B

- validation success: false
- episode length: 29
- end reason: death
- damage taken: 100
- enemy contacts: 10
- hazard contacts: 0
- episode return: -70.0

Behavior summary:
The policy died after 29 steps with 10 enemy contacts and 100 damage. Unsafe behavior was penalized but did not prevent it.

### Safety - Candidate C

- validation success: false
- episode length: 29
- end reason: death
- damage taken: 100
- enemy contacts: 10
- episode return: -140.0

Behavior summary:
Candidate C produced the same unsafe behavior as Candidate B but made penalty -140, exceeding the +100 goal reward.

Safety selection:
Candidate B is preferred since candidate A provided weaker safety penalties, while C increased penalty magnitude without improving pilot behavior.

### Balanced - Candidate A

- validation success: false
- episode length: 29
- end reason: death
- damage taken: 100
- enemy contacts: 10
- invalid moves: 27
- episode return: -36.45

Behavior summary:
The policy died after 29 steps with 10 enemy contacts and 100 damage. Candidate A combined the step and safety penalties correctly but didn't prevent unsafe behavior.

### Balanced - Candidate B

- validation success: false
- episode length: 100
- end reason: timestep limit
- damage taken: 10
- hazard contacts: 1
- enemy contacts: 0
- invalid moves: 98
- episode return: -17.0

Behavior summary:
The policy timed out with 98 invalid moves, one hazard contact, and 10 damage. Candidate B avoided repeated enemy contact death seen in Candidate A but still showed severe invalid action behavior.

### Balanced - Candidate C

- validation success: false
- episode length: 100
- end reason: timestep limit
- damage taken: 50
- enemy contacts: 5
- invalid moves: 96
- episode return: -90.0

Behavior summary:
Candidate C avoided death but still had 96 invalid moves, 5 enemy contacts, and 50 damage. Its stronger penalties produced a much larger negative return without significantly improving behavior.

Balanced selection:
Candidate B is preferred. It had far less damage than Candidates A and C and avoided repeated enemy contacts, while using more moderate penalty magnitudes, still has repeated invalid actions.

### Progress - Candidate A

- validation success: false
- episode length: 100
- invalid moves: 98
- wait actions: 0
- total distance change: +2.0
- episode return: +2.005

Behavior summary:
The policy timed out with 98 invalid moves but had positive return. This could show progress reward, giving credit for bad behavior.

### Progress - Candidate B

- validation success: false
- episode length: 100
- invalid moves: 98
- total distance change: +2.0
- episode return: +4.01

Behavior summary:
The policy again timed out with 98 invalid moves but earned an even larger positive reward. Similar to A.

### Progress - Candidate C

- validation success: false
- episode length: 100
- invalid moves: 98
- total distance change: +2.0
- episode return: +8.02

Behavior summary:
The policy again timed out with 98 invalid moves and earned an even larger positive reward, so increasing the progress scale made the reward problem worse.

Progress selection:
Candidate A is preferred because it produced the smallest positive reward during the failed invalid action pilot. Higher progress scales increased the reward without improving behavior. 
