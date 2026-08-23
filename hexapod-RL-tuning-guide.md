# Hexapod RL Locomotion: Development & Reward Tuning Log

This document tracks the progress, errors, fixes, and rationale behind the reinforcement learning reward shaping for the Interbotix PhantomX Mark IV Hexapod in Isaac Lab. 

## 1. Environment Setup & Critical Bugs Resolved

During the initial setup of the `Hexapod-Flat-v0` environment, we encountered several framework-level crashes and bugs.

### Bug 1: `base_contact` Termination Crash
* **Error**: `ValueError: The term 'base_contact' expects mandatory parameters...`
* **Root Cause**: The template code used `mdp.base_height_l2` as a termination condition. However, `base_height_l2` is a **reward function** that returns a continuous tensor, not a boolean termination function.
* **Fix**: Replaced it with the correct built-in termination function: `mdp.root_height_below_minimum` with a threshold of `0.05m`.

### Bug 2: `ContactSensor` USD Segfault (Exit Code 139)
* **Error**: A low-level C++ segmentation fault occurring during `dlopen` at module load time.
* **Root Cause**: Placing `from isaaclab.sensors import ContactSensorCfg` at the top level of the environment config file forced the `TfRegistryManager` to register USD schema types before Isaac Sim's plugin stack was fully initialized, corrupting memory.
* **Fix**: Removed the `ContactSensor` from the scene and attempted to use the articulation's `body_contact_forces`. 

### Bug 3: `body_contact_forces` Attribute Error
* **Error**: `AttributeError: 'ArticulationData' object has no attribute 'body_contact_forces'`
* **Root Cause**: The installed version of Isaac Lab's `ArticulationData` did not expose `body_contact_forces` directly.
* **Fix**: Removed the custom foot contact force penalty entirely. We rely on the **joint torque penalty** as a proxy: if the robot pushes against the ground with excessive force, joint torques will naturally spike and be penalized.

---

## 2. Reward Shaping Journey

Training a hexapod to walk requires carefully balancing rewards to prevent it from finding "lazy" local optima.

### Phase 1: The "Ignore Height" Problem
* **Initial Setup**: A simple `alive` bonus (+1.0) and an L2 `base_height` penalty with a weight of `-1.0`. 
* **Observed Behavior**: The robot immediately dropped to the ground and dragged itself. 
* **Why**: The L2 penalty for being on the ground (~0.17m error) was only `-0.0289` per step. The `alive` bonus (+1.0) completely overpowered it. The agent learned it was mathematically optimal to safely lay on the floor and collect +1.0 rather than risk falling.
* **Fix**: Increased the height penalty weight to `-10.0`.

### Phase 2: Introducing Locomotion
* **Goal**: Make the robot walk forward.
* **Changes**: 
  * Added `forward_velocity` (weight `+2.0`), scaling linearly with speed.
  * Added `lateral_velocity` penalty (weight `-0.3`) to prevent sideways drift.
  * Added `ang_vel_xy` penalty (weight `-0.5`) to prevent excessive rocking (pitch/roll).
  * Added `joint_torque_excess` penalty (weight `-0.05`, threshold `2.5 Nm`) to prevent the robot from thrashing its motors.
  * Random yaw initialization `[-π, π]` was strictly enforced so the robot learns to walk relative to its *own* heading, not the world's X-axis.

### Phase 3: One-Sided Height Penalty
* **Observed Behavior**: The L2 height penalty forced the robot to maintain an *exact* height. If it naturally stood taller during a gait, it was penalized.
* **Fix**: Replaced the L2 penalty with a custom `base_height_penalty`. 
  * The initial spawn height was adjusted to `0.12m`.
  * The penalty only applies if the root drops **below `0.08m`** (weight `-50.0`). 
  * The agent is now free to stand as tall as it wants (reward = 0 for height >= 0.08m) but is heavily punished for collapsing.

### Phase 4: The "Stand Still" Exploit
* **Observed Behavior**: The robot would take 2-3 steps, achieve a stable stance, and freeze.
* **Why**: Moving is risky and incurs joint velocity / action rate penalties. Standing still safely farms the `+1.0` alive bonus in perpetuity.
* **Fix**: Added a `stand_still_penalty`.
  * If forward velocity $v_x < 0.1 \text{ m/s}$, the agent receives a penalty.
  * The weight is set to `-10.0`. If $v_x = 0$, the penalty evaluates to $-1.0$, perfectly cancelling out the `+1.0` alive bonus. 
  * **Result**: Standing still yields a net reward of `0`. The robot is economically forced to move forward to accumulate a positive score.

---

## 3. PPO & Infrastructure Improvements

To support the complexity of a 6-legged robot (18 DOF, 63-dimensional observation space), standard Cartpole parameters were insufficient.

1. **Neural Network Capacity**:
   * Increased MLP hidden layers from `[32, 32]` to `[256, 128, 64]`.
2. **Observation Normalization**:
   * Enabled `obs_normalization=True` to maintain running statistics, crucial for stable training with complex joint state observations.
3. **Training Length**:
   * Increased `max_iterations` from `150` to `1500`.
4. **Developer QoL (`--resume_last`)**:
   * Modified `train.py` to include a `--resume_last` CLI flag. This automatically detects and loads the most recent timestamped run directory in `logs/rsl_rl/hexapod_flat/`, allowing us to rapidly tweak reward weights and resume training without manually copy-pasting folder names.

---

## 4. Future Development Steps

If the current reward structure results in a dragging or highly inefficient gait, consider the following future steps:

1. **Curriculum Learning**: Start by training the agent just to stand still and track a target height. Once mastered, smoothly interpolate the `forward_velocity` reward weight from `0.0` to `2.0`.
2. **Foot Clearance Reward**: If the robot drags its feet, add a penalty for the feet having negative Z-velocity when they are close to the ground, or a reward for tracking a specific foot swing trajectory.
3. **Explicit Tripod Gait Enforcement**: Add a reward term that maximizes the phase difference between specific sets of legs (e.g., Left-Front, Right-Middle, Left-Back vs Right-Front, Left-Middle, Right-Back).

---

## 5. Phase 5: The "Sliding" Exploit

### Observed Behavior
After introducing the `foot_contact_force` penalty (later removed due to ContactSensor segfault issues), the agent discovered that **sliding on its belly** was the highest-value strategy:
- `forward_velocity` rewards *any* CoM forward motion — sliding counts.
- With all joints frozen, `joint_torque_excess` fires zero penalty.
- `action_rate` fires zero penalty when actions don't change.
- `ang_vel_xy` fires zero penalty when flat on the ground.
- Result: The agent could earn ~+2.0/step by simply dragging itself, far cheaper than the torque cost of a proper gait.

### Root Causes
1. `forward_velocity` is body-frame CoM velocity — leg movement is **not required** to earn it.
2. `joint_torque_excess` was intended as a foot-force proxy, but dragging actually requires *less* torque than stepping, so the proxy fails.

### Fix (Phase 5)
Three changes were applied simultaneously:

**1. Height penalty tightened**
- `minimum_height` raised from `0.08 m` → `0.095 m` (fires earlier, giving the agent more time to react before the 0.05 m termination threshold).
- Weight increased from `-50.0` → `-60.0` for a stronger corrective gradient.
- The termination threshold at `0.05 m` is intentionally left alone — it is a physical floor, not a gait constraint.

**2. Foot clearance reward (Option A)**
- `foot_clearance_reward` reads `body_pos_w` for the six `*_foot_link` bodies.
- Reward = `clamp(foot_z, 0, 0.02 m)` per foot, normalised → max 1.0 per foot.
- Implemented **without ContactSensors** (uses articulation body positions directly, avoiding the USD segfault).
- Weight: `+0.8`. A sliding robot earns 0 from this term; a proper swing phase earns up to +4.8/step.

**3. Joint motion + acceleration reward**
- `joint_motion_reward` = `sum(|ω_j|) - 0.1 × sum(|α_j|)`.
- Rewards smooth continuous joint movement; penalises abrupt jerk so the agent can't exploit this by vibrating.
- Weight: `+0.05`. Modest contribution — acts as a tiebreaker between "all-frozen" and "leg-pumping" policies.

### Economic Analysis After Fix

| Strategy | foot_clearance | joint_motion | forward_vel | Net change |
|---|---|---|---|---|
| Sliding (belly) | 0 | ~0 (frozen joints) | +2.0 | **-4.8 vs. gait** |
| Proper tripod gait | up to +4.8 | +~0.14 | +2.0 | **clearly better** |

---

## 6. Phase 6: Persistent Backward Pitch

### Observed Behavior
After Phase 5 fixes the robot began moving with legs, but developed a sustained **backward pitch** (torso tilted rearward) that persisted throughout an episode.  The front four legs bore most of the load; the rear two frequently dragged a femur joint instead of foot-tips.

### Root Cause
`ang_vel_xy` (weight `-0.5`) penalises angular **velocity** (rate of rotation), not the **angle** itself.  A robot that pitches to 30° and then holds that angle perfectly still earns **zero** from that term — it's a rate sensor, not a level sensor.

### Fix

**New reward function: `base_orientation_penalty`** using `projected_gravity_b`.

For a level robot, the gravity vector in body frame is `[0, 0, -1]`.  The x/y components equal `sin(pitch)` and `sin(roll)` respectively, giving a static angle signal.

**Two-zone design** (weight `-5.0`, `soft_threshold_deg=8°`, `hard_multiplier=10`):

| Tilt | soft (`sin²θ`) | hard (`10 × excess`) | total | × -5 |
|---|---|---|---|---|
| 5° | 0.008 | 0 | 0.008 | **-0.04/step** |
| 8° | 0.019 | 0 | 0.019 | **-0.10/step** |
| 15° | 0.067 | 0.480 | 0.547 | **-2.74/step** |
| 25° | 0.179 | 1.600 | 1.779 | **-8.9/step** |

At 15° sustained pitch the orientation penalty already **exceeds the +2.0 forward velocity reward**, making backward lean economically irrational regardless of speed.  Small transient tilts during the step cycle (< 8°) cost barely anything, preserving natural gait dynamics.

### Environment Parameter Updates
Also applied in this phase:
- **`episode_length_s`**: 5 s → **10 s** (500 steps at 50 Hz) — longer episodes let the robot learn longer-horizon stable gaits.
- **`env_spacing`**: 5 m → **10 m** — reduces inter-robot interference over the longer travel distance.

---

## 7. Phase 7: Bent-Knee Local Optimum (~4500 iterations)

### Observed Behavior
After ~4500 training iterations with the Phase 6 reward set, **one front leg** developed a persistent pathological posture: the **femur (second-to-last joint) folded completely**, bringing the femur link into contact with the ground rather than the tibia tip. The overall robot motion degraded to a **slant or circular trajectory** — the asymmetric leg load caused a yaw bias that no amount of continued training resolved.

### Why Continued Training Failed to Fix It

This is a classic **RL local optimum** problem, not a data problem. By ~4500 iterations, the policy had converged to a stable attractor state. The gradient signal is insufficient to escape once the value estimates for the broken-leg posture are well-established in the critic. Simply training longer does **not** help — the policy hill-climbs toward the local optimum faster than the reward signal can steer it away.

Key factors that allowed the optimum to persist:

| Factor | Why it enabled the exploit |
|---|---|
| `foot_clearance_reward` (+0.8) | Targets tibia bodies — a folded femur can still produce tibia airtime on swing, earning partial clearance reward |
| `base_height_penalty` (min 0.095 m) | 0.095 m was achievable even with one leg folded — the remaining 5 legs can compensate |
| No per-link body penalty | Femur dragging on ground had **zero direct penalty** — only indirect cost via `base_orientation` if the torso tilted |
| Non-deterministic RL init | Different random seeds produce different local optima; this one was unlucky enough to lock in the bent-knee policy |

### General Principle: Reward Shaping vs. Retraining

A common instinct is to retrain from scratch with a different seed. This can work, but is unreliable — the same pathology can re-emerge in another run. **The correct fix is to make the local optimum economically unviable** via targeted reward shaping:

> *If a behaviour persists for thousands of iterations, the reward structure is not making it costly enough. Identify the exact physical state the exploit relies on, and add a direct penalty for that state.*

In this case: the exploit relies on a **femur body being near the ground**. Adding a direct penalty for femur proximity to the ground makes the broken posture unprofitable regardless of seed.

---

## 8. Phase 8: Height Increase + Non-Foot Body Penalty

### Changes Applied

**1. Raise `base_height` minimum: 0.095 m → 0.13 m**
- At 0.095 m, the robot could maintain sufficient height with one leg folded under.
- At 0.13 m, **all legs must extend significantly** to clear the threshold. A robot balancing on a folded femur cannot reach this height — the only viable strategy is proper leg extension.
- Economic impact: height error of 0.03 m (just under threshold) → `-60 × 0.03 = -1.8/step`. Exceeding the threshold is now a near-constant cost for the broken-leg posture.

**2. New `non_foot_body_ground_penalty` (weight `-30.0`)**
- Uses `body_pos_w` (same API as `foot_clearance_reward`) — **no ContactSensor required**.
- Targets all 6 femur bodies via `body_names=[".*femur.*"]`.
- Penalty fires when any femur Z-height drops below `0.05 m` (termination floor).
- Returns `sum(clamp(0.05 - femur_z, 0))` across all 6 femur links.
- **Economic analysis for bent-knee posture:**

| State | femur_z | penalty per femur | × weight -30 | vs. benefit |
|---|---|---|---|---|
| Normal standing | ~0.08 m | 0 (above threshold) | 0 | — |
| Mild droop | 0.03 m | 0.02 m | **-0.6/step** | wipes out joint_motion gain |
| Fully folded (ground) | ~0 m | 0.05 m | **-1.5/step** | far exceeds any leg-planted benefit |

- **Why not `.*coxa.*` too?** Coxa links are the first segment and naturally higher — including them risks spurious penalties during normal motion. Femur is the specific link implicated in the bent-knee pathology.

### Why This Works Where Retraining Might Not

The two changes together close the economic gap from **both sides**:
- Raising `base_height` to 0.13 m makes maintaining height with a folded leg **mechanically harder** (positive pressure to extend).
- The femur proximity penalty makes the folded-leg state **directly costly** regardless of height.

A retrain from scratch with these rewards in place will not allow the bent-knee optimum to form — the gradient signal correctly penalises it from the first episode.

---

## 9. Phase 9: Airborne Leg Exploit & Yaw Drift (~1000 iterations)

### Observed Behavior
With the height and bent-knee fixes in place, the hexapod learned a near-tripod gait very quickly (~1000 iterations). However, two new issues emerged:
1. **Permanently Airborne Leg**: The back-left leg remained stuck in the air, with the back-right leg barely moving.
2. **Yaw Drift**: The robot slowly rotated as it walked due to the asymmetric gait, meaning it traveled in a wide arc instead of a straight line.

### Issue 1: The "Hanging Toe" Local Optimum
The `foot_clearance_reward` used a pure position-based signal: `clamp(foot_z, 0, 0.02m)`. The agent discovered that simply lifting one leg and keeping it permanently at `>0.02m` earned maximum reward every single step without the effort of cycling it back down.

**The Fix:**
Replaced `foot_clearance_reward` with a hybrid `foot_gait_reward`:
- **70% Velocity-based**: `clamp(|foot_vel_z|, 0, 0.3 m/s)`
- **30% Position-based**: `clamp(foot_z, 0, 0.02m)`
A permanently airborne leg now has near-zero Z-velocity, capping its reward at `0.3` per step. A properly cycling leg earns up to `1.0` per step, heavily incentivizing the agent to keep all legs moving.

### Issue 2: Yaw Drift and "Forward" Ambiguity
The `forward_velocity` reward measured velocity in the *body frame's X-axis*. If the robot started walking in a circle, its body-frame forward velocity was still high, so it had no incentive to straighten out (there was no direct penalty for yawing). 

**The Fix:**
Added `forward_distance_x` as the new **primary positive reward** (weight `+2.5`) and `distance_y` as its **negative counterpart** (weight `-2.5`).
- `forward_distance_x` measures linear velocity in the **absolute world-frame X-axis**.
- `distance_y` measures linear velocity in the **absolute world-frame Y-axis**.
- Because the robot's initial yaw is fully randomized `[-pi, pi]` every episode, it must immediately turn to face the world +X direction and maintain that heading.
- If the robot's gait is asymmetric causing it to rotate or drift diagonally, the `distance_y` penalty will cancel out the `forward_distance_x` reward, strongly penalizing slanted movement and forcing a straight path.
- Retained the body-frame `forward_velocity` (weight `+2.0`) as a secondary local reward to encourage forward effort even while turning.

### Outcome: Failure of Velocity-Based Constraints
The strategy implemented in Phase 9—specifically the heavy reliance on global world-frame velocity constraints to fix gait issues—was a **failure**.
- **Non-convergence**: The policy often failed to converge entirely. The heavy punishment from the `distance_y` penalty and strict global directional requirements created harsh loss landscapes where the agent simply got stuck.
- **Pathological Gaits**: When the policy did manage to converge, the resulting behavior was worse than before. The robot adopted a severely broken gait using only 2 or 3 legs, perpetually rotating in a small circle rather than achieving the intended stable, straight-line walking.

Adding strict global velocity rewards conflicted with the natural exploration of complex legged locomotion, trapping the agent rather than guiding it.
