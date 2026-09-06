# Hexapod RL Locomotion: Development & Reward Tuning Log

This document tracks the progress, errors, fixes, and rationale behind the reinforcement learning reward shaping for the Interbotix PhantomX Mark IV Hexapod in Isaac Lab. 

## 1. Environment Setup & Critical Bugs Resolved

During the initial setup of the `Hexapod-Flat-v0` environment, we encountered several framework-level crashes and bugs.

### Bug 1: `base_contact` Termination Crash
* **Error**: `ValueError: The term 'base_contact' expects mandatory parameters...`
* **Root Cause**: The template code used `mdp.base_height_l2` as a termination condition. However, `base_height_l2` is a **reward function** that returns a continuous tensor, not a boolean termination function.
* **Fix**: Replaced it with the correct built-in termination function: `mdp.root_height_below_minimum` with a threshold of `0.05m`.

### Bug 2: `ContactSensor` Setup & Initialization Issues
* **Initial Error**: Exit code 139 segfaults and `RuntimeError: Failed to initialize contact reporter for specified bodies`.
* **Root Causes**: 
  1. **Missing Contact API Schema in USD**: PhysX contact reporting requires the `PhysxContactReportAPI` schema applied directly to the collision rigid bodies in the USD stage (`hexapod.usda`).
  2. **Regex Over-Matching**: Prim paths like `.*tibia.*` or `.*_tibia_link.*` matched both top-level rigid bodies and nested non-rigid collision mesh prims (e.g., `pxmark4_tibia_link_1`), which caused PhysX tensor reporter initialization to crash.
* **Fix**:
  1. Applied `PhysxContactReportAPI` across all tibia links in `hexapod.usda`.
  2. Added `ContactSensorCfg` to `HexapodSceneCfg` in `hexapod_env_cfg.py` using the precise rigid body regex `"{ENV_REGEX_NS}/Robot/.*tn__pxmark4.*_tibia_link.*"` (targeting exactly the 6 rigid tibia links).

### Bug 3: `body_contact_forces` Attribute Error (Legacy Approach)
* **Error**: `AttributeError: 'ArticulationData' object has no attribute 'body_contact_forces'`
* **Root Cause**: The installed version of Isaac Lab's `ArticulationData` did not expose `body_contact_forces` directly.
* **Fix**: Replaced by standard `ContactSensorCfg` in `InteractiveSceneCfg`, which exposes clean `net_forces_w`, `last_air_time`, and contact histories.
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

---

## 10. Phase 10: Tibia Foot Contact Sensor Integration & CPG Foundation

To move toward **Central Pattern Generator (CPG)** guided reward formulation and avoid the pitfalls of naive velocity-based constraints observed in Phase 9, accurate ground contact sensing for all 6 feet (tibias) was successfully integrated.

### 1. USD Setup & Scene Configuration

* **USD Prim Schema (`hexapod.usda`)**:
  `PhysxContactReportAPI` is prepended to the rigid body scopes for all 6 legs:
  - `tn__pxmark4right_front_tibia_link_ri0`
  - `tn__pxmark4right_middle_tibia_link_ck0`
  - `tn__pxmark4right_back_tibia_link_bh0`
  - `tn__pxmark4left_front_tibia_link_bh0`
  - `tn__pxmark4left_middle_tibia_link_ri0`
  - `tn__pxmark4left_back_tibia_link_qf0`

* **Isaac Lab Scene Integration (`hexapod_env_cfg.py`)**:
  ```python
  from isaaclab.sensors import ContactSensorCfg

  # Inside HexapodSceneCfg:
  contact_forces = ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/Robot/.*tn__pxmark4.*_tibia_link.*",
      history_length=3,
      track_air_time=True,
  )
  ```

### 2. Contact Point Grounding: Why Tibia Sensing Measures Toe Contact

In the URDF / USD kinematic tree:
1. Each leg's lower link is a single rigid body (`pxmark4/xxx_tibia_link`) containing the collision geometry `pxmark4_tibia_link.stl`.
2. The distal tip (toe) of the tibia link corresponds to the `foot_link` origin at $(x=0.095, y=0, z=-0.09)\text{ m}$ relative to the tibia joint.
3. Because the tibia link is angled downward during standing and locomotion, the lowest geometric collision point contacting the ground surface is the **curved rubber toe tip**.
4. PhysX computes collision pairs between this convex collision hull and the ground plane. When the toe tip touches the ground, PhysX registers the normal force on the tibia rigid body. If the robot stumbles or drags its shank, contact is also registered on the tibia body, physically capturing contact anomalies.

### 3. Verification: Free-Fall Drop Test ($0.60\text{ m}$)

To verify sensor operation and dynamic sensitivity, the robot was dropped from a spawn height of $0.60\text{ m}$ in simulation ($dt = 0.02\text{ s}$):

* **Free-Fall ($0.00\text{ s} - 0.26\text{ s}$)**: Contact forces are strictly $0.00\text{ N}$ across all 6 feet.
* **Impact Spike ($0.28\text{ s} - 0.30\text{ s}$)**: Contact force peaks at **$48.90\text{ N}$** on individual feet, with a total impact force of **$276.11\text{ N}$**.
* **Standing Equilibrium ($0.80\text{ s} - 2.00\text{ s}$)**: Forces settle to $\approx 2.0\text{ N} - 2.2\text{ N}$ per foot, yielding a total normal force of **$\approx 11.8\text{ N}$** (matching the $1.2\text{ kg} \times 9.81\text{ m/s}^2$ weight of the robot).

```text
Time (s)   | Step  | Max Z-Force (N) | Total Z-Force (N)  | Per-Leg Z-Forces [RF, RM, RB, LF, LM, LB]
-------------------------------------------------------------------------------------------------
    0.02 s |     0 |            0.00 |               0.00 | [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00]
    0.28 s |    13 |           36.46 |             215.18 | [ 35.92,  36.20,  36.46,  35.55,  35.10,  35.95]
    0.30 s |    14 |           48.90 |             276.11 | [ 43.91,  43.24,  44.92,  46.73,  48.90,  48.40]
    0.40 s |    19 |            1.57 |               7.93 | [  1.20,   1.38,   1.05,   1.44,   1.57,   1.29]
    1.00 s |    49 |            2.20 |              11.92 | [  2.03,   2.20,   2.11,   1.83,   1.86,   1.89]
    2.00 s |    99 |            2.19 |              11.74 | [  2.03,   2.19,   2.01,   1.79,   1.86,   1.87]
```

### 4. Planned Usage of Contact Sensory Data in RL

With working contact sensors, the RL pipeline can leverage four primary mechanisms:

1. **CPG Gait Phase Tracking (Stance vs. Swing Synchronization)**:
   Given target CPG phases $\phi_i \in [0, 2\pi)$ for each leg $i \in \{1 \dots 6\}$:
   - Reward contact during planned stance phase ($\phi_i \in [\pi, 2\pi)$) where $F_{z, i} > F_{\text{threshold}}$.
   - Penalize contact during planned swing phase ($\phi_i \in [0, \pi)$) where $F_{z, i} > 0$.
   This directly eliminates both the "permanently airborne leg" and "dragging leg" exploits without rigid global velocity penalties.

2. **Air-Time Consistency Reward**:
   Using `contact_forces.data.last_air_time`, reward symmetric swing duration across opposing tripod pairs ($\{\text{RF, LM, RB}\}$ vs $\{\text{LF, RM, LB}\}$).

3. **Impact Force Penalization**:
   Penalize excessive peak contact forces ($F_z > 15\text{ N}$) to encourage smooth foot placement over slamming.

4. **Policy Observation Input**:
   Expose binary contact booleans or normalized contact forces ($6\text{D}$ tensor) into the policy observation space (`ObsGroup`) for reactive terrain response.

### 5. Implementation Progress: CPG Curriculum Integration

**Current Status:**
- Implemented `cpg_reward` in `mdp/rewards.py` utilizing the time-based `cpg_clock` mechanism. It penalizes ground contact during the planned swing phase and rewards contact during the planned stance phase (with an alternating tripod setup offset by `0` and `pi`).
- Successfully exported `cpg_reward` in `mdp/__init__.py` after a test run failed with `AttributeError: module 'hexasim.tasks.manager_based.hexasim.mdp' has no attribute 'cpg_reward'`.
- Verified the RL training script `train.py` correctly parses the custom CPG term.
- Set up a transition curriculum (`CurriculaCfg`) in `hexapod_env_cfg.py` to fade out scaffolding rewards and introduce the CPG rewards after the initial gait is learned (at step 72,000 or ~3,000 iterations):
  - `foot_clearance`: drops to `0.0`
  - `joint_motion`: drops to `0.0`
  - `cpg`: activates with weight `+2.0`
- `base_height` curriculum testing was successful and its weight has been reverted to the static `-60.0` to maintain posture constraints throughout training.



The curriculum should remove the previous basic rewards of joint_motion and foot_clearance as soon as the CPG rewards are introduced.

The CPG reward weightage should be increased over 150 iterations, to prevent catastrophic forgetting.

### 6. Phase 10b: Empirical Ground-Offset Calibration of `cpg_z_target_reward`

The `cpg_z_target_reward` term assumed the tibia rigid-body origin sat $0.02\text{ m}$ above the ground at toe contact (`target_z_abs = target_z + 0.02`). This was unverified.

**Measurement method (slow quasi-static descent):**
- Robot spawned at $0.30\text{ m}$; root lowered in $2.5\text{ mm}$ increments with velocities zeroed each step.
- Tibia body-origin Z recorded alongside `ContactSensor` Z-forces; the first force spike ($>1\text{ N}$) marks exact toe/ground contact.

**Result — first-contact offset per leg:**

| Leg | Offset |
|---|---|
| right_front | 0.08923 m |
| right_middle | 0.08922 m |
| right_back | 0.08921 m |
| left_front | 0.08922 m |
| left_middle | 0.08922 m |
| left_back | 0.08921 m |

Mean = **$0.0892\text{ m}$**, inter-leg spread $< 0.1\text{ mm}$. All six toes touch simultaneously — confirming a level stance geometry.

**Fix:** `target_z_abs = target_z + 0.02` → `target_z_abs = target_z + 0.0892` in `mdp/rewards.py`. The old value made every planted foot carry a constant $(0.0892 - 0.02)^2/\sigma^2$ error, so the Gaussian reward was near-zero exactly when it should have been maximal.

Caveat: this offset is exact for the default rest pose ($\text{femur}=30°$, $\text{tibia}=-30°$); it shifts slightly with joint angles during active gait, but remains the correct stance-phase baseline since the CPG swing trajectory is defined relative to ground contact.

### 7. The Force Penalty Bug: The "Floor is Lava" Exploit

`cpg_force_target_penalty` originally used **Mean Squared Error (MSE)** on physics contact forces:

```python
penalty = torch.square(error)   # REMOVED
```

Squaring contact forces in a rigid-body simulator is catastrophic. Isaac Lab routinely produces transient micro-collision spikes of $20\text{ N} - 40\text{ N}$ from slight over-extensions or foot slams.

**Economic asymmetry that broke the policy:**

| Behaviour | Force error | MSE penalty | × weight -0.5 |
|---|---|---|---|
| Leg slams at 22.35 N | 20 N | $20^2 = 400$ | **-200 / step** |
| Leg hovers in air | 2.35 N | $5.5$ | **-2.75 / step** |

A single physics spike cost ~73× more than permanently suspending the leg. The agent rationally concluded that keeping legs safely airborne (or grazing the ground at $<2\text{ N}$) was infinitely cheaper than risking a 400-point nuke on impact — recreating the "permanently airborne leg" exploit from Phase 9 through an entirely new mechanism.

**Fix:** switched to **Mean Absolute Error ($L_1$)**:

```python
penalty = torch.abs(error)
```

$L_1$ yields a constant-magnitude linear gradient that guides the leg down toward its target force ramp without exploding on transient impact spikes.

**General principle:** never square raw simulator signals with unbounded transients (contact forces, torques, impulses). Squaring is only safe for bounded, smooth quantities (e.g., joint position errors). Reserve $L_2$ shaping for states, $L_1$ for physics measurements.

### 8. Phase 10c: The Gaussian Dragging Exploit in `cpg_z_target_reward`

**Diagnosis from TensorBoard:** the `cpg_z_target` reward curve was a *mathematically perfect straight line* rising in lockstep with its curriculum weight ramp. Neural-network convergence is never perfectly linear — a straight reward curve that tracks the scheduler means the policy is earning a **static score** and the curriculum multiplier is doing all the movement. The pink line was not "learning to lift legs"; it was the anneal multiplying a guaranteed baseline.

**The loophole:** the Gaussian $\exp(-\text{err}^2/\sigma^2)$ was evaluated across the **entire CPG cycle**. With $\beta = 5/6$, stance occupies $83.3\%$ of every cycle, during which the target is $z_{\text{target}} = 0 + 0.0892$. A planted (or dragging) foot sits at exactly $0.0892\text{ m}$:

| Behaviour | Phase | Error | Gaussian | Share of cycle |
|---|---|---|---|---|
| Foot on ground, doing nothing | stance | $0$ | **1.0** | $83.3\%$ |
| Foot on ground while others swing | swing | $-0.05$ | $\approx 0.002$ | $16.7\%$ |

A do-nothing dragging policy guarantees a mean score of $\approx 0.83$ of maximum — an A-grade for sitting on the floor. Swinging legs risked instability to chase the remaining $17\%$. The rational policy dragged.

**Fix (`mdp/rewards.py`, `cpg_z_target_reward`):**
1. **Stance contributions zeroed**: per-leg Gaussian is multiplied by the swing mask — stance-phase reward is exactly $0$ regardless of foot position. Ground contact during stance is the force penalty's job; kinematic tracking applies only to swing.
2. **Swing-count renormalisation**: the sum is divided by the *current* number of legs in swing (clamped to ≥ 1), keeping the reward scale in $[0, 1]$ as the gait moves between double-support and single-tripod phases instead of collapsing with swing count.

The only way to earn from this term is now physically lifting a leg along the sine wave during its own swing window. The dragging exploit is mathematically eliminated.

### 9. Curriculum Handover at Step 4500

All transition schedules were unified around the CPG takeover at **step 4500**:

| Term | From → To | Window |
|---|---|---|
| `foot_clearance` | $+0.8 \to 0.0$ | 4500 → 5000 |
| `joint_motion` | $+0.05 \to 0.0$ | 4500 → 5000 |
| `cpg_z_target` | $0.0 \to +2.0$ | 4500 → 40000 |
| `cpg_force_target` | $0.0 \to -0.5$ | 4500 → 40000 |
| **`forward_velocity`** *(new)* | $+2.0 \to +1.0$ | 4500 → 6000 |

**Rationale for the new `forward_velocity` anneal:** once CPG phase-tracking dominates, the raw velocity incentive must yield so gait-quality gradients are not drowned out — but it is halved rather than removed, preserving the drive to translate forward. The slow 1500-step fade avoids a step change in the reward landscape that would shock the critic mid-training.

### 10. Phase 11: Gait-Math Alignment — Duty Factor Contradiction & Adaptive Baseline

**Diagnosis:** the CPG reward stack contained a *fatal internal contradiction*. `cpg_z_target_reward` and `cpg_force_target_penalty` used `duty_factor = 5/6` (a wave-gait duty factor) while `HexapodEnv._cpg_base_offsets` used alternating-tripod offsets `[0, π]`. The consequence is exact: Tripod A swings $[0°, 60°)$, Tripod B swings $[180°, 240°)$ — so for the remaining **66.6% of every cycle all six legs are commanded to be planted simultaneously** (6-leg double-support), while `forward_velocity` (+2.0) demands continuous fluid motion. Lifting three legs at once for only a short window while pushing forward causes instability; the policy resolves the contradiction by abandoning the Z-target entirely, eating the force penalty, and dragging its feet to farm forward velocity.

**Fixes applied (all three must ship together — they are one gait):**

1. **`duty_factor` 5/6 → 0.5** in both CPG rewards (and made explicit in `RewardsCfg.params`). With offsets `[0, π]`, swing windows become Tripod A $[0°, 180°)$ and Tripod B $[180°, 360°)$: exactly 3 legs swing at every instant, no double-support dead zone. Bonus: each swing now lasts 500 ms (25 control steps at 50 Hz) instead of ~8, making the sine-tracking task far more learnable at the existing 1.0 Hz clock.
2. **`max_force` 2.35 N → 4.0 N.** Measured robot weight is 11.8 N; with only 3 tripod legs supporting, each stance foot carries ≈ 3.93 N. The old 2.35 N target punished *correct* mid-stance loading as an error.
3. **Fixed foot baseline → adaptive contact baseline.** `target_z_abs = target_z + 0.0892` assumed the tibia origin rests exactly 0.0892 m above ground — true only for the default rest pose. Any torso sag (or non-zero-terrain use) lowers the physical foot below that constant, making targets unreachable no matter how hard the policy tries. Replacement: `HexapodEnv.cpg_foot_baseline`, a per-leg $(num\_envs \times 6)$ EMA ($\alpha = 0.2$) of tibia world-Z updated **only when that foot registers > 1 N contact**, refreshed after physics each step and reset to 0.0892 on episode reset. Swing targets are then `baseline + target_z` per leg — self-calibrating to whatever height the torso currently holds the standing feet at. Reward functions fall back to the static 0.0892 offset if the attribute is absent (non-HexapodEnv harnesses).

### 11. Phase 12: Adaptive CPG Frequency & Coxa Velocity Reward

**Diagnosis:** Despite the tripod gait math fixes, the hexapod policy remained stuck in a local optimum (4 legs moving minimally, 2 legs airborne). Lingering CPG issues and lack of explicit forward stride incentives were identified as the cause. Additionally, the fixed CPG clock (1.0 Hz) was fighting the agent's natural learned velocity, holding it back.

**Fixes applied (Two-Phase Testing):**

1. **Pure CPG Baseline (Phase 1):** We completely removed the `cpg_z` and `cpg_force` curriculum terms. This activates the `cpg_z_target` (weight 2.0) and `cpg_force_target` (weight -0.5) rewards from step 0, forcing the policy to experience pure CPG shaping immediately rather than getting trapped in an early dragging optimum before step 4500.

2. **Adaptive CPG Phase Increment (Phase 2):** The hardcoded `f = 1.0` clock in `HexapodEnv._step_impl` was replaced with a dynamic frequency based on the physical forward velocity: `f = v_x / 0.2`, clamped to `[0.2, 2.5]`. This scales the clock with the robot's stride (assumed nominal 0.2m), preventing the clock from fighting the agent's velocity.

3. **Coxa Velocity Reward (Phase 2):** To provide a direct incentive for sweeping legs forward during swing, we added `cpg_coxa_velocity_reward` targeting the `.*_coxa` joints. It rewards absolute angular velocity during the swing phase (positive multiplier) and penalizes it during the stance phase (negative multiplier). Weight: 1.0.

### 12. Phase 13: Curriculum Reinstatement (Delayed CPG Introduction)

**Diagnosis:** The Phase 12 pure CPG baseline (introducing CPG constraints from step 0) failed to produce significant improvement over the old dragging behavior. The policy cannot handle the complex rhythmic constraints before it has learned basic stability (i.e., until the mean episode length reaches the maximum 500 steps/10 seconds).

**Fixes applied:**
We reinstated the two-stage curriculum approach, but shifted the transition from step 4500 to step 3000 (since 3000 is enough time to learn to stand and move forward without falling).

1. **Base Rewards (0 → 3000):** `foot_clearance` (0.8) and `joint_motion` (0.05) remain active early to enforce basic smooth stepping, then anneal to 0.0 by step 3500.
2. **CPG Introduction (3000 → 8000):** `cpg_z_target` (0.0 → 2.0), `cpg_force_target` (0.0 → -0.5), and `cpg_coxa_velocity` (0.0 → 1.0) all linearly scale up together starting at step 3000, "taking over" the gait once the policy is already capable of surviving the full 500-step episode.
3. **Velocity Fade (3000 → 4500):** `forward_velocity` incentive fades from 2.0 to 1.0 simultaneously to allow the CPG gradients to dominate.

## 13. Phase 14: Analysis of Local Optima and Proposed CPG Refinements

### Observed Behavior
Despite the reinstatement of the CPG curriculum, the policy remains stuck in a local optimum. The robot continues to rely on 5 legs, with the back-left leg mostly hanging in the air and moving ~1-2cm from the ground with no contribution. The expected textbook tripod gait (lifting legs significantly during swing and pushing during stance) is absent.

**Key Insights from Current Weights:**
1. **Coxa Velocity Optimization:** The `cpg_coxa_velocity_reward` is functioning correctly (mean ~8). The policy optimizes leg sweeping, demonstrating that phase-based joint targets *are* learnable.
2. **CPG Z and Force Ineffectiveness:** The `cpg_z_target` (mean ~2) and `cpg_force_target` (mean ~-2.5) have marginal impacts. The policy avoids lifting the leg entirely, thereby bypassing the CPG Z target sine wave and avoiding the large force penalties that would result from placing a lifted foot back down incorrectly.

### Diagnosis
The core issue is that Cartesian Z-height targets (`cpg_z_target`) and global velocity constraints are easily exploited by a policy that learns to keep legs mostly planted or permanently airborne. The policy finds a "safe" lazy gait that satisfies forward velocity and coxa sweeping without the risk of lifting and replacing the feet.

### Proposed Reward Refinements (To Be Implemented)
To explicitly force the robot out of this local optimum, we must enforce the mechanical reality of walking at the joint level, rather than just the Cartesian level. 

1. **Joint-Space Swing Trajectory (Femur Target):**
   *   **Concept:** Instead of rewarding Cartesian Z-height (which is exploited by keeping a leg permanently airborne), we will reward the *femur joint* for following a targeted trajectory. 
   *   **Mechanism:** Penalize the deviation of the femur joint position from a nominal swing sine-wave trajectory during the swing phase. This directly forces the physical hardware to lift the leg and lower it back down.

2. **Contact-Conditioned Propulsion (Coxa):**
   *   **Concept:** Currently, `forward_velocity` rewards global movement, and `cpg_coxa_velocity` rewards leg sweeps. A lazy policy can sweep its coxa in the air while dragging itself forward. 
   *   **Mechanism:** We will modify the reward to encourage the coxa to move backward (propulsion) *only* when the corresponding foot is in physical contact with the ground during the stance phase. This links propulsion effort directly to physical grip.

3. **Stance Rigidity Penalty:**
   *   **Concept:** Legs should act as rigid struts during the stance phase. 
   *   **Mechanism:** Add a penalty for any velocity in the femur and tibia joints during the stance phase. This prevents the robot from buckling, vibrating, or slowly sliding its legs when they are supposed to be firmly planted and carrying the robot's weight.

### 14. Phase 15: The Frozen CPG Clock Bug

**Discovery:**
While running a detailed offline debug script (`play_debug.py`) to analyze the CPG trajectory, a catastrophic bug was discovered: the CPG phase clock was **completely frozen** at `0.0` and `3.1415927` for the entire duration of the episode.

**Root Cause:**
In Phase 12, the dynamic CPG update logic (scaling the clock frequency with forward velocity) was implemented by overriding a method named `def _step_impl(self, action):` in `hexapod_env.py`. However, unlike standard OpenAI Gym setups, Isaac Lab's `ManagerBasedRLEnv` class does **not** have a `_step_impl` method; it uses `def step(self, action):`. Because of the incorrect method name, the Isaac Lab environment completely ignored our custom override during the entire training run, and the phase clock was never updated after initialization.

**Impact on the Policy:**
Because the clock never ticked, the legs were statically assigned a permanent phase from step 0:
*   **Tripod A (Phase 0.0)** was permanently frozen in the **swing phase**. The policy was continuously commanded to hold these 3 legs in the air forever. 
*   **Tripod B (Phase 3.14)** was permanently frozen in the **stance phase**. The policy was continuously commanded to plant these 3 legs and push forever.

This completely explains the asymmetric, dragging policy observed in Phases 13 and 14:
*   The robot learned to permanently lean all its weight onto Tripod B (explaining the massive 5-9 N forces on those specific legs).
*   It let Tripod A drag or hover (explaining the 0.01 N force on Leg 5). 
*   It never learned a gait because the target sine wave never actually moved. The agent was being penalized by perfectly valid reward math applied to a completely broken, static clock.

**Fix:**
Renamed `def _step_impl` to `def step` inside `hexapod_env.py` and updated the `super()` call accordingly. The CPG clock will now tick correctly in the next training run.

**The Tensor Crash Anomaly:**
When `step` finally ran, code crashed on `net_forces_w_history[:, :, 0, :]` slice. Why? Bug was always there. `_step_impl` was dead code, so crash hid. Renamed to `step` -> code ran -> immediate shape mismatch crash.
Fixed by using current force: `net_forces_w`.

### 15. Phase 16: The "2 Airborne, 4 Dragging" Exploit (Reward Economy)

**Observed Behavior:**
After fixing the clock, the policy plateaued quickly (~1000 iterations) with a mean reward of ~60. The robot kept 4 legs planted and dragged them, while holding 2 legs permanently airborne. The CPG Z target reward and force penalties were completely ignored.

**The Reward Economy (Why it happens):**
With the clock moving, the agent must either learn to perfectly follow a 5cm sine wave, or find the cheapest way to ignore it. 
1. **Airborne Leg:** Incurs a stance force penalty (misses the 4N push target), but easily stays near the Z target sine wave during swing. Net cost: ~ `-0.38` per step.
2. **Dragging Leg:** Incurs a swing force penalty (applies force when it shouldn't) and misses the Z target completely. Net cost: ~ `-1.28` per step.

**The Local Optimum:**
Being airborne (-0.38) is mathematically cheaper than dragging (-1.28). The agent wants *all* legs airborne to minimize penalties! However, lifting too many legs causes the robot to fall, triggering the massive `-60` `base_height` penalty. 
The agent calculates the cheapest survival strategy: plant exactly 4 legs to prevent falling (absorbing the heavy dragging penalty), and hold the remaining 2 legs airborne to farm the cheaper -0.38 rate. 

The `cpg_z_target` reward had a `sigma=0.02`. If a dragging leg tried to lift 1cm, its error went from 5cm to 4cm. The Gaussian reward went from ~0.001 to ~0.018. The gradient was virtually zero, teaching the policy that lifting was impossible before it could reach the peak.

**Fixes Implemented (Phase 14 applied):**
Replaced the broken Gaussian Z-reward with direct joint-level L1 penalties to provide a constant gradient that crushes this exploit:
1. **`cpg_femur_target_penalty`:** Replaced `cpg_z_target`. Applies an L1 penalty on the femur joint deviating from a sine wave during swing. L1 provides a constant gradient; if the leg is wrong, it always feels a push in the exact right direction. Target weight: `-2.0`.
2. **`cpg_coxa_contact_reward`:** Replaced `cpg_coxa_velocity`. Rewards backward coxa velocity *only* if the foot's contact force > 1N during stance. Ensures sweeping actually propels the robot. Target weight: `1.0`.
3. **`cpg_stance_rigidity_penalty`:** Penalizes femur and tibia velocity during stance, preventing buckling or sliding. Target weight: `-0.5`.

### 16. Phase 16 Results: The Femur-Tibia Compensation Exploit

**Observed Behavior:**
The Phase 14 reward changes (specifically replacing the Cartesian Z-height target with a joint-space femur target) resulted in a degraded policy that plateaus around a mean reward of ~60. The robot exhibits a 6-leg dragging behavior, heavily optimizing the CPG coxa sweep while ignoring the mechanical intent of the femur target.

**Trajectory Analysis:**
- Mean Z-heights for all 6 legs are firmly between 0.080m and 0.102m (ground level).
- Contact forces for all legs average > 3.1N, proving all 6 legs are continuously planted.
- Coxa joints sweep nearly 60 degrees (~1.0 rad), confirming the policy is aggressively farming the `cpg_coxa_contact` reward.
- Femur joints *do* move (range ~0.42 rad), but the feet never lift.

**The Exploit:**
Controlling only the femur joint opened a new kinematic loophole. The policy learned to rotate the femur to satisfy the `cpg_femur_target_penalty`, but simultaneously extended the tibia joint downwards to keep the foot planted. This allows it to farm the femur reward without risking the `-60` fall penalty that comes with actually lifting a foot.

**Why not a Cartesian Z-height L1 penalty?**
The joint-space femur target was originally introduced because a Cartesian Z-height target can often be exploited by tilting the entire torso (pitch/roll) to artificially raise a foot relative to the ground. However, since a strict `base_orientation_penalty` was introduced in Phase 6, the torso-tilt exploit is securely blocked. 

Therefore, an L1 Cartesian Z-height penalty (`weight * abs(target_z - current_z)`) is now mathematically superior to the femur target. It provides a constant gradient that guarantees the foot leaves the ground regardless of what the individual joints do, without the dead-zone issues of the older Gaussian reward.

### 17. Phase 17: The Cartesian L1 Revert

**Diagnosis & The Exploit:**
The Phase 16 setup is a critical success in one regard: it is the only setup so far that successfully prevented the permanently airborne leg exploit and forced almost all legs to participate in movement. The robot established a solid foundation of walking on all six legs rather than cheating with a 4-drag/2-hover broken stance.

However, the policy still lacked proper vertical swing (ground clearance). It was discovered that the policy was not "farming" the femur target (which is a penalty), but rather it was simply accepting and eating the mild penalty (plateauing at ~-1.7). Economically, eating a -1.7 femur penalty and continuing to drag the leg was vastly cheaper than lifting the foot and risking a massive `-60` fall/height penalty if the base height dropped.

**Fix:**
To add the missing vertical ground clearance while keeping all legs active, we reverted to tracking the foot's Cartesian Z-height, but this time using a constant-gradient L1 penalty (MAE) rather than the exploitable Gaussian. 
1. **Removed `cpg_femur_target_penalty`**.
2. **Implemented `cpg_z_target_l1_penalty`**: Applies `abs(foot_z - target_z_abs)` only during the swing phase, with a heavily increased weight of `-20.0` in the curriculum (`cpg_z`). This massive penalty guarantees that dragging a swinging foot results in a cost (~-2.0 per step) that heavily outweighs the "safety" of keeping the leg planted to avoid a minor height drop.

### 18. Phase 17 Results: The Minimal Error and Pre-CPG Hovering Exploit

**Observed Behavior:**
Despite the heavy curriculum weight of `-20.0` for `cpg_z_target_l1_penalty`, the policy plateaued without lifting its legs. The robot reverted to keeping a leg permanently airborne (before CPG activation) and dragging the remaining legs. The user noted the CPG Z target reward converged at merely `~-1.6`, indicating the error signal was too minimal to drive behavioral change.

**Diagnosis 1: The Pre-CPG Hovering Exploit**
The policy learns to keep a leg permanently airborne *before* the CPG curriculum activates (around step 3000). The `foot_clearance` reward active in the early curriculum was reverted from the Phase 9 hybrid approach back to a pure position-based `clamp(foot_z - ground_z, 0, 0.02)`. This provides a continuous `+0.8` reward for simply hanging a leg 2cm in the air permanently. The agent locks into this "hovering toe" local optimum because it is perfectly stable and highly rewarded, rendering it unprepared for the periodic sine-wave constraints introduced later.

**Diagnosis 2: The Minimal Z-Error Math**
The `cpg_z_target_l1_penalty` calculates error in absolute meters: `abs(foot_z - target_z_abs)`.
- The target swing trajectory (`step_height`) peaks at `0.05m` (5 cm).
- If a foot completely ignores the swing and drags on the ground, its maximum error is exactly `0.05m`.
- The average error over the half-cycle swing phase is `~0.0318m` per leg.
- With 3 legs swinging, the total raw error is `~0.095m`. 
- Multiplied by the `-20.0` weight, the policy receives a total penalty of exactly `~-1.9` (matching the user's `~-1.6` observation).

The Z-target gradient is mathematically dwarfed. The agent rationally chooses to "eat" the tiny `-1.6` penalty because properly lifting the legs risks instability, which triggers the massive `-60.0` base height penalty.

**Proposed Corrections (To Be Implemented):**
1. **Normalize the CPG Z Error:** In `cpg_z_target_l1_penalty`, the physical meter error must be normalized by the `step_height`: 
   `error = torch.abs(foot_z - target_z_abs) / step_height`
   This converts the error from tiny physical meters `[0, 0.05]` to a normalized ratio `[0, 1.0]`. A completely ignored swing will then generate a massive `-38.0` penalty (average across 3 legs), providing an unmistakable gradient that forces the agent to comply.
2. **Fix Pre-CPG `foot_clearance`:** Address the early curriculum hovering exploit by either reinstating the hybrid velocity/position reward from Phase 9 or heavily scaling down the position-only reward so the agent doesn't enter the CPG curriculum already committed to a broken 5-leg stance.

### 19. Phase 17b: Normalized CPG Z Error Correction

**Fix Implemented:**
Following the diagnosis of the "Minimal Error Exploit" in Phase 17, the `cpg_z_target_l1_penalty` mathematical calculation was updated.

**Changes in `mdp/rewards.py`:**
The absolute physical error in meters is now normalized by dividing by the `step_height` (0.05m):
```python
error = torch.abs(foot_z - target_z_abs) / step_height
```

**Expected Outcome:**
Instead of generating a tiny raw penalty (e.g., `-0.05` * `weight`), the penalty now correctly acts as a ratio `[0, 1.0]`. If a leg stays perfectly grounded and ignores the `0.05m` swing apex, the normalized error peaks at `1.0`. Multiplied by the `-20.0` weight, this yields a robust, inescapable penalty gradient (averaging `~-38.0` total across all swinging legs). The agent will now find it mathematically necessary to clear the ground to avoid this massive loss.

### 20. Phase 17c: Debugging the Curriculum Trap

**Observed Behavior:**
After training with the normalized `cpg_z_target_l1_penalty` (Phase 17b), the policy *still* refused to lift its legs properly, retaining the 5-legs-grounded, 1-leg-airborne gait. 

**Debug Trajectory Analysis:**
We ran `play_debug.py` with expanded logging to trace the exact internal CPG math. The normalized error logic is functioning perfectly and generating massive gradients, but the policy is choosing to ignore them.

*   **Leg 4 (Dragging):** Bearing heavy load (`8.78 N`). Because it stays planted at `0.094 m`, its normalized error against the swing target is huge, resulting in a mean penalty of **`-6.73` per step**.
*   **Leg 5 (Permanently Airborne):** Hovering at `0.113 m` with `0.06 N` force. Because it is already elevated, its error against the swing target is smaller, yielding a penalty of only **`-3.22` per step**.
*   **Total CPG Z Penalty:** The combined penalty across all swinging legs averages **`~-30.0` per step**.

**Diagnosis: The Curriculum Trap**
If the policy is willing to absorb a massive `-30` penalty every step, it means the alternative (obeying the CPG and lifting the dragging legs) is perceived by the critic network as even *more* expensive.

1.  **The Bad Foundation:** Before step 3000 (when CPG is disabled), the `foot_clearance` reward gives a static `+0.8` bonus for keeping a foot in the air. The agent learns an asymmetric, lazy 5-leg walking style to farm this reward safely.
2.  **The Lock-in:** By step 3000, the neural network has committed its weight distribution to this 5-leg gait. Leg 4 is now a critical, load-bearing pillar.
3.  **The Trap:** When the CPG activates and demands Leg 4 be lifted, the agent realizes that lifting it will cause the torso to sag or collapse. A collapse triggers the massive `-60` `base_height` penalty and potentially ends the episode. 
4.  **The Local Optimum:** The gradient to switch from a 5-leg gait to a proper 3-leg tripod gait requires passing through an unstable transition phase. The critic network correctly evaluates that trying to learn the new gait will cause immediate, catastrophic falls. It is mathematically "cheaper" to just drag the legs and pay the `-30` tax than to fall over.

**Implemented Corrections (Phase 17c Curriculum Overhaul):**
Since reinstating the hybrid velocity/position reward from Phase 9 is not viable (the policy exploits it by swinging in place just to farm the velocity component), we must instead tackle the curriculum timing directly. The agent is spending too much time in the pre-CPG phase and locking into a 5-leg stance.

1.  **Earlier CPG Introduction:** Shifted all CPG curriculum terms (`cpg_z`, `cpg_rigidity`, `cpg_force`, `cpg_coxa`) to start much earlier at **step 1500** (approx 100 iterations), giving the agent less time to lock into a bad gait.
2.  **Extended CPG Z Annealing:** Increased the `end_step` for `cpg_z` (target weight `-2.0`) to **step 20000**. This extremely slow ramp-up ensures the agent doesn't suffer catastrophic forgetting when the massive penalty kicks in.
3.  **Phase 8 Reward Disablement:** Added fade-out curriculum terms for the two harsh Phase 8 static posture penalties (`base_height` at `-60.0`, and `non_foot_body` at `-30.0`). These now linearly decay to `0.0` between steps 1500 and 2000. As the CPG takes over gait control, these static crutches are removed so they don't fight the dynamic height variations natural to a tripod gait.

### 21. Phase 18: Telemetry Validation & The Episode Termination Crutch

**Observed Behavior (Post-Phase 17c):**
The curriculum overhaul yielded a major breakthrough: the hexapod achieved a recognizable partial tripod gait. However, the gait was asymmetric. 
To diagnose why, we ran a custom offline Pandas analysis script (`analyze_spikes.py`) on the telemetry data logged during inference (`debug_robot_kinematics.csv`).

**Finding 1: PhysX Transient Spikes (Validating the L1 Error Switch)**
The Pandas analysis proved the existence of massive, instantaneous physics collision spikes.
- Right Front (RF) leg recorded a 1-frame spike of **28.09 N**.
- Left Middle (LM) leg recorded a 1-frame spike of **44.81 N**.
*Conclusion:* This definitively proves why removing Mean Squared Error (MSE) from the `cpg_force_target_penalty` (Phase 10) was necessary. Under MSE, a 44.8 N toe-stub during swing would inflict a catastrophic $-1,003$ point penalty per step, immediately destroying the critic network and causing the "floor is lava" exploit. Using L1 error (Mean Absolute Error) turns this into a survivable, linear gradient ($-22.4$) that teaches the robot to simply "lift the leg higher."

**Finding 2: Perfect Left-Side Tripod Gait**
Telemetry analysis on the Left Side legs (LF, LM, LB) showed mathematically flawless CPG execution:
- During its swing phase, the Left Middle leg force drops exactly to **0.00 N**, while LF and LB bear full load (10 N - 22 N).
- At the phase transition, LF and LB lift cleanly (dropping to **0.00 N**), while LM takes the load (~15 N). 
*Conclusion:* The CPG reward logic is functionally perfect. The policy *has* learned how to execute the alternating tripod gait.

**Finding 3: The Right-Side Crutch & The Fear of Termination**
Telemetry analysis on the Right Side (RF, RM, RB) showed that the Right Back (RB) and Right Middle (RM) legs **never hit 0.00 N**. They remain permanently planted between 3 N and 11 N, completely ignoring their swing commands.

*Why is it doing this?*
We previously hypothesized the agent was afraid of the massive `-60.0` `base_height` penalty. This was incorrect—the curriculum successfully anneals `base_height` to `0.0` by step 2000. 
The true fear is **Episode Termination (`base_contact`)**.
- If the robot's root height drops below $0.05\text{ m}$, the episode ends instantly.
- If it dies at step 100, it loses 400 steps of `alive` ($+1.0$) and `forward_velocity` ($\approx +1.7$) bonuses, representing a loss of $> +1,000$ future points.

*The RL Economics of the Crutch:*
The neural network critic has calculated that obeying the CPG on the right side risks instability and early termination (return = 0). Conversely, keeping the right legs permanently planted provides a wide, 5-point stable base that guarantees survival to step 500. 
The tax for dragging the right legs is:
- `cpg_z_target` L1 penalty: $\approx -2.0$ per swinging leg.
- `cpg_force_target` penalty: $\approx -1.0$ per swinging leg.
Total tax is $\approx -3.0$ per step. Since the robot earns $\approx +2.7$ per step for moving forward, its net reward is near $0.0$.
Because the policy isn't bleeding massive points, it is perfectly content to survive 500 steps in this cost-neutral, dragging local optimum rather than risk an unstable transition to a perfect 3-leg gait.

**Next Steps:**
To break this crutch, **dragging must become mathematically worse than dying**. We must increase the CPG penalties (e.g., cranking `cpg_z_target` up to `-10.0` or higher, or applying a dedicated `cpg_swing_contact_penalty`) so that surviving while dragging yields a massive negative score (e.g., $-2,900$ per episode). The agent will be forced to lift its legs to stop the point bleed, inadvertently discovering the balance of the true tripod gait.