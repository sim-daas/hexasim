# Hexapod RL Locomotion: Research Progress

## Phase 1 Summary

Phase 1 successfully developed a Manager-Based RL environment in Isaac Lab for an Interbotix PhantomX Mark IV Hexapod, culminating in an emergent, approximate tripod gait driven by a dynamic Central Pattern Generator (CPG).

### Final State Reached
The current baseline policy was trained under the following constraints:
- **State Representation:** Pure proprioception data only (joint positions/velocities, base linear/angular velocity, projected gravity, CPG clocks). No exteroceptive sensors (vision/lidar).
- **History:** Single history of state (no observation history buffer or RNN/LSTM architectures).
- **Environment:** Infinite flat ground plane.
- **Domain Randomization:** None (no terrain noise, push forces, or friction/mass randomization).
- **Locomotion:** Achieved an approximate tripod gait with all 6 legs actively participating (overcoming earlier "crutch" local optima). The gait currently exhibits a net backward-left drift, indicating forward-velocity tracking has not fully converged.

### Metric Results (44k Steps Checkpoint)
Based on a 500-step (10.0 s) headless evaluation run:

* **Displacement:** 0.706 m total (dX = -0.600 m, dY = 0.373 m)
* **Mean Forward Velocity:** 0.209 m/s (Max: 0.461 m/s)

**CPG Phase-Speed Verification**
* **Mean Actual Δφ:** 0.1385 rad/step
* **Mean Expected Δφ:** 0.1315 rad/step (at stride length = 0.20m)
* **Ratio:** 1.053 (Target is 1.00; confirms CPG frequency correctly tracks robot speed)

**Stability Metrics (Unfiltered raw PhysX data)**
* **Vertical Acceleration (a_z) std:** 2.27 m/s² 
* **Roll Acceleration (alpha_x) range:** [-66.55, 85.89] rad/s²
* **Pitch Acceleration (alpha_y) range:** [-87.95, 68.17] rad/s²

**Per-Leg Gait Activity (Approximate Tripod)**
* **Tripod A (RF, LM, RB):** 52.6% swing time
* **Tripod B (LF, RM, LB):** 47.4% swing time
*(A perfect tripod gait operates at exactly 50.0% swing per tripod)*
## Phase 2.0 (S0a) Summary

Correctness fixes (spec §12 step 1) and S0a reconditioning (spec §4.2) of the Phase 1 base policy; full log in
`research/phase2-progress.md`.

### Final State Reached
- **Environment:** `Hexapod-S0a-v0`: AX-12A servo model (explicit, datasheet values), dropped reads with ZOH and
  validity mask, two-sample joint velocity, IMU noise, deployable 114-D actor input (111-D o_p + V_0 velocity
  estimate), separate critic with privileged block, camera-mast mass + payload U[0.1, 1.0] kg, flat ground μ 0.75.
- **Key fixes found on the way:** toe point everywhere; rest pose moved to femur/tibia +30° (the Phase 1 pose needs
  134 % of servo stall in tripod stance); PhysX `enable_external_forces_every_iteration=True` (without it explicit
  servo torques barely act); reward economics rebuilt so that walking > standing > falling (bounded toe-swing
  reward, alive/stand_still rescaled), signed coxa term kept.
- **Policy:** π_S0a = `logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt` (seed 42).

### Metric Results (256 episodes × 10 s, deterministic, `scripts/rsl_rl/eval_s0.py`)
* **Fall rate:** 0 % (π_base: 0 % on its own env, 9 % with random reset phase)
* **Mean forward velocity:** 0.291 m/s (π_base 0.159 m/s; the Phase 1 "0.209 m/s" came from a single 10 s run)
* **Contact duty factor:** 0.487 (π_base 0.648)
* **Per-leg real swings/s (RF RM RB LF LM LB):** 1.49 1.50 1.48 1.49 1.49 1.49. π_base: 0.81 0.10 0.11 0.82 0.81
  0.82 — its right-middle and right-rear legs were crutches.
* **Toe swing apex (q25/50/75):** 3.7 / 3.8 / 3.9 cm (target 3.8 cm)
* **Coxa power above 5 Hz:** 0.134 rad² (π_base 0.183)
* **V_0 velocity-estimate RMSE:** 0.031 m/s
* **Reproduction, spec checklist and deviations:** `research/phase2-progress.md` §2.0.15. Single seed (spec: S0 is one
  shared run; seed variance from S1 on). Servo/drop/IMU parameters are datasheet placeholders until the bench.
