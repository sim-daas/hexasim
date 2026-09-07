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