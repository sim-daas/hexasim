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

## Phase 2 Summary — Comparison Baselines: S0 Reconditioning → S1 Blind Trunk (started 2026-09-21, in progress)

Phase 2 builds the shared training paradigm and the baseline arms that the Phase 3 VP-JEPA policy will be compared
against (Shared Training Paradigm spec v2 + VP-JEPA Architecture v7). It covers the correctness fixes of spec §12
step 1, stage S0 (S0a reconditioning + S0b commands/terrain/randomisation) and stage S1 (the blind trunk with
spatial terrain materials, the proprioceptive encoder E_p, the velocity estimator V and the universal probes).

Detailed chronological log: `research/phase2-progress.md` (design, root causes, fixes, reward economics, decisions);
verbatim per-run logs and milestone tables: `research/phase2-run-logs.md`. Section numbers (§2.x.y) below refer to
`phase2-progress.md`. Plain-language overview (architecture built vs planned, stages, arms, rewards, progress):
`docs/dashboard/hexasim-dashboard.html` (open in a browser; source fragment `docs/dashboard/dashboard-source.html`).

### Status at a glance (2026-10-02)

| Stage | Status | Artefact |
| --- | --- | --- |
| Correctness fixes (spec §12 step 1) | Done | toe point, clock, coxa, duty factor, force target, friction, torque envelope |
| S0a (AX-12A, dropped reads, deployable obs) | Done, superseded as a lineage | π_S0a `logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt` |
| S0b (commands, clock, standing, terrain, DR, mirror) | Done | π_S0 `logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt` |
| From-scratch control arm (spec §4.2) | Passed at 0.30 × the warm-start budget | → warm start and KL anchor dropped for every arm |
| S1 infrastructure (materials, E_p, V, probes, contract P, S1PPO) | Done, merge-gating tests pass | `Hexapod-S1-v0` |
| S1 pilot (seed 42, W 50) | Accepted 2026-09-27 | `logs/rsl_rl/hexapod_s1/2026-09-26_19-41-49_s1_pilot3_seed42_resume4500/model_7999.pt` |
| S1 W sweep (W 25, 75; seed 42) | Done: rule selects no W; **W = 50 kept as deviation D32** (§2.2.10) | `…/hexapod_s1/2026-09-27_12-18-34_s1_sweep_w25_seed42`, `…/2026-09-27_20-17-13_s1_sweep_w75_seed42` |
| S1 seeds at W 50 (5 trunks, D31) | **Done: all 5 seeds (42–46) evaluated — every gate passes, 0 falls in 21,760 episodes, small seed spread** (§2.2.11) | `…/hexapod_s1/2026-09-2{8,9}_*_s1_w50_seed4{3,4,5,6}` |
| S1-K (B1-K's own trunk, ×5) | Not started, deferred by design (B1-K is first in the cut order) | — |
| S1 review response (2.3) | Done: spec drift fixed, W = 50 as deviation, latency / held-out layout / compliance / probe-skill evals, anticipation economy | §2.3 |
| O vs B0 headroom pilot (B0, O-mat, O-geo; seed 43, 4,096 × 16) | **Run and evaluated 2026-10-03: no-go for both oracles** (§2.3.12) — neither uses its foresight (ablation); arm differences are gait differences. Next step is the user's decision | `logs/rsl_rl/hexapod_s2/*_s2_headroom_*_seed43`, `logs/eval/headroom_summary.txt` |
| Soft-ground compliance (5–10 mm sink, D39) | Solver check passed at k 4e2 (§2.3.8); `soft_soil`, `wet_clay` in the S2 library; V passes its tests on them | `logs/eval/soft_solver_check.txt`, `logs/eval/pre/` |
| N_v benchmark, S1b, S2, S3 | Not started (after the headroom pilot) | — |

Headline: π_S0 passes every spec §4.2 S0b gate with **0 falls in 17,408 evaluation episodes**, all 108
training-range terrain cells and 0 falls in all 180 sweep cells up to 4 cm stairs / ≈ 20° slopes; the S1 pilot keeps
every S0b gate on the spatial-material terrain with the deployment velocity input (V replaces the privileged-trained
V_0), a velocity estimator ≈ 2.5 × more accurate than V_0, and a physically safe, coordinated gait. Across all five S1
seeds (42–46) every S0b gate passes on every seed with **0 falls in 21,760 evaluation episodes**, and the seed spread of
every task, estimator and gait metric is small next to its gate margin: the S1 result is a property of the
configuration, not of one lucky seed. **The blind O vs B0 headroom pilot (2026-10-03) is a no-go for both oracles:**
privileged foresight of material (O-mat) or shape (O-geo) is not used by the policy under the shared reward and buys
nothing measurable; the arms' differences are gait differences (§2.3.12). Per spec §11 the vision runs stop before N_v
unless the task is redesigned — the user's decision.

---

### Phase 2.0 — Correctness fixes and S0a reconditioning

**Scope** (agreed with the user): spec §12 step 1 (env-level correctness), the static torque envelope (§4.2
prerequisite) and S0a (§4.2). Decisions taken before implementation:

| Topic | Decision | Deviation |
| --- | --- | --- |
| Batch | 4,096 envs × 16 steps (π_base's real batch; compute limit) — spec says 8,192 × 16 | D1 |
| Isaac Sim / Lab | Stay on the installed version | — |
| Servo parameters | AX-12A datasheet values, provisional until the bench | — |
| V_0 | Small MLP on a 5-frame o_p history, trained online, output detached | D2 (spec leaves V_0 undefined) |
| S0a clock | Driven by the same ramped v_x the actor sees (GT → V_0), 0.8 Hz floor, per-env phase jitter | D3 |
| S0a critic | sg[actor obs] + 22-D privileged subset of contract P | D4 |
| Rest pose | Femur +30°, tibia +30° (was −30°), warm start folded | D5 (forced by the torque gate) |
| Payload | U[0.1, 1.0] kg per env brought into S0a (user request) | D6 |

#### Correctness fixes (§2.0.1–2.0.2)

* **Layout probe.** Joint order `[coxa × 6, femur × 6, tibia × 6]`, legs RF RM RB LF LM LB; asserted at env build.
  Robot mass 3.22 kg, so tripod-stance leg load ≈ 10.5 N (the old reward comment assumed ≈ 4 N).
* **Toe point** (`mdp.rewards.toe_pos_w`) replaces the tibia origin + 0.0892 m constant everywhere; the contact
  baseline EMA starts at ground height.
* **Clock**: floor 0.2 → 0.8 Hz (stall lock), per-env random phase at reset.
* **Coxa terms**: the contradictory velocity term deleted; `cpg_coxa_contact` uses the **signed** backward sweep,
  clipped ±π rad/s. With `abs()`, a coxa buzzing back and forth while planted earned the full rate — this turned out
  to be ~92 % of π_base's coxa reward (below).
* **Stance rigidity** excludes the coxa by design (D9): the coxa sweep *is* the stroke `cpg_coxa_contact` pays for.
* **Duty factor**: single source `env.cfg.duty_factor = 0.5`, asserted against the tripod offsets.
* **Force target**: weight / 3 read from `body_mass` (was 4 N from a wrong mass, which paid for extra legs in contact).
  L1, never squared. World-Z instead of surface normal on flat ground only (fixed in S0b).
* **Friction**: combine mode multiply, robot μ 1.0; sim sliding test with a negative control (average mode) PASS.
* **Ground μ 0.75** (D7): after the fixes π_base fell in 40 % of episodes. Ablation: the robot USD has no collider
  material, so π_base had trained at PhysX default 0.5 averaged with ground 1.0 = 0.75 effective. At matched μ the
  remaining 9 % falls come from the random reset phase π_base never saw (0 % without phase jitter):

| Environment (π_base, 256 episodes) | fall | v_x | contact duty | clearance |
| --- | --- | --- | --- | --- |
| Phase 1 code | 0.0 % | 0.159 m/s | 0.648 | 8.2 cm |
| All fixes, ground μ 1.0 | 39.8 % | 0.166 | 0.566 | 2.9 cm |
| **All fixes, ground μ 0.75 (Flat env)** | **9.0 %** | 0.181 | 0.627 | 4.9 cm |
| … without phase jitter | 0.0 % | 0.179 | 0.636 | 5.0 cm |

* **Camera mast**: provisional 0.15 kg at (+0.10, 0, +0.10) m on every task, blind arms included (spec §1).

#### Static torque envelope and rest pose (§2.0.3–2.0.4)

Isaac Lab's `applied_torque` for implicit PhysX drives is a PD estimate that under-reads ≈ 2×; lever-arm torques from
contact force × joint–toe arm are used instead. At the Phase 1 rest pose the femur needs **134 % of the 1.5 N·m
AX-12A stall** in tripod stance — gate FAILED. Sweep:

| femur, tibia | femur % stall | tibia % stall | stance height |
| --- | --- | --- | --- |
| +30°, −30° (Phase 1) | 134 % | 92 % | 0.123 m |
| **+30°, +30° (adopted)** | **12 %** | 29 % | 0.168 m |
| +45°, 0° | 33 % | 3 % | 0.183 m |

Payload up to +1.0 kg keeps femur ≤ 14 %, tibia ≤ 42 %. π_base re-centred on the new pose just stood (v_x −0.004 m/s),
so the warm start was **folded**: the pose shift is absorbed into the normaliser means and output bias, and the
widened 114-D actor reproduces π_base's joint targets exactly (2.7e-7 over 10,000 observations in float64).

#### AX-12A servo model and read pipeline (§2.0.5–2.0.6)

`hexasim/actuators/ax12a.py` — explicit torque: 10-bit quantised goal and position, bus latency 1–2 physics steps,
compliance margin/slope/punch, backlash 0.5°, τ = clip(τ_stall·(u − q̇/ω_nl)), stall 1.5 N·m, 59 rpm no-load,
armature 0.003 kg·m². `mdp/proprio.py`: ZOH of q, q̇_t = (q_t − q_{t−2})/2Δt held on dropped endpoints, 18-bit
validity mask. Drop probability was first tied to tracking error (18 % drops while walking — an infeasible target
grows the error without bound); now p = 0.005 + 0.068·|τ|/τ_stall (≈ 5–7 %, D8), provisional until the bus bench.

`Hexapod-S0a-v0`: actor 114-D = o_p 111 [q − q_rest, q̇, q_cmd − q, last action, IMU accel, IMU gyro, gravity, clock,
mask] ⊕ velocity 3; IMU noise; actions clipped to servo limits; V_0 α ramp 0 → 1 over iterations 300–1,500; S0PPO
with actor frozen until critic EV ≥ 0.8, normaliser re-estimated in a shadow copy and folded, std floor 0.05.

#### What went wrong on the way — and why (§2.0.7–2.0.13)

1. **Run 1 stood still** (v_x 0.003 m/s). Replaying π_base's own actions showed it walks under any PhysX *implicit*
   drive (even at AX-12A-like gains) and under **no explicit actuator** at any gain, rate, armature or friction.
2. **Root cause: PhysX TGS applied explicit joint efforts only once per step.** With Isaac Lab's default
   `enable_external_forces_every_iteration=False`, explicit torques barely act on this articulation (floating
   actuation test: joints tracked with q̇ rms 0.01–0.02 rad/s instead of ≈ 1). Fix: `PhysxCfg(enable_external_forces_
   every_iteration=True)` in S0a (D11); armature 0.003 kg·m² keeps the explicit PD stable at 5 ms. π_base then walks
   0.165 m/s under the AX-12A model. Every earlier explicit-actuator number was invalidated.
3. **Run 2 learned to fall** (falls 57 % → 99.7 % within 240 iterations of unfreezing). Reward economics: π_base's
   return rested on the old `abs(q̇)` coxa term (+8.81 per 10 s in Phase 1 vs +0.68 signed for the same gait), and the
   toe-point swing error was ~3× larger than the tibia-origin one, so every surviving behaviour had a negative
   per-step reward while a fall cost −5 once.
4. **Reward recalibration** (§2.0.11, measured with `scripts/tests/reward_economy.py`): L1 swing penalty replaced by a
   bounded toe-swing kernel exp(−(e/σ)²) (×5), alive ×10, stand_still −75 (D10). Ordering became
   walk +3.48 > stand +2.47 > fall −0.1 (discounted return); every improvement from the warm start now raises return.
5. **Run 3 walked** (0 falls from iteration 800) but the clearance gate "failed" — a metric bug: sub-5 mm stutter
   micro-lifts were averaged in. Fixed metric: apex 3.7 / 3.9 / 4.1 cm on the 3.8 cm target. π_base's 6–8 cm apex was
   high-stepping from its tibia-origin target, so the gate "± 10 % of π_base" was reinterpreted with the user (D12).
   The per-leg metric exposed that **π_base had two crutch legs** (RM, RB at 0.10–0.11 swings/s since Phase 1).
6. **Run 4** (σ 2 → 1 cm, continued) removed the crutch legs.

#### S0a result (256 episodes × 10 s, deterministic, `scripts/rsl_rl/eval_s0.py`)

* **Environment:** `Hexapod-S0a-v0`: AX-12A servo model (explicit, datasheet values), dropped reads with ZOH and
  validity mask, two-sample joint velocity, IMU noise, deployable 114-D actor input (111-D o_p + V_0 velocity
  estimate), separate critic with privileged block, camera-mast mass + payload U[0.1, 1.0] kg, flat ground μ 0.75.
* **Policy:** π_S0a = `logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt` (seed 42;
  runs 3 + 4, 4,498 iterations ≈ 295 M env steps after the warm start).
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

---

### Phase 2.1 — S0b: commands, clock, standing, terrain, dynamics randomisation

#### Design (§2.1.1–2.1.3)

| Piece | Implementation |
| --- | --- |
| Commands | `UniformVelocityCommand`, resampled every 5 s (D15); 10 % standing; later (D19) 15 % of moving segments turn on the spot (\|yaw\| 0.1–0.4 rad/s), 30 % walk with v_y ±0.05 / yaw ±0.4, 55 % straight; v_x U[0.05, 0.20] |
| Actor | 117-D = o_p 111 ⊕ V_0 3 ⊕ twist_cmd 3 |
| Clock | f_ref = clamp(v_cmd,x / 0.20, 0.8, 2.5) Hz from the command; frozen and **zeroed in the observation** in standing mode |
| Standing mode (D13) | zero command → clock stopped, every CPG term masked (`env.cpg_active`), standing terms on |
| Terrain | `mdp/terrain.py`: flat, rough, pyramid slopes up/down, bumps, stairs up/down; height map gives `env.terrain_z` / `terrain_normal` (never absolute z); curriculum `terrain_levels_cmd` promoted on commanded-path progress |
| Forces | terrain-normal / tangential split (`foot_forces_nt`) — closes the world-Z item |
| Rewards | per-step linear tracking (exp σ 0.05 + L1, ×30) and yaw tracking (×10); phase-free floors `swing_clearance` and `stance_rigidity_contact` introduced here (spec §5) |
| DR (D17, provisional) | foot μ_s U[0.3, 1.1], μ_d = μ_s·U[0.8, 1], compliance on 50 % of envs (k 2e3–5e4 N/m, c 10–200 N·s/m), payload + CoM ±1 cm, servo stall ×U[0.9, 1.1], compliance slope {16, 32, 64}, backlash 0.3–1.0°, Coulomb friction ≤ 0.05 N·m, latency 1–2 steps, per-episode IMU bias |
| Mirror augmentation (D21) | `mdp/symmetry.py` left–right reflection, rsl_rl `symmetry_cfg`; on for every arm from S0b |

Bugs caught while building: foot materials had no effect because the tibia colliders are **instance proxies** and
`bind_physics_material` silently skips them (fix: bind on the tibia rigid-body prim; verified by a tilt test, μ_eff =
sampled μ); Kit crashed at start because the task config imported `pxr` / Isaac Lab command classes at module level
(train.py loads the config before Kit) — imports made lazy. PhysX applies contact stiffness per contact point, so
effective foot stiffness ≈ 2.5 × the nominal k — foam must be fitted on measured sink, not nominal k.

#### Warm-start lineage: runs 1–14 (seed 42, from π_S0a) — what each run taught

| Run(s) | Change | Outcome / lesson |
| --- | --- | --- |
| 1 | S0b as designed | Gates pass except stride at 0.10 m/s (+18 %); right-side imbalance partly back |
| 2 | Standing: `standing_contact` ×5, joint-speed penalty | Stride +16 % plateau at 0.10; standing flickers on 3–4 legs |
| 3 | Gait-period-averaged velocity tracking | Worse (+27 %): the within-stride surge grew once only its average was scored. The exp/L1 kernels on instantaneous velocity peak at the modal instant, ≈ 7 % above the mean |
| 4 | Per-step tracking ×3 | **Root cause was command insensitivity** (speed slope ≈ 0.36 vs 1: zero-init command columns vs much larger command-independent terms). All walking gates pass |
| 5 | Zero the clock observation when standing | The actor had kept obeying the frozen clock (legs reading "swing" in contact 39 % vs 87 %); five legs now stand |
| 6–8 | `action_beyond_clip` (−1 → −10), `standing_pose` | Right-back coxa parked 14° past its servo stop — a gradient dead zone. Penalties fixed the clip, not the tucked leg |
| 9 | 50 % standing share (D18) | No change: standing is a barely-trained input region |
| 10 | **Settle window** 1.5 s | Root cause: with six feet planted the pose can only change by stepping, and every hold penalty made stepping dearer. First standing fix that worked (drift 4.0 → 1.8 cm); all walking gates pass on L0 / L5 / L9 |
| 11 | Turning: coxa sign from the commanded twist, yaw tracking ×10 | Right turns ≈ 75 %, left turns 27 % |
| 12 | Training terrain = levels 0–14 + stairs (D20) | Curriculum level 9.3 / 14, 0 falls |
| 13 | Reach-weighted coxa stroke | Exploit found: tucked toes near the coxa earned full reward for spinning; fix pays toe stroke speed. Left turn still 38 % |
| 14 | Mirror augmentation | Every asymmetric measure improved (left turn 52 %, standing contact 0.80); gap not closed |

Terrain decisions (§2.1.19–2.1.24): belly clearance ≈ 13–14 cm is not binding; the binding limit is the ≈ 3.8 cm swing
height, and at the lowest sampled μ 0.3 anything above ≈ 17° slides. An eval-only sweep of run 10 (108 cells) found
**0 % falls everywhere and no break point within the ≤ 3 cm / ≤ 15° cap**, so the user decided: train on levels 0–14,
hold out 15–20 (rough ±3 cm, ≈ 20° slopes, 4 cm stairs/bumps) as a secondary, sim-only result (D22). Beyond the cap
the policies degrade in speed and stride accuracy, not survival — the held-out suite must be scored on tracking and
stride, not falls.

#### The from-scratch control arm fired the spec §4.2 rule (§2.1.29–2.1.31)

Budget = the warm-start reconditioning effort (S0a + S0b = 26,596 iterations) × 1.2 ≈ 32,000 iterations; final S0b
task, random actor, same everything, actor freeze off.

| Iteration | falls | 0.10 m/s stride err | 0.15 m/s stride err | turn ±0.3 | standing duty / drift |
| --- | --- | --- | --- | --- | --- |
| 4,000 | 0 % | −36 % | −29 % | +0.064 / −0.063 | 0.90 / 1.5 cm |
| **8,000 (0.30 × budget)** | 0 % | −4.9 % | −6.8 % | +0.187 / −0.191 | 0.97 / 1.0 cm |
| 18,750 | 0 % | −1.8 % | −1.8 % | +0.257 / −0.254 | 0.95 / 0.8 cm |

At 8,000 iterations it passed all four S0b gates and already beat the warm-started lineage on standing, walk-and-turn
and — notably — turned **symmetrically**, showing the left/right asymmetry was inherited from π_base. Consequence
(paradigm spec §1, §4.2, §4.4 updated 2026-09-23): **the warm start and the KL anchor are dropped for every arm**; the
π_base → S0a → S0b lineage is kept as history. The run stopped at 20,749 on a return plateau (old π_S0,
`…/2026-09-22_22-50-43_s0_fromscratch_seed42/model_20700.pt`); 1,024-episode gates all passed with 0 falls in 17,408
episodes, and its 180-cell terrain sweep had 0 falls with every training-range cell passing.

#### Gait-quality corrections: why π_S0 was retrained (§2.1.34–2.1.38)

Gates alone did not reveal these; video inspection by the user and new probes did.

1. **Feet under the hips.** In the old π_S0 the middle/back toes stood 1–2 cm from their hips (rest reach 6.7 cm),
   shins slanting inward, stance ~25 % narrower, middle coxa arcs of 95°. Static torque scales with horizontal
   joint–foot distance and nothing in the reward asked for width, so the cheapest posture won — a sim2real contact
   risk (`scripts/rsl_rl/record_gait.py`).
2. **Rest-reach bug.** `coxa_rest_reach` had been measured in the USD zero pose (legs straight: 0.208 m for every leg)
   before the rest pose was applied, so from run 13 on — including old π_S0 — the coxa stroke term ran at ≈ 0.32 × its
   intended strength. Fixed (measure after writing the rest pose and stepping physics; 0.0669 m, asserted < 0.15).
3. **A stance-footprint term failed**: the policy met it by bending the tibia further (+43°) rather than extending.
4. **Mid-air toe in/out motion** (`scripts/tests/swing_radial_probe.py`): π_S0 lifted by folding the tibia, pulling
   the toe ≈ 5 cm inward and landing ≈ 5 cm inside lift-off. Geometry: at this rest pose the toe sits almost under the
   femur joint, so femur-up throws the toe out (1.7 mm/°) and tibia-fold pulls it in (2.2 mm/°); a vertical lift
   needs both (≈ 0.75 : 1). User decision: the baseline uses a **coordinated lift**.
5. **New terms**: `swing_radial_motion` (Σ airborne |Δ|r_xy||/dt, −2) and `touchdown_reach` (two-sided, |landing
   reach − rest reach|, −600). Fine-tuning the converged π_S0 did not reach a coordinated lift even at 5× weights —
   it shrank the sweep and froze the tibia folded (the basin is deep) — so S0 was retrained from scratch; by 1,750
   iterations the middle/back radial path was 6 cm per swing instead of 20.
6. **Leg-to-leg clearance.** Self-collision is off in sim, so legs passed through each other (capsule gap −3.8 cm in
   the new run, −6.3 cm in old π_S0; `scripts/tests/leg_gap_probe.py`, capsules fitted to the USD meshes).
   `leg_proximity_penalty` = Σ same-side neighbour pairs max(0, 2 cm − capsule gap), weight −400 (≈ 17 % of the walking
   reward at the overlap — a safety term). Per-joint coxa limits were rejected (collisions depend on angle pairs).

Every reward change was sized with `reward_economy.py` so that walk ≫ stand ≫ fall held (e.g. walk 0.70 ≫ stand 0.27
≫ fall −0.1 per step after the proximity term).

#### Final π_S0 (seed 42, trained from scratch, no warm start)

`logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt` — 12,214
iterations, 7.3 h, stopped on the plateau rule (return gain < 1.5 % per 2,000 iterations; returns 291.6 → 337.3 →
365.3 → 382.9 → 389.5 → 395.3 at 2k … 12k). Worst leg gap +0.5 cm at 2,000 → +1.3 cm at the end.

Gates (`scripts/rsl_rl/eval_s0b_gates.sh <ckpt> coordlift_prox_final 1024`, 1,024 episodes per condition, 17
conditions, L0 / L5 / L9):

| Metric | Result | Spec §4.2 gate |
| --- | --- | --- |
| Falls | **0 in 17,408 episodes** (95 % upper bound 0.29 % per condition) | ≤ 5 % |
| v_x cycle RMSE | ≤ 0.0046 m/s | ≤ 0.03 m/s |
| Stride error 0.10 m/s | +1.0 / +0.3 / −0.7 % | ± 10 % |
| Stride error 0.15 m/s | −0.2 / −0.6 / −1.4 % | ± 10 % |
| Dropped reads vs drop-free | equal | no degradation |
| Turn on the spot ±0.3 rad/s (own gate) | +0.321 / −0.311 | yaw cycle RMSE ≤ 0.05 |
| Walk 0.10 + yaw 0.2 (own gate) | v_x 0.100, yaw 0.216, stride +4.6 % | — |
| Standing | contact 0.97, drift 0.8 cm / 10 s | — |

Terrain sweep (`scripts/rsl_rl/eval_terrain_sweep.sh`, 180 cells × 128 episodes): **all 108 training-range cells
(L0–14) pass, 0 falls in all 180**. Held-out L15–20: 50 / 72 cells pass; failures are stride shortfalls, never falls
(rough from L16, −11 … −18 %; uphill slopes and stairs from L18–19).

Comparison with the superseded π_S0 (§2.1.39):

| | old π_S0 (`…22-50-43…/model_20700`) | final π_S0 (`…00-35-51…/model_12200`) |
| --- | --- | --- |
| Iterations / wall time | 20,749 / 12.9 h | **12,214 / 7.3 h** |
| Stride error at 0.10 m/s (L0/5/9) | +3.9 / +4.2 / +4.6 % | **+1.0 / +0.3 / −0.7 %** |
| Turn on the spot | ±0.26 | **+0.321 / −0.311** |
| Sweep L0–14 | 108 pass; mean \|stride err\| 2.4 % | 108 pass; 2.1 % |
| Sweep L15–20 | 52 pass; RMSE 0.0200 | 50 pass; **RMSE 0.0158** |
| Middle/back radial toe path per swing | ≈ 20 cm (tibia-fold lift) | **5.7–7.9 cm** (coordinated) |
| Middle/back stance reach / half-width | 1.8–2.1 cm / 0.11 m | 6.7–7.1 cm / 0.14 m |
| Worst leg-to-leg capsule gap | −6.3 cm (overlap in sim) | **+1.3 cm** |

Lesson carried forward: a silent measurement bug (rest reach) and missing collision/landing terms shaped a policy that
passed every gate; behaviour probes (`swing_radial_probe.py`, `leg_gap_probe.py`) are now part of every review.

Deviations added in S0b: D13 standing mode, D14 per-env foot material instead of a ground μ, D15 5 s command
resampling, D16 cycle-averaged tracking gate, D17 full DR in S0b, D18 temporary 50 % standing share, D19 turning
commands + twist-projected coxa stroke, D20 training terrain = levels 0–14 + stairs, D21 mirror augmentation, D22
held-out harder-terrain suite (TBD).

---

### Phase 2.2 — S1: blind trunk (started 2026-09-24)

**Plan** (agreed 2026-09-24): terrain materials before S1; V_0 → V by a widen-and-anneal strategy; per-env drop
slope; 8 S1 seeds; S1b blocked on the N_v benchmark (spec §6.2).

#### Spatial terrain materials (§2.2.1–2.2.2)

Until S1 every env had one foot material for the whole run. Contract P's material strips, the touchdown-material
probe targets and V's slip test need materials that vary over space. A feasibility spike (`scripts/tests/
material_spike.py`) showed per-foot writes of μ_s, μ_d and stiffness through the PhysX tensor API take effect
mid-contact (stiffness via the restitution slot: restitution < 0 ⇒ compliant), cost ≈ 55 µs per env written, and
that a compliant *ground* overrides the foot's written stiffness with no visible damping effect (test T6 failed).

Design (user decision, D27): a static Voronoi material field per 4 m cell (1 site 25 %, 2 sites 35 %, stripes 15 %,
mosaic 25 %; transition width U[0, 0.8] m), fixed seed so every arm and seed sees one layout, independent of the
height field (v7 §5.4). A provisional 10-entry library (concrete, rubber mat, PTFE, talc/graphite, dust, foam, wet
tile, wet rubber, glossy grit, polished coating) with a per-episode palette inside each entry's range. One visual +
semantic submesh per entry (for S2 depth and S3 appearance); the original mesh stays the single rigid collider;
per-foot physics written only when a low or contacting foot's target changed. Damping stays per env on the foot,
fixed per run (D25). Verified with `scripts/tests/material_field_test.py`: 82 / 82 slippery-footing envs slide and
225 / 225 grippy ones hold on a 20° tilt.

#### S1 implementation (§2.2.3)

| Piece | Where |
| --- | --- |
| Single-source layouts: actor 1,125 = π_S0's 117 ⊕ z_p 64 ⊕ ê 15 ⊕ zero-held blocks for later arms (M* 576, tokens 192, …); contract P 683; mirror rule per block | `mdp/layout.py` |
| E_p: causal dilated TCN (k 3, d {1,2,4,8}, RF 61 ≥ W asserted), 128 → 64 + LayerNorm | `rl/s1/tcn.py` |
| URDF forward kinematics (vs sim toe 0.008 mm) and v_kin with stance weights from tracking error + toe-plane fit (no contact switches) | `rl/s1/kinematics.py` |
| V = v_kin + Δv̂(z_p), contact-force estimate f̂; universal probes (current 84 + touchdown 102), β-NLL β 0.5, touchdown material ×2, constant weight; VICReg | `rl/s1/aux.py`, `losses.py` |
| Sequence buffer L 128 with episode-boundary masking; touchdown targets = first touchdown after t | `rl/s1/seq_buffer.py` |
| `Hexapod-S1-v0`: materials, per-env drop slope × U[0.5, 2] (D24), env-owned aux learner, contract P at the nominal foothold | `hexapod_env.py:HexapodS1Env` |
| S1PPO: aux update per iteration, frozen normaliser dims, V tests every 100 iterations → ê gate, V_0 retirement | `rl/s1_ppo.py` |

**V_0 → V switch (D23).** V_0 stays in its old columns, frozen at π_S0's weights and normaliser. ê (V's output) is
appended with zero-init columns behind a binary gate (exact zero input while shut → no gradient). When V passes v7
§4.3 on three consecutive checks (slip-stratified error < 3 × rigid; zeroing the IMU raises error ≥ 1.5 ×), ê's
statistics are set while its columns are still zero (function-preserving) and the gate opens; 1,000 iterations later
V_0 is dropped per episode (never per step), probability 0 → 1 over 1,000 iterations. After S1 V_0's columns are
permanently zero.

**Merge-gating tests** (`scripts/tests/test_s1.py`): widened actor = π_S0 exactly (max |Δ| 0.0 over 10,000 random
observations with random values in every new block); value-loss backward leaves actor grads None; PPO backward leaves
every aux grad None; aux backward leaves actor and critic None; gated and zero-held columns get exactly zero gradient;
TCN RF; β-NLL gradient; buffer semantics across a reset; mirror involution. Wiring in sim:
`scripts/tests/s1_env_check.py` PASS. v_kin stance constants calibrated in sim (D26; v_kin RMSE 0.063 m/s, 90 %
agreement with the contact sensor; a lower-RMSE optimum that weighted only the lowest feet was rejected because it
collapses on uneven ground).

**Collection cost** (§2.2.4): the first launch ran 5.9 s / iteration against 2.15 s for S0b. Fewer envs would not
help (the batch is an invariant). Causes: per-entry *colliders* (every foot tested against 10 terrain-wide meshes,
+60 ms / step) → visual-only submeshes; fp16 inference for rollout latents (1.6×); fp16 mixed-precision aux training.
Result 4.0 s / iteration at 4,096 × 16.

#### S1 pilot 1 (seed 42, W 50) — infrastructure bugs and two gait regressions (§2.2.5–2.2.6)

Bugs fixed before the pilot of record: an in-place write to an inference tensor when the ê gate opened (rsl_rl
recreates normaliser tensors in inference mode; regression test added); probe target statistics frozen from the
first buffer fill at curriculum levels 0–2 (now fixed statistics from a π_S0 rollout over levels 0–14,
`hexasim/rl/s1/probe_stats_pi_s0.pt`, shared by every seed and arm); the long-standing Kit "telemetry flake" was
actually numpy's OpenBLAS at-fork handler racing a Kit start-up `fork()` (fix `OPENBLAS_NUM_THREADS=1`); evals next to
training ran out of GPU memory because PhysX pre-allocates 4,096-env buffers (fix `load_policy.eval_physx_buffers`).

Pilot 1 (`logs/rsl_rl/hexapod_s1/2026-09-25_03-28-11_s1_pilot_seed42`): actor unfrozen at 300, ê gate open at 399,
V_0 fully retired at ≈ 2,400. Every S0b gate passed at every 2,000-iteration milestone with 0 falls; V ≈ 2.5 × more
accurate than the frozen V_0 (walking v̂ RMSE 0.019–0.025 vs 0.050–0.057), slip / rigid 1.5–2.5 ×, IMU dropout ≈ 5–6 ×.
But two gait regressions appeared: the worst leg gap oscillated around zero (+1.3 cm in π_S0 → −0.6 … +0.3 cm, up to
7.5 % of steps < 2 cm) and swing lift drifted from 4.0–4.5 to 3.5–4.0 cm. Stopped at 8,650 to diagnose.

**Diagnosis.** Not friction (slippery and grippy footing produce the same gait). The policy pulled the front toes
inward and swept the middle coxas back, so neighbours closed in. Cause: the **two-sided `touchdown_reach` pulls the
front legs from their natural ≈ 12 cm landing to the single 6.7 cm rest target** (+0.70 / s), which outweighed the
proximity cost (−0.61 / s) — overlapping was profitable. S1 simply continued optimising the S0b reward (π_S0 was
stopped on a plateau while `touchdown_reach` still improved). The lift trend is `cpg_z_target` converging on its own
3.8 cm half-sine; a real risk for S2, where only `swing_clearance` (saturating at 3 cm) defends lift. A retrain with a
one-sided `touchdown_reach` sprawled the legs (reach 0.16–0.18 m, back legs stopped lifting) because the reach-weighted
coxa term pays extension — the two-sided form had been holding it in check, so it stays.

#### S1 pilots 2 and 3 — the final S1 configuration (§2.2.7–2.2.8)

**Pilot 2** (D28, S1-only overrides shared by every arm): `leg_proximity` −400 → **−1,600** (overlap now nets
−0.69 / s), `swing_clearance` saturation 3 → **4 cm**; plateau-based budget. Clearance held at π_S0's level all run
(worst gap +0.8 … +1.3 cm). Return gains 1.54 % → 2.17 % → **1.47 % at 8,000 → S1 budget = 8,000 iterations**
(≈ 524 M env steps), fixed for the W sweep and every seed. (The wrapper first compared a rounded "+1.5" and missed
the stop; the parser now uses raw values.) Residuals: front legs still pulled inward (reach 0.052 m, coxa range 72°,
landing 9.5 cm) and lift converging on 3.8 cm.

**Pilot 3** (D29): `cpg_z_target` step height 3.8 → **4 cm** (clock term and floor now agree) and a **per-leg
two-sided `touchdown_reach`**: front legs target 10 cm, middle/back keep the rest reach. A pre-declared regression
watch stopped it at 4,500 (coxa clip term 0.025 from the back coxa reaching its stop; lift 4.8–5.1 cm, overshooting);
the user chose to resume to 8,000 and decide at 6,000, where the clip term was falling (0.0206) and lift converging.

Pilot 3 at 8,000 (`…/2026-09-26_19-41-49_s1_pilot3_seed42_resume4500/model_7999.pt`; gates 256 episodes on
`Hexapod-S1-v0`, against π_S0 on the same env):

| | π_S0 (S1 env) | pilot 2 it 8,000 | **pilot 3 it 8,000** |
| --- | --- | --- | --- |
| Falls, 17 conditions | 0 % | 0 % | 0 % |
| Stride err 0.10 (L0 / L5 / L9) | +0.2 / −1.0 / −1.4 % | +1.3 / +1.5 / +2.0 % | +0.5 / +0.2 / +0.7 % |
| Stride err 0.15 (L0 / L5 / L9) | −0.9 / −1.9 / −2.1 % | −2.3 / −2.4 / −2.2 % | −2.6 / −2.9 / −2.7 % |
| Cycle RMSE (max walking) | ≤ 0.0050 | ≤ 0.0048 | ≤ 0.0051 |
| Turn ±0.3: ω̄_z; walk-and-turn yaw RMSE | +0.320 / −0.309; 0.021 | +0.310 / −0.313; 0.007 | +0.309 / −0.306; 0.007 |
| Standing contact L0 / L5 / L9 | 0.97 / 0.84 / 0.82 | 0.98 / 0.95 / 0.91 | 0.98 / 0.95 / 0.91 |
| V: v̂ RMSE; slip / rigid; IMU zeroed | — | 0.019–0.025; ≤ 2.2 ×; ≈ 8–9 × | 0.019–0.025; ≤ 2.2 ×; ≈ 8–9 × |
| Worst leg gap; < 2 cm at 0.15 / 0.20 | +1.3 cm; ≈ 0 % | +1.3 cm; ≤ 2.7 / ≤ 0.1 % | +0.5 cm; ≤ 1.8 / ≤ 0.1 % |
| Lift f / m / b (swing probe) | 4.0–4.5 cm | 3.5–3.7 / 3.8–3.9 / 4.1–4.2 | 3.8–3.9 / 4.3–4.4 / 4.3 |
| Front landing reach; front coxa range | 11.9 cm; 49° | 9.5 cm; 72° | 11.2 cm; 60° |
| Coxa clip term (rad/step) | 0.015 | 0.022 | 0.026–0.029 |

**Accepted by the user 2026-09-27** as the S1 configuration: S0b reward + `leg_proximity` −1,600, `swing_clearance`
4 cm, `cpg_z_target` 4 cm, per-leg `touchdown_reach` (front 10 cm); budget 8,000 iterations. Against the spec §6.1 S1
gate: performance ≥ S0 on the S1 env, V passes v7 §4.3 (gate open at ≈ 400), probe NLL flat, V_0 retired, return
plateaued. Known imperfection **D30**: back-coxa requests past the ±55° clip (0.026–0.029 rad/step vs 0.015 in π_S0;
−0.05 / s of ≈ 40 / s reward, clipped by the servo, no gate affected) — revisit in the S2 reward design. Pilot 3 counts
as seed 42 / W 50 for the paired branching (one difference from a clean run: resumed at 4,500 with a restarted RNG
and refilled aux buffer).

#### Demo-change audit, W sweep and S1 seeds (§2.2.9–2.2.11)

**Audit.** The Play-env / teleop changes made around pilot 3 (`research/playenv-teleop-updates.md`) did not touch
training or evaluation: the saved configs of every S1 run differ only in their intended seed / window and a terrain
`color_scheme`, which is verified to leave the geometry bit-identical. Found on the way (pre-existing): Isaac Lab's rough
and bump terrains use numpy's global RNG, so each seed trains on its own terrain instance; eval seeds must stay fixed
across arms.

**W sweep → W = 50, as a deviation (D32).** All three windows pass every gate with equivalent gaits. Against the clean
W 50 seeds 43–46 (seed 42 was resumed): W 25 is better on V's slip error (0.0527 vs 0.0550–0.0566) but worse on both
probes (current NLL −0.090 vs −0.113 … −0.107); W 75 is worse on V and the current probe. The §9.1 rule therefore selects
no window. W 50 is kept because blind S1 metrics are biased toward short windows: what a longer window buys (estimation
off the limit cycle once the clock leaves, touchdown timing; W 25 is below one stance at 0.8 Hz) is invisible in S1.

**S1 seeds (W 50), all five (42–46), 256 episodes × 17 conditions each:**

| Metric | mean ± sd across seeds [min, max] | π_S0 on the S1 env |
| --- | --- | --- |
| Falls | **0 in all 85 cells (21,760 episodes)** | 0 |
| All S0b gates | **pass on every seed** | pass |
| Worst-cell v_x cycle RMSE (m/s) | 0.0055 ± 0.0008 [0.0048, 0.0067] | ≤ 0.0050 |
| Stride err 0.10 / 0.15 m/s (L0) | +1.2 ± 1.1 %; −1.8 ± 1.1 % | +0.2 %; −0.9 % |
| Turn on the spot ±0.3 rad/s | +0.302 ± 0.004 / −0.301 ± 0.005 | +0.320 / −0.309 |
| Standing contact, level 9 | 0.918 ± 0.010 | 0.82 |
| V slip RMSE; slip / rigid; IMU-zero ratio | 0.0561 ± 0.0006 m/s; 2.24 ± 0.08 ×; 8.3 ± 2.3 × | — |
| Probe NLL current / touchdown | −0.107 ± 0.005 / 0.065 ± 0.006 | — |
| Worst leg gap; front lift; front landing reach | +0.88 ± 0.37 cm; 4.35 ± 0.24 cm; 11.1 ± 0.1 cm | +1.3 cm; 4.0 cm; 11.9 cm |
| Coxa clip term (D30) | 0.020 ± 0.004 (fresh seeds 0.018–0.019; pilot 3 0.026) | 0.015 |

Adding seed 46 to the four-seed set moved no aggregate materially (V slip RMSE 0.0561 ± 0.0007 → ± 0.0006; worst leg gap
+0.95 → +0.88 cm), i.e. the spread is converged. The **S1 result is robust, not lucky**: five independent seeds — each on
its own terrain instance — reproduce it. These five trunks are the pairing units for the spec §10.1 paired statistics.

#### Carrying into S2 and the arms

**Things to keep in mind:**

* **Longer training exposes the reward's true optimum.** The leg overlap and lift drift only appeared after 4,000–8,000
  iterations, and S2 / S3 are several times longer. Keep the gait regression watch at every milestone.
* **These S1 reward imperfections need settling in the S2 reward design:**
  * the coxa-limit term;
  * the front-leg reach target;
  * the weight of the 4 cm lift floor once the clock terms go;
  * `touchdown_reach` survives the clock anneal while the coxa term that it balances leaves.
* **Before changing a reward term, check the per-term ledger on more than one posture.** The one-sided
  `touchdown_reach` sprawl was exactly this mistake.
* **Keep evaluation seeds fixed across arms**, because terrain geometry depends on the run seed.

**Ready for the next phase, but not yet for the vision arms.** The spec's order from here is:

1. Benchmark the vision batch size on the A4000.
2. Run S1b at that batch.
3. Design the S2 reward with the clock anneal, checked with the per-term ledger.
4. Run the blind O vs B0 headroom test. If the oracle doesn't beat the blind arm on transition terrain, anticipation has
   no value here and the vision runs are stopped.
5. Only then B2 and F, then B1-M and B1-K, with B3 optional.

---

### Phase 2.3 — S1 review response (2026-09-30 → 2026-10-02)

An external review of the S1 results asked for eight checks before more seeds. Sim-only answers (the servo bench is
deferred until the sim results are presented, D34; keep/rerun rule pre-registered in spec §12 step 7):

| Review item | Result |
| --- | --- |
| W choice | Against the clean seeds 43–46 the §9.1 rule selects no window (W 25 better V, worse probes); W 50 kept as deviation D32 — blind S1 metrics cannot see what a longer window buys later |
| Servo / bus latency | Spec 4–12 ms vs trained 5–10 ms. At 15 ms and 2× drop slope all five trunks keep every gate, 0 falls (stride −2…−3 %, turn rate −5 %). Latency DR widened to 5–15 ms from S1b (D36) |
| Reward vs anticipation | `anticipation_economy.py`: flat-ground friction is harmless for this robot (even μ 0.05: 0 falls); on 15° slopes a μ 0.25 strip costs ≈ 80–90 reward (runaway / stall), but stride shortening barely reduces slip and loses to velocity tracking. → primary metrics become transition-window tracking / slip / falls and first-contact impact; stride descriptive (D38) |
| Mirror in vision stages | Kept: colour optic on the sagittal plane, depth corruption after mirroring (D35) |
| Compliance signature | Small but consistent: foam sinks 1.6 mm more, stance tracking error +5–12 %; compliance kept as a secondary claim |
| Probe skill | z_p beats the mean predictor: friction R² 0.63–0.70 (rear legs 0.70 — the trailing-leg half exists), log k 0.56, touchdown timing 0.92 |
| Headroom test now | Designed: B0 vs O-mat (material foresight) vs O-geo (shape foresight, new, D37) from seed 43, blind, non-reportable go/no-go with pre-registered rules (phase2-progress §2.3.7) |
| Held-out material layout | Layout seed 1 ≡ layout 0 on every metric — nothing memorised; frozen eval sets use layout ≠ 0 |
| Spec drift | Batch 4,096 × 16, one-sided paired t-test with the seed as unit, sign test secondary, symmetry row, dropped-reads row, §9.1, §12 progress — fixed |

#### Planned: soft-ground compliance (D39, decided 2026-10-02 — to do, nothing implemented or run yet)

**Why.** Outdoor dirt / clay sinks ≈ 5–10 mm under a foot. The simulator has never produced that: the softest library
entry (foam, nominal k 2e3–8e3 N/m) gave a *measured* 1.6 mm of extra stance sink at k 3.2e3 (§2.3.5), and scaling by
1 / k gives an *estimated* ≈ 2.6 mm at k 2e3, ≈ 3.5 mm with the +1 kg payload — PhysX applies the stiffness per contact
point, so the ground behaves ≈ 2.5 × stiffer than nominal. Every trunk has therefore seen ≤ ≈ 3.5 mm, and a
"no headroom for compliance" result from the pilot would say nothing about real soft ground.

**What is and is not modelled** (to be stated in the paper):

| | Status |
| --- | --- |
| Friction | Spatial material field, 10-entry library μ 0.2–1.1, per-episode value inside each entry's range, per-foot writes, combine mode multiply verified by slide / tilt tests |
| Compliance | PhysX spring–damper contact, k 2e3–5e4 N/m, c 10–200 N·s/m — too stiff for soil; spatially varying damping had no effect (spike T6) |
| Not modelled (PhysX cannot) | Permanent sinkage and ruts, granular flow (gravel rolling, mud shearing), sinkage-dependent friction |

Scope of the claim that follows: surfaces with Coulomb friction and *elastic* softness (tiles, mats, foam, firm soil).
Mud and loose gravel are out of scope and reported as a limitation unless a sinkage model is added. The sim2real risk is
the appearance → physics binding (it transfers only for surfaces whose real μ and k are measured and inside the
randomised ranges); the trailing-leg path learns from what the feet feel and transfers as long as real contact falls
inside those ranges.

**Clarifications recorded with the decision.** (i) Joint tracking error `q_cmd − q` is an observation and v_kin's stance
cue, not a reward term: soft ground costs the policy only through body height, pitch, touchdown timing and velocity
tracking. (ii) The clock and the four clock-phase reward terms leave over the first 30 % of **S2**, not in S1b (S1b is
the blind batch-size change with the clock on); the headroom pilot is a blind S2, so they leave there. (iii) "Oracle ≫
blind on soft ground" is the hypothesis the pilot tests, not an assumption — on flat friction it already failed.

**Steps, in order:**

1. **Solver check (sim only, before any training on it).** Static-load and walking sink test at nominal k 4e2 N/m, 5 ms
   step (pattern: `scripts/tests/material_field_test.py`, `gait_by_material.py --split compliance`): sink ≈ leg load /
   k_eff, no jitter, no toe passing through the terrain mesh. If it fails: raise the softest k until it holds and record
   the reachable sink.
2. **Two provisional library entries** (wide randomisation instead of bench numbers, user decision): `soft_soil`
   k 4e2–1e3 N/m, μ 0.4–0.7; `wet_clay` k 6e2–2e3 N/m, μ 0.25–0.5 — target 5–10 mm sink at 10.5–14 N per leg. Ranges are
   placeholders until the bench.
3. **Introduce them for every arm at the same point:** at the branch of the headroom pilot (B0, O-mat, O-geo, non-
   reportable) and formally at **S1b** for the reportable runs — the library is a shared invariant (spec §1), like the
   latency change D36. Probe target statistics (`probe_stats_pi_s0.pt`) stay fixed (softer k only gives larger z-scores);
   recompute only if the log-k probe degrades.
4. **Re-run V's tests on soft ground** (slip-stratified error, IMU-zero ratio): uneven sink changes the tracking-error
   and toe-plane cues v_kin's stance weights use (D26).
5. **Pilot evaluation gets soft-ground conditions:** foam, soft_soil and wet_clay strips at matched μ on flat and on 15°
   slopes; window metrics as §2.3.7 plus stance sink, body height / pitch error and touchdown-timing error.
6. **Claim rule.** Compliance is promoted from a secondary to a primary claim (beside slip) only if O-mat beats B0 on the
   soft-ground conditions under the pre-registered go rule (≥ 10 %, CI excluding 0, every gate passing); otherwise the
   headline stays friction and soft ground is reported as tested-without-benefit.
7. **Bench, with the deferred servo bench (D34):** static and walking sink per leg on real dirt / clay under known load;
   fit k and μ; apply the keep/rerun rule (keep iff the fitted values lie inside the randomised ranges).

#### Task list up to the N_v benchmark (order fixed 2026-10-02; details in phase2-progress §2.3.8–2.3.12)

| # | Task | Status (2026-10-02) | Result |
| --- | --- | --- | --- |
| 1 | Soft-ground solver check at nominal k 4e2 N/m | **Done, passed** (§2.3.8) | Stable down to k 4e2, 0 falls; stance sink walking 8.5 mm (k 4e2) … 2.3 mm (k 2e3), standing 5.0 … 1.1 mm; ground acts ≈ 3 × stiffer than nominal |
| 2 | H* block (144-D, O-geo) | **Done** (§2.3.9) | S2 layout 1,269-D; S1 stays 1,125-D |
| 3 | Trunk widening at the branch, every arm | **Done** | Actor + critic match the seed-43 trunk to 2.3e-12 over 10,000 observations |
| 4 | Two soft library entries for every arm from the branch | **Done** (D39, D40) | `soft_soil` k 4e2–1e3, μ 0.4–0.7; `wet_clay` k 6e2–2e3, μ 0.25–0.5 (S2 uses 12 entries, S1 keeps 10) |
| 5 | Pre-training checks: reward economy, V on soft ground, material field, merge tests | **Done, all pass** (§2.3.10) | Walk 0.897 > stand 0.302 > fall −0.10 per step without the clock terms; V error 0.020 → 0.034 m/s at k 4e2, both v7 §4.3 tests pass |
| 6 | 20–40-iteration branch smoke test per arm | **Done** — found and fixed two branch bugs (§2.3.9) | Critic saw a newly live block as noise × 100; Adam sign-steps on new columns wrecked O-mat. After the fixes the three arms are identical while frozen and train alike |
| 7 | Headroom evaluation scripts (window + impact metrics, soft conditions, verdict) | **Done, not yet run on arms** | `eval_headroom.py` / `.sh`; reaction lead time left out (not in the go rule) |
| 7b | Blind-trunk economy on soft strips, bumps, rough, stairs (walk vs scripted slow-down, as §2.3.4 did for friction) | **Done** (§2.3.11) | Only soft k 4e2 on a 15° slope breaks the trunk (stalls at ≈ 0.05–0.06 of 0.15 m/s, slip × 6); flat soft, bumps, rough, stairs: speed held, 0 falls, slowing costs 83–113 reward. Cliff lies between k 4e2 and 1e3 |
| 8 | Stop for the user's OK | Done (OK 2026-10-02, soft range kept as is) | — |
| 9 | Runs B0 → O-mat → O-geo, 6,000 iterations each from seed 43, milestone reviews | **Done** (§2.3.12), ≈ 7.3 h each | Same reward, three gaits: O-mat lifts every leg 1–1.8 cm lower than B0, O-geo its front legs 1.5 cm lower; gaps fine; reward 350 / 351 / 355 |
| 10 | Evaluation (50 cells + 17 gates per arm, held-out levels 15–20 added by the user) + pre-registered go / no-go | **Done: no-go for both** (§2.3.12) | O-mat "wins" slope-up μ 0.25 (+14 % tracking, +19 % slip) but fails the tracking gate and its flat negative control; the placebo O-geo gains more on the same cell; with M* uninformed its slip is unchanged. O-geo: no impact gain, 20–33 % more short swings. Spec §11 first row: stop before N_v |
| 11 | **User decision:** stop the vision arms (spec §11), or redesign the task so late reaction is costly and repeat the pilot, or train the oracles longer / live from S1 | **← here** | — |
| 12 | **Decision (user, 2026-10-04): redesign.** Missing metrics + baseline, slip measured properly (§2.3.14); redesign reward `Hexapod-S1R-v0` (loaded slip, landing speed, rocking, tilt against the ground, stronger lift floor; 3.5 % each, ramp to 2 ×), economy check, 1 m material cells, pre-registered two-gate rule (§2.3.15) | **Done** | Blind walker: 1–3° wobble, half of all stances slide > 1 cm on hard terrain, feet land at 0.26–0.31 m/s everywhere; walk > stand > fall at both ramp ends |
| 13 | Shape oracle alone from π_S0 with H* live (`…/hexapod_s1r/2026-10-04_14-36-36_s1r_O-geo_seed43`, 8,000 iterations); gate 1 (hide test) at each 2,000-iteration milestone | **Stopped at 7,400** (user) | Gate 1 at 2,000 / 4,000 / 6,000: H* used on slopes (slip +24 … +67 % without it), never on bumps / stairs / rough |
| 14 | Root cause (§2.3.16): heading bug — the strip / scan / footprint points did not turn with the robot (14 % of footfalls inside their own patch); fixed 2026-10-05 (91–96 %). Affects M*, H*, contract P, probe grids of every earlier run, **incl. the §2.3.12 no-go** | **Fixed, verified** | Blind trunks kept (user); shape oracle rerun with the fix, same recipe |
| 15 | Shape oracle with the fix (`…/hexapod_s1r/2026-10-05_00-07-06_s1r_O-geo_seed43_yawfix/model_7999.pt`) | **Done: gate 1 passed** | Hiding H* costs ≈ 20 % on slip / rocking / wobble on rough and stairs down, 74–78 % slip on the slippery slope, flat unchanged; grows with training; landing speed does not depend on H* |
| 16 | Blind arm, same recipe (`…/2026-10-05_10-31-40_s1r_B0_seed43`), then gate 2 (20 cells, terrain excess vs B0) | **Done: gate 2 not passed** (§2.3.17) | 0 falls both arms. Oracle moderately better on rough ground (6–18 % on slip / rocking / wobble, 16–34 % tilt) and steadier on slopes; nothing on stairs up / bumps; worse on soft ground; 14–28 % more energy per metre everywhere. One seed per arm; the terrain-excess statistic proved fragile. Next step: user decision |
| 17 | 0.3 m/s check on both checkpoints; per-leg body-referenced height oracle `O-leg` (12-D, 1,281-D layout), 6,000 iterations, vs B0 and O-geo at 6,000 (§2.3.18–19) | **Done** | Untrained speed opens no gap (half of all stances slide on flat). O-leg: calmer but more expensive gait on every surface incl. flat, no advantage on stairs / bumps; O-geo stays the better oracle on rough ground. Pattern stable: moderate steadiness gain on rough / slopes only. Next step: user decision |
| — | After the verdict: N_v benchmark → S1b (latency 5–15 ms, soft entries; reset the critic's contract-P statistics, §2.3.9) → S2 | Out of this list | — |

### Deviations from the specs (D1–D40, reasons in `research/phase2-progress.md`)

| # | Deviation | § |
| --- | --- | --- |
| D1 | Batch 4,096 × 16, not 8,192 × 16 | 2.0 |
| D2 | V_0 = 5-frame o_p MLP trained online, output detached | 2.0.6 |
| D3 | S0a clock from the ramped v_x (replaced by the command clock in S0b) | 2.0 |
| D4 | S0a critic privileged block = 22-D subset of P | 2.0.6 |
| D5 | Rest pose femur/tibia +30° (torque gate), folded warm start | 2.0.3–2.0.4 |
| D6 | Payload U[0.1, 1.0] kg from S0a | 2.0 |
| D7 | Flat-ground μ 0.75 = π_base's effective training friction | 2.0.2 |
| D8 | Drop probability driven by servo load, not tracking error | 2.0.5 |
| D9 | Coxa excluded from `cpg_stance_rigidity` | 2.0.2 |
| D10 | Bounded toe-swing kernel, alive ×10, stand_still −75 in S0a | 2.0.11 |
| D11 | PhysX `enable_external_forces_every_iteration=True` for explicit servos | 2.0.9 |
| D12 | Duty gate as contact duty; clearance gate reinterpreted | 2.0.13 |
| D13 | Standing mode at zero command | 2.1.1 |
| D14 | Per-env foot material instead of a sampled ground μ | 2.1.1 |
| D15 | twist_cmd resampled every 5 s | 2.1 |
| D16 | Tracking gate on cycle-averaged v_x | 2.1.3 |
| D17 | Full DR from S0b | 2.1 |
| D18 | Standing share 50 % for runs 9–10 (history) | 2.1.17 |
| D19 | Turning commands; coxa stroke projected on the commanded twist, reach-weighted | 2.1.21, 2.1.26 |
| D20 | Training terrain levels 0–14 + stairs; 15–20 eval-only | 2.1.24–2.1.25 |
| D21 | Left–right mirror augmentation, every arm from S0b | 2.1.27 |
| D22 | Held-out harder-terrain suite as a secondary result (TBD) | 2.1.24 |
| D23 | V_0 kept frozen in S1 and retired by per-episode dropout | 2.2.3 |
| D24 | Drop slope × U[0.5, 2] per env and episode | 2.2.3 |
| D25 | Damping per env on the foot, fixed per run | 2.2.1–2.2.2 |
| D26 | v_kin stance constants calibrated in sim | 2.2.3 |
| D27 | Materials: static per-entry visual submeshes + per-foot tensor writes, per-episode palette | 2.2.2 |
| D28 | S1 reward: `leg_proximity` −1,600, `swing_clearance` 4 cm | 2.2.7 |
| D29 | S1 reward: `cpg_z_target` 4 cm, per-leg `touchdown_reach` (front 10 cm) | 2.2.8 |
| D30 | Back-coxa clip excess accepted in S1, revisit in S2 (mostly the resumed pilot: fresh seeds 0.018–0.019) | 2.2.8, 2.2.11 |
| D31 | 5 S1 trunk seeds (42–46), not 8: caps B2 / F at the 5-seed floor unless three more trunks are trained | 2.2.12 |
| D32 | W 50 kept as a deviation: §9.1 rule selects no window vs clean seeds 43–46; blind S1 metrics favour short W | 2.2.10 |
| D33 | `non_foot_body` (femur < 5 cm, −30) never annealed: live since S0a (the resume config dropped its schedule); ≈ 0 effect, same for every arm; decide in S2 | 2.2.12 |
| D34 | Servo / bus bench deferred past S1; keep/rerun rule pre-registered | 2.3.3 |
| D35 | Mirror kept in vision stages: colour optic on the sagittal plane, corruption after mirroring | 2.3.3 |
| D36 | Latency DR 5–15 ms from S1b, every arm | 2.3.5 |
| D37 | Second oracle O-geo (height strips) beside O-mat | 2.3.6 |
| D38 | Primary metrics: transition-window tracking / slip / falls + first-contact impact; stride descriptive | 2.3.4, 2.3.6 |
| D39 | Soft-ground entries `soft_soil`, `wet_clay` (k 4e2–2e3 N/m; measured walking sink 2.3–8.5 mm) for every arm from the headroom pilot / S1b; elastic only, mud and gravel out of scope | Phase 2.3 summary above, 2.3.8 |
| D40 | S2 layout = S1 layout ⊕ H* (1,269-D) with the trunks widened at the branch; S2 library has 12 entries, so its material layout seed 0 is a new layout, not the trunks' | 2.3.9 |

### Lessons from Phase 2 (checklist for S2/S3)

1. **Check the solver, not only the model.** A PhysX default (`enable_external_forces_every_iteration=False`) made
   every explicit actuator look like a bad servo model; replaying a known-good policy's actions isolated it.
2. **Reward economics before every training.** Run 2's fall exploit, the 0.10 m/s stride equilibrium, the
   coxa-spin exploit and the S1 leg overlap were all profitable under the reward as written. `reward_economy.py` /
   `gait_by_material.py` per-term ledgers found each one; walk ≫ stand ≫ fall is checked for every change.
3. **Gates are necessary, not sufficient.** Old π_S0 passed every gate with legs overlapping in sim, feet under the
   hips and a silently mis-scaled coxa term. Swing, gap and geometry probes plus video are part of every review.
4. **Inherited habits survive warm starts.** π_base's right-side crutch legs reappeared as a tucked right-back leg
   and lopsided turning through 14 runs of fixes; the from-scratch arm was symmetric and better at 0.30 × the budget.
5. **Measure at the right time.** The rest reach was read before the rest pose was applied; probe statistics were
   frozen from a curriculum-level-0 buffer. Both passed tests and silently shaped training.
6. **Stateful observations leak.** A frozen clock in the observation kept the standing policy in "swing" mode;
   absent signals are zeroed (the same flag the S2 clock dropout uses).
7. **Pre-declared stop rules.** The plateau rule (< 1.5 % return gain per 2,000 iterations) set both the π_S0 stop
   and the S1 budget; the pilot-3 regression watch stopped a breach instead of letting it run.
8. **Longer training exposes the reward's optimum.** The S1 leg overlap, the lift drift and the clip term appeared only
   after 4,000–8,000 iterations. S2 / S3 are several times longer: keep the regression watch (leg gap, lift per leg,
   landing reach, coxa range, clip term) at every milestone, and settle the known imperfections in the S2 reward.
9. **A ledger of one posture is not enough.** Removing half of `touchdown_reach` released another term's incentive
   (the reach-weighted coxa stroke) and the legs sprawled; before changing a term, check what it was holding back.

### Open items and next steps

**Left in S1:** S1-K ×5 for B1-K later (its own KiVi-style trunk, not needed before B1-K's S2); S1b once N_v is known.
All five S1 trunks (seeds 42–46) are trained and evaluated.

**Superseded 2026-10-02 by the order at the end of the Phase 2.3 section** (headroom pilot before N_v; S2 reward =
S1 reward, D38). Kept for the record — spec §12 order as of 2026-09-30: (1) **N_v benchmark** on the A4000 with a rendering + E_v
prototype (decides the S2 / S3 batch N_v × T_v for every arm, blind ones included); (2) **S1b** per seed at N_v × T_v;
(3) **S2 reward design** with the clock and clock-phase terms annealed together over the first 30 % of S2, checked
with the per-term ledger (4 cm lift floor weight ≈ 0.48 / s, D30 clip term, the front-leg reach target, and
`touchdown_reach` surviving the anneal while `cpg_coxa_contact` leaves); (4) the blind **O vs B0 headroom test** (O's
576-D M* strips go live) — if O ≈ B0 on transition terrain, anticipation has no value here and the vision runs are
stopped (spec §11); (5) only then the vision arms B2 and F (then B1-M, B1-K from S1-K; B3 optional), after the bench
items that gate S2 / S3.

**Bench (hardware), still provisional in sim:** AX-12A gains / compliance / backlash / latency, drop rate vs load,
static torque envelope, mast mass, per-surface μ and foam stiffness, IMU noise, L1/R1 toe lateral offset, D435 Z_min,
mount pitch, latent_age p99.

**Spec reconciled 2026-09-29:** seed counts (5 S1 trunks, 5 S1-K / B1-K with 3 as the first cut) and the KL anchor
(void since 2026-09-23) now read consistently; `eval_s0.py` refuses the demo-only Play tasks.

**Repo hygiene:** the pre-commit `insert-license` hook is broken (missing `.github/LICENSE_HEADER.txt`).

### Reproduction (final artefacts)

```
export OPENBLAS_NUM_THREADS=1
# π_S0 (from scratch, S0b task)
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 18000 \
  --run_name s0_coordlift_prox_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
# (stopped at 12,2xx on the plateau rule; model_12200.pt)
scripts/rsl_rl/eval_s0b_gates.sh <π_S0> coordlift_prox_final 1024
scripts/rsl_rl/eval_terrain_sweep.sh <π_S0> pis0cl 0 3 6 9 10 11 12 13 14 15 16 17 18 19 20

# S1 pilot 3 (seed 42, W 50), warm-started from π_S0 (widening, not the dropped S0 warm start)
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 8000 \
  --warmstart <π_S0> --run_name s1_pilot3_seed42
# stopped at 4,500 (watch breach), resumed:
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3500 \
  --resume --load_run 2026-09-26_13-50-10_s1_pilot3_seed42 --checkpoint model_4500.pt --run_name s1_pilot3_seed42_resume4500
scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0> 8000 6000

# W sweep (seed 42, W 25 / 75) and S1 seeds 43–46 at W 50 (each 8,000 iterations, own nohup log)
scripts/rsl_rl/run_s1_sweep_and_seeds.sh sweep
nohup scripts/rsl_rl/run_s1_sweep_and_seeds.sh seeds 50 > logs/s1_seeds_w50_driver.log 2>&1 &
# per-run evaluation: eval_s0b_gates.sh (TASK=Hexapod-S1-v0, 256), gait_by_material.py, milestone_review.sh <run> 7999
python scripts/rsl_rl/summarize_s1_runs.py s42=<run>:<tag> s43=… --seeds s42,s43,…

# Headroom pilot (phase2-progress §2.3.7–2.3.10), from the S1 seed-43 trunk
python scripts/tests/gait_by_material.py --headless --split compliance --k {400,600,1000,2000} --cmd {0,0.15} s43=<trunk>
python scripts/tests/test_s1.py
python scripts/tests/reward_economy.py --headless --task Hexapod-S1-v0 --ckpt <trunk> --cmd 0.15 0 0
python scripts/tests/material_field_test.py --headless --task Hexapod-S2-v0      # alone on the GPU
python scripts/rsl_rl/train.py --task Hexapod-S2-v0 --headless --num_envs 4096 --seed 43 --max_iterations 6000 \
  --branch_from <trunk> env.arm=<B0|O-mat|O-geo> --run_name s2_headroom_<arm>_seed43
scripts/rsl_rl/eval_headroom.sh b0=<ckpt> omat=<ckpt> ogeo=<ckpt>
```

Every run directory holds `params/env.yaml`, `params/agent.yaml` and `git/` (commit + diff). All numbers are
single-seed (seed 42) except §2.2.10–2.2.11 (W sweep at seed 42; S1 results across all five seeds 42–46).
