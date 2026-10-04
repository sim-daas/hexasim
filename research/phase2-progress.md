# Phase 2: Baseline Paradigm — Development Log

Chronological log for Phase 2 (the comparison baselines of the Shared Training Paradigm spec v2). Format as in
`hexapod-RL-tuning-guide.md`: Observed Behavior / Root Cause / Fix / Result. Every number below was produced by
the command next to it on the code of this phase; raw outputs live in `logs/` (JSON under `logs/eval/`).

Run-by-run detail (every training command, per-run and per-milestone tables, superseded runs) lives in
[`phase2-run-logs.md`](phase2-run-logs.md), verbatim; the sections below keep the design, root causes, fixes, reward
economics, decisions and the final results, with a summary where a section was mostly run log.

---

## Phase 2.0: Correctness fixes + S0a reconditioning (started 2026-09-21)

Scope agreed with the user: paradigm spec §12 step 1 (env-level correctness fixes) + the static torque
envelope (§4.2 prerequisite) + S0a (§4.2). S0b (commands, terrain, phase-free floors) and S1 infrastructure
(contract P, sequence buffer, TCN, KL anchor) are out of scope.

Decisions taken with the user before implementation:

| Topic | Decision | Spec deviation? |
| --- | --- | --- |
| Batch | 4,096 envs × 16 steps: the base run's actual batch and the compute limit | Yes — spec §1/§8.1 say 8,192 × 16, which is wrong for `model_14998.pt` |
| Isaac Sim / Lab | Stay on the installed version | No |
| Servo parameters | AX-12A datasheet defaults, flagged provisional; bench later | No (spec asks for measured values eventually) |
| V_0 | Small learned MLP (5-frame o_p history → v), trained online, output detached | Spec leaves V_0 undefined |
| Clock in S0a | Driven by the same ramped v_x the actor sees (GT → V_0), floor 0.8 Hz, per-env phase jitter | Spec only defines the clock from S0b (command-referenced) |
| S0a critic | sg[actor obs] + true base velocity (3) + base height (1) + per-foot normal/tangential force (12) + contact state (6) | Subset of contract P (full P at S1) |
| Rest pose | Femur +30°, tibia +30° (was −30°), warm start **folded** (see 2.0.4) | Spec §4.2 says "action offsets kept"; its own torque-envelope gate forces the change |
| Robot mass | URDF masses are the robot's exact masses; payload randomised U[0.1, 1.0] kg per env | Payload is a §1 dynamics-randomisation item brought forward to S0a at the user's request |

### 2.0.1 Runtime layout probe

`python scripts/tests/probe_layout.py --headless` → `logs/probe_layout.txt`.

* Joint order is grouped by type, `[coxa × 6, femur × 6, tibia × 6]`, legs RF, RM, RB, LF, LM, LB within each
  group. Tibia bodies and contact-sensor bodies share the same leg order. So `phases.repeat(1, 2)` in
  `cpg_stance_rigidity_penalty` was **correct** (femur 0–5, tibia 0–5); it is now replaced by an explicit
  per-joint leg lookup from joint names, and the order is asserted at env construction (`HexapodEnv.load_managers`).
* Toe offset `(0.095, 0, −0.09)` in the tibia frame puts the toe 0.2 mm above the ground at rest: confirmed.
* +coxa moves right toes forward (+x body) and left toes backward.
* Robot mass 3.22 kg (URDF), so the tripod-stance leg load is ≈ 10.5 N, not the "11.8 N robot / 3 ≈ 4 N" in
  the old `cpg_force_target` comment.

### 2.0.2 Correctness fixes (spec §12 step 1, v7 §2)

**Toe point.** *Root cause:* `foot_z` used the tibia-link origin + a 0.0892 m constant valid only at rest (v7 A13).
*Fix:* one helper `mdp.rewards.toe_pos_w` used by `cpg_z_target_l1_penalty`, `foot_clearance_reward`, the env's
contact-baseline EMA and `play_debug.py`. The 0.0892 constants and fallbacks are deleted; the baseline starts at 0
(ground-plane height) and is EMA-tracked from toe height on contact.

**Clock.** *Root cause:* floor 0.2 Hz (stall lock, v7 §2.2) and identical phases at every reset (v7 §2.3).
*Fix:* floor 0.8 Hz; per-env uniform global phase offset at reset (tripod structure kept). The clock reads
`env.v_in`, i.e. ground truth in `Hexapod-Flat-v0` and the GT → V_0 ramp in `Hexapod-S0a-v0`.

**Coxa terms.** *Root cause:* `cpg_coxa_velocity_reward` and `cpg_coxa_contact_reward` contradicted each other
and both used `abs(joint_vel)` (v7 §2.6). Only `cpg_coxa_contact` was weighted (+1.0) in π_base.
*Fix:* `cpg_coxa_velocity_reward` and the unused `cpg_z_target_reward` are deleted. `cpg_coxa_contact_reward` now
uses the **signed** backward sweep (per-leg sign from the probe), clipped to ±π rad/s per leg.

*Economic analysis.* In π_base this term dominated the reward: ~3 planted stance legs × |q̇_coxa| ≈ 1–2 rad/s
gives +3 to +6 per step, against forward_velocity 1.7 × 0.2 = +0.34. With `abs`, a coxa buzzing back and forth
while planted earned the full rate in both directions. Signed, the stance integral is the net backward stroke:
a proper stance stroke earns the same as before, buzzing earns ≈ 0, and a forward sweep in stance costs.
Residual risk, unchanged from π_base: a planted foot that slips backward still earns stroke reward without
progress. That becomes the job of `stance_rigidity_contact` (toe tangential velocity) from S0b.

**Stance rigidity.** v7 §2.6 asks to include the coxa or document the exclusion. *Decision: excluded,
documented in the config.* The backward coxa sweep *is* the stance stroke that `cpg_coxa_contact` pays +1 per
rad/s for. Adding the coxa at −0.5 per rad/s would halve the stroke incentive (and make a forward stance
sweep cost −1.5) — the same kind of contradiction v7 §2.6 removes.

**Duty factor.** Single source `HexapodEnvCfg.duty_factor = 0.5`; every CPG term reads it (no per-term
parameter), and `HexapodEnv` asserts 0.5 because the tripod offsets {0, π} assume it.

**Force target.** *Root cause:* `max_force = 4.0 N` from a wrong mass (the robot is 3.22 kg + mast, not 1.2 kg).
The target was ~2.6× below the real tripod load, so the term paid for unloading stance legs, i.e. for extra
legs in contact (plausibly part of π_base's 0.63–0.65 contact duty factor). *Fix:* peak target = per-env total
weight / 3, read from `body_mass` after the mast/payload startup events. *Economics:* penalty in good tripod
stance drops from ≈ −1.6 to ≈ −1 per step, which moves the optimum toward genuine tripod loading (the spec's
duty-factor gate). Still L1, never squared (CLAUDE.md rule 3). World-Z = surface normal holds on flat ground
only (marked `ponytail:`; project at S0b).

**Friction combine.** *Fix:* ground `PhysxRigidBodyMaterialCfg` with `friction_combine_mode="multiply"`; every
robot body μ = 1.0 via a startup material event. Effective μ = μ_ground.
*Test:* `python scripts/tests/friction_slide_test.py --headless` (gravity tilted by θ, ground μ 0.4, slip onset
atan 0.4 = 21.8°): drift 3.4 mm at 18°, 2.72 m at 26° → **PASS**. Negative control `--combine average`: 0.1 mm
and 1.4 mm → FAIL, as expected (μ_eff 0.7). The test discriminates.

**Ground μ = 0.75.** *Observed:* after the fixes, π_base fell in 40 % of evaluation episodes on
`Hexapod-Flat-v0` (0 % on the Phase 1 code). *Root cause (ablation, below):* the robot USD has no collider
material, so π_base trained with PhysX's default foot μ 0.5 averaged with ground 1.0 = **0.75 effective**; at
μ 1.0 its feet catch. *Fix:* ground μ 0.75 under multiply reproduces π_base's training friction exactly. It is
a flat-ground placeholder until the per-surface μ bench (spec §12 open item); the material library replaces it
at S0b.

**Camera mast.** Every task now carries a provisional 0.15 kg mast at (+0.10, 0, +0.10) m from the base origin
(added base mass + CoM shift, `camera_mast_mass` / `camera_mast_com` events). Blind arms carry it too (spec §1).
Weigh the real mast.

**Stale text** fixed: "63-dim obs", "tibia at −60°", `[0, π/3)` / "duty factor 5/6".

#### π_base fall-rate ablation

`python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-Flat-v0 --ckpt logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt [--ablate …]`
(256 envs, one 10 s episode each, deterministic, seed 0). The Phase 1 row runs the same script against a git
worktree of commit `fe78300` via `PYTHONPATH=…/hexasim_phase1/source/hexasim`.

| Environment | fall | v_x (m/s) | contact duty | clearance (m) | coxa > 5 Hz (rad²) |
| --- | --- | --- | --- | --- | --- |
| Phase 1 code (`fe78300`) | 0.0 % | 0.159 | 0.648 | 0.082 | 0.183 |
| All fixes, ground μ 1.0 | 39.8 % | 0.166 | 0.566 | 0.029 | 0.192 |
| … without mast | 27.7 % | 0.162 | — | 0.030 | — |
| … friction back to average (μ_eff 0.75) | 9.0 % | 0.181 | — | 0.049 | — |
| … neither | 6.6 % | 0.177 | — | 0.050 | — |
| **All fixes, ground μ 0.75 (current Flat env)** | **9.0 %** | 0.181 | 0.627 | 0.049 | 0.158 |
| … without phase jitter | 0.0 % | 0.179 | 0.636 | 0.050 | 0.149 |
| … without mast | 6.6 % | 0.177 | 0.623 | 0.050 | 0.154 |

Reading: at the matched μ, the remaining 9 % falls come from the random reset phase, which π_base never saw
(fixed-phase resets). That is a robustness gap S0a has to close, not a bug. The clearance drop 8.2 → 5.0 cm
comes from the 0.8 Hz clock floor (the clock no longer nearly stops at low speed). Note: the contact duty
factor (|F| > 1 N) of π_base is 0.63–0.65 in every configuration, outside the spec's 0.5 ± 10 % S0a gate
before S0a starts.

### 2.0.3 Static torque envelope — gate FAILED at the Phase 1 rest pose; rest pose changed

`python scripts/tests/static_torque_envelope.py --headless [--femur_deg F --tibia_deg T --payload M]`: settle on
six legs, lift tripod A, report tripod-B holding torque.

*Observed:* the implicit actuator's `applied_torque` reported ≈ 0.99 N·m (66 % of stall), but joint torques
computed from contact force × lever arm (((toe − joint) × F)·axis) gave **2.0 N·m = 134 % of the 1.5 N·m
AX-12A stall** for the femur. That is consistent with F·r ≈ 13.5 N × 0.15 m.
*Root cause of the discrepancy:* Isaac Lab's `applied_torque` for implicit PhysX drives is a PD estimate, not
the constraint force; it under-reads by ~2×. π_base's `joint_torque_excess` term (threshold 2.5 N·m) was
therefore reading a meaningless number. The gate uses lever-arm torques.

Rest-pose sweep (lever-arm torques, tripod stance, % of stall; `logs/pose_sweep.txt`):

| femur, tibia | femur | tibia | stance height |
| --- | --- | --- | --- |
| +30°, −30° (Phase 1) | **134 %** | 92 % | 0.123 m |
| +30°, 0° | 73 % | 32 % | 0.167 m |
| **+30°, +30° (adopted)** | **12 %** | 29 % | 0.168 m |
| +45°, 0° | 33 % | 3 % | 0.183 m |
| +15°, +30° | 51 % | 2 % | 0.158 m |

Payload check at the adopted pose (`logs/payload_sweep.txt`): +0.1 kg femur 13 % / tibia 30 %; +0.5 kg 14 % /
34 %; +1.0 kg 14 % / 42 % → PASS across the whole payload range (static; dynamic walking needs margin, which
remains).

### 2.0.4 Warm start at the new rest pose

*Re-centred attempt:* π_base's actions applied as residuals around the new pose (`--rest_tibia_deg 30`, ideal
actuators): 0 % falls but v_x = −0.004 m/s, contact duty 0.70 — it stands instead of walking. Rejected, as
agreed.

*Adopted — folded warm start* (`hexasim.rl.s0_ppo.warmstart_actor_state`): the rest-pose shift Δ (+60° per
tibia) is folded into the joint-position normaliser mean (μ − Δ), the last-action normaliser mean (μ − Δ/0.5)
and the output bias (−Δ/0.5). The widened 114-D actor therefore commands exactly π_base's joint targets at
initialisation; the new rest pose only moves the action-space centre, the reset pose and the clip window.
S0a has to move the posture into the feasible region itself.

`python scripts/tests/test_s0_ppo.py`: joint targets match π_base over 10,000 random observations to
2.7e-7 (float64 forward; the residual is the fp32 storage of the shifted means; in fp32 the forward itself
differs by ~4e-6 from summation order alone) → PASS. Normaliser fold 1.9e-14 → PASS. Value-loss backward leaves
every actor grad None → PASS. A frozen-actor S0PPO update changes the critic and not the actor → PASS.

### 2.0.5 AX-12A servo model and read pipeline

`hexasim/actuators/ax12a.py` (explicit torque, replaces `ImplicitActuatorCfg` in `PXMARK4_AX12A_CFG`; the
Flat task keeps the implicit PD so π_base stays reproducible):

```
q_goal = delay(quantise(q_cmd))                 bus latency 1–2 physics steps (5–10 ms), per env at reset
e      = q_goal − quantise(q)                   10-bit, 0.293°/tick, output-shaft potentiometer
dz     = margin·tick + backlash/2               margin 1 tick, backlash 0.5° total
u      = sign(e)·(punch + (1 − punch)·clip((|e| − dz)/(slope·tick), 0, 1)) if |e| > dz    slope 32, punch 32/1023
τ      = clip(τ_stall·(u − q̇/ω_nl), ±τ_stall)   stall 1.5 N·m, no-load 59 rpm; back-EMF is the only damping
```

Armature 0.003 kg·m² (reflected rotor inertia estimate). The USD also sets `physxJoint:jointFriction = 0.5`
(unitless), which π_base had as well; kept. All constants are provisional calibration knobs (v7 item 14).

*Standing test* (`python scripts/tests/ax12a_stand_test.py --headless`, 16 envs, 2 s after settling):

| Pose | root z | drops | mean \|err\| coxa / femur / tibia (rad) | saturated femur / tibia | q̇ rms tibia |
| --- | --- | --- | --- | --- | --- |
| S0a rest (tibia +30°) | 0.170 m | 1.0 % | 0.002 / 0.017 / 0.019 | 0 % / 0 % | 0.013 rad/s |
| π_base pose (tibia −30°) | 0.163 m | 6.9 % | 0.032 / 0.085 / 0.361 | 10 % / 28 % | 0.176 rad/s |

Stable, no oscillation at the feasible pose; the Phase 1 pose saturates the tibias, as the envelope predicted.
(The drop column is from the first, error-based drop model; see below.)

`mdp/proprio.py` `ProprioReader`: 10-bit reads, ZOH of q, q̇_t = (q_t − q_{t−2})/2Δt held when either endpoint
is held, 18-bit mask; self-check in `__main__` (quantum 0.128 rad/s) passes.

*Drop model, revised.* The first version, p = 0.005 + 0.41·|q_cmd − q|, produced 18 % drops while walking in
the smoke run. *Root cause:* at an infeasible target the tracking error grows without bound while the current
— the actual driver of supply sag — is capped at stall. *Fix:* p = 0.005 + 0.068·|τ|/τ_stall (≈ 5 % at 1 N·m,
7.3 % max). Smoke run after the fix: 6.2 % drops while walking. Provisional until the bus bench.

### 2.0.6 S0a environment and training

`Hexapod-S0a-v0` (`HexapodS0aEnv`, `HexapodS0aEnvCfg`):

* Actor 114-D = o_p 111 [q − q_rest, q̇, q_cmd − q, last action, IMU accel, IMU gyro, gravity, clock, mask] ⊕
  v_in 3. IMU noise σ 0.05 m/s² / 0.02 rad/s, gravity σ 0.02; noise and drops are applied in the env.
* Critic: same ⊕ 22-D privileged block, separate MLP [512, 256, 128] with its own normaliser.
* Actions clipped to the joint limits (servo CW/CCW) in joint space.
* V_0: MLP(5 × 111 → 128 → 64 → 3) with its own input normaliser, Huber (δ 0.1) against true base velocity,
  trained online each control step on the env batch; saved in the checkpoint. α ramps 0 → 1 from iteration 300
  to 1,500.
* Rewards: π_base's final weights (`HexapodFlatResumeEnvCfg`, verified against the π_base run's logged
  `env.yaml`) + the fixes above; `joint_torque_excess` threshold 2.5 → 1.2 N·m, because `applied_torque` is now
  the servo model's own torque (≤ 1.5 N·m) and the old threshold could never fire. Worst case −0.27 per step.
* Payload U[0.1, 1.0] kg on the base, per env, at startup (plus the 0.15 kg mast).

`S0PPO` (`hexasim/rl/s0_ppo.py`, `PPORunnerCfg_S0a`): actor LR fixed 1e-4, critic 3e-4, entropy 5e-4; actor
frozen until the critic has had ≥ 300 iterations **and** explained variance ≥ 0.8; the actor normaliser is
re-estimated in a shadow copy over 50 iterations and swapped in with the change folded into the first layer
(π(raw obs) unchanged), count then capped; std floored at 0.05.

### 2.0.7 S0a run 1 (seed 42) — FAILED: the policy stands

Warm start from π_base, 3,000 iterations (`logs/rsl_rl/hexapod_s0a/2026-09-21_14-09-06_s0a_seed42/`). The schedule
mechanics behaved as designed (actor frozen to 300 with critic EV 0.98, normaliser fold at 50, V_0 ramp, std ≥ 0.05,
6.2 % dropped reads), but the robot stood: 256-episode eval v_x 0.003 m/s, clearance 6 mm, 0 % falls, V_0 RMSE
0.024 m/s. **S0a gates not met.** Superseded: produced under the PhysX solver artefact of §2.0.9.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.7.

### 2.0.8 Diagnosis — π_base's gait does not survive explicit actuation

π_base's own actions from ground-truth observations replayed inside `Hexapod-S0a-v0`
(`scripts/tests/diag_obs_equivalence.py`, 64 envs; Flat control 0.177 m/s):

* Under **every explicit joint-effort actuator** tested — AX-12A model, stiffened variants, Isaac Lab's
  `IdealPDActuator` at Kp 20–160, 200 Hz / 1 kHz physics, any armature or joint friction — v_x ≤ 0.03 m/s.
* Under a **PhysX implicit drive** it walks: 0.178 m/s at π_base's gains, 0.190 m/s even at AX-12A-equivalent
  gains (Kp 9.2, Kd 0.24, 1.5 N·m). Softness and torque limit are not what break it.
* Insensitivity to all servo parameters pointed at the explicit-effort path itself (→ §2.0.9). Also noted: the S0a
  reset pose, action clip and payload together cut π_base's gait 0.19 → 0.03 m/s.

User decision (2026-09-21): find the cause before training again; the implicit drive is an acceptable fallback.
Every number here was later invalidated (flag off, §2.0.9).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.8.

### 2.0.9 Root cause: PhysX TGS applied explicit joint efforts only once per step

*Observed:* under π_base's ground-truth actions, explicit joints barely moved (q̇ rms 0.13–0.16 rad/s vs 1.7–2.7
under the implicit drive) while carrying *more* torque (≈ 2.5 N·m). Sim joint properties matched in both modes
(friction 0 — the USD `jointFriction` is not applied, so it is exonerated — velocity limit π rad/s, same limits).

*Isolation* (`python scripts/tests/actuation_air_test.py --headless --mode … [--armature A] [--physics_hz H]
[--ext_every_iter]`: robot floating in zero gravity, every joint tracking rest + 0.25·sin(2π·1 Hz·t);
`logs/actuation_air_test.txt`):

| Actuation (Kp 20, Kd 0.5, 3 N·m) | coxa / femur / tibia \|q_target − q\| (rad) | q̇ rms (ideal 1.11 rad/s) |
| --- | --- | --- |
| Implicit PhysX drive | 0.017 / 0.016 / 0.017 | 0.77 / 1.04 / 0.81 |
| Explicit IdealPD, armature 0 | 0.80 / 6.45 / 0.66 | unstable |
| Explicit IdealPD, armature 0.003 | 0.14 / 0.15 / 0.14 | 0.02 / 0.01 / 0.00 — joints do not move |
| Explicit IdealPD, armature 0, 1 kHz / 2 kHz | 0.52–0.24 / 1.33–0.91 / 0.25 | 0.04–0.99 |
| **Explicit IdealPD, armature 0.003, `enable_external_forces_every_iteration=True`** | **0.014 / 0.014 / 0.014** | **0.86 / 0.96 / 0.89** |
| Explicit IdealPD, armature 0, flag on | 0.60 / 2.76 / 0.29 | unstable |

*Root cause:* Isaac Lab's default `PhysxCfg` uses the TGS solver with `enable_external_forces_every_iteration=False`
(every run printed the warning "TGS solver with enable_external_forces_every_iteration=False may cause noisy
velocities"). Explicit actuator torques are passed to PhysX as joint actuation (external) forces, and in that
mode they barely act on this articulation. The implicit drive is solved inside the iterations and is unaffected.
Separately, an explicit PD with armature 0 is numerically unstable here: back-EMF/Kd damping on links with
I ≈ 7e-5–5e-4 kg·m² violates b·dt/I < 2 at 5 ms. The 0.003 kg·m² reflected-rotor armature of the AX-12A model
keeps it stable.

*Fix:* `HexapodS0aEnvCfg` sets `sim.physics = PhysxCfg(enable_external_forces_every_iteration=True)`. The Flat
task (implicit PD, π_base reference) is unchanged. **No switch to the implicit drive is needed**: the explicit
AX-12A model keeps its compliance curve, torque–speed falloff, backlash, quantisation and latency.

*Verification:* π_base's ground-truth actions in `Hexapod-S0a-v0`, datasheet AX-12A model, flag on
(`logs/diag_fixed.txt`): **0.165 m/s** with the old reset pose and no clip/payload (Flat control: 0.177 m/s);
0.096 m/s with S0a as configured (V_0 clock, new reset pose, clip, payload) — the gap S0a has to close. Standing
test re-run (`logs/ax12a_stand_test_fixed.txt`): S0a rest pose 0.170 m, errors 0.008 / 0.043 / 0.013 rad, no
saturation, 1.1 % drops; Phase 1 pose sags to 0.127 m, tibia 7 % saturated.

*Invalidated:* every explicit-actuator number in 2.0.7–2.0.8 (and the first standing-test table in 2.0.5) was
produced with the flag off and describes the solver artefact, not the servo. Run 1 is superseded by run 2
(2.0.10). The lever-arm torque envelope (2.0.3) used the implicit Flat task and stands.

### 2.0.10 S0a run 2 (seed 42, PhysX fix) — stopped at iteration 700: termination exploit

Run 2 (`…/2026-09-21_15-11-28_s0a_seed42_r2`), same command as run 1 with the PhysX fix. While frozen (= π_base) the
robot walked with 57–59 % falls; after the unfreeze at 300 PPO **learned to fall** (93 % by 340, 99.7 % by 540,
episode length 322 → 64 steps; V_0 α still ≈ 0.03). Stopped at 700.

*Root cause — reward economics.* π_base's positive Phase 1 return rested on the old `abs(q̇)` coxa term: +8.81 per 10 s
in Phase 1 vs **+0.68** for the same gait under the signed term, i.e. ~92 % of it paid for back-and-forth coxa motion.
With that gone and the toe-point swing error ~3× larger (`cpg_z_target` −4.66 → −14.55), every surviving behaviour had
a negative per-step reward (≈ −0.575) while a fall cost −5 once. Ending the episode was cheapest; PPO found it in 40
iterations. Run 1's standing optimum was the same economics. → reward recalibration (§2.0.11).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.10.

### 2.0.11 Reward recalibration for S0a (economic analysis, measured)

Tool: `python scripts/tests/reward_economy.py --headless --ckpt <run-2 model_250.pt> [--set …]`. It runs the frozen
warm-started actor (= π_base, "walk") and zero action ("stand") in `Hexapod-S0a-v0`, 256 envs × one episode, α = 0.
It reports per-term reward per live step (dt-scaled, as PPO sees it) and the discounted return (γ 0.99). A fall
on the first step is scored analytically as the terminating term (−5 × 0.02 = −0.1).

| Setting | walk: return (per step), falls | stand: return (per step) | fall |
| --- | --- | --- | --- |
| Run-2 rewards (L1 `cpg_z_target` −10, alive 1, stand_still −10) | −49.7 (−0.567), 56 % | −43.5 (−0.436) | **−0.1 — best** |
| Toe-swing kernel +5, alive 5 | +0.32 (+0.018) | **+5.45** (+0.059) — best | −0.1 |
| Kernel +5, alive 10, stand_still −50 | +5.53 (+0.091) | **+7.44** (+0.079) — best | −0.1 |
| **Kernel +5, alive 10, stand_still −75 (adopted)** | **+3.48** (+0.074) — best | +2.47 (+0.029) | −0.1 |

Economic reading:
* The L1 swing penalty was the root of the inversion: −0.47/step walking, −0.38/step standing, against alive
  +0.02. π_base survived that term in Phase 1 only because it tracked the *tibia origin* almost perfectly
  (−0.009/step); on the toe point, which moves differently as the tibia pitches, the same gait misses by
  ~0.8 step-heights per swinging leg.
* *Fix:* `cpg_z_target_kernel_reward` (S0a only; the Flat task keeps the L1 form): per swinging leg
  exp(−(e/σ)²), σ = 0.02 m, averaged over swinging legs, weight +5. Bounded in [0, 1] per step, same target
  and swing-only masking, so a dragged leg earns ≈ 0.33 and cannot farm it. Measured: 0.39 for frozen π_base,
  0.33 standing — a small margin today, up to 1.0 for a gait that tracks the toe target.
* The alive bonus is raised so that survival has positive value (falling loses the stream). stand_still is
  scaled with it (at v = 0 it costs 1.5 × alive), as in Phase 1's intent that stand_still cancels alive. That
  puts standing below even the falling-prone π_base gait.
* Adopted ordering: π_base-like walking with 56 % falls +3.48 > standing +2.47 > falling −0.1. Estimated for
  better gaits (per-step values held, full episodes): the same gait without falls ≈ +7.3; at v ≥ 0.1 m/s,
  where stand_still vanishes, ≈ +12. Every improvement from the warm start raises return.
* Stance rigidity is the largest remaining walking-specific cost (−0.078/step vs −0.003 for π_base in Phase 1):
  the compliant AX-12A and the payload let stance joints move. It is left at π_base's weight (spec: full CPG
  shaping in S0/S1). Watch it; it is the next lever if the policy drifts toward standing.

### 2.0.12 S0a run 3 (seed 42, recalibrated rewards) — walks; 3 of 4 gates pass

Run 3 (`…/2026-09-21_15-39-59_s0a_seed42_r3`, 3,000 iterations, kernel σ 2 cm; reproduction in §2.0.15). Falls 57 %
frozen → 0 % from iteration 800; the V_0 ramp to α = 1 caused no regression; the signed coxa stroke reached +7.2 per
10 s (a genuine stroke, vs +0.68 for frozen π_base).

Eval (256 episodes, V_0 only, AX-12A, drops, payload): falls **0 %**, contact duty **0.513**, coxa > 5 Hz 0.143 (all
PASS), v_x 0.262 m/s, V_0 RMSE 0.037; clearance first scored as a FAIL (2.9 cm — metric bug, §2.0.13). Per-leg duty
exposed that π_base's right-middle leg was a permanent crutch (duty 0.99); run 3 reduced it (0.79) but not away.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.12.

### 2.0.13 Clearance metric corrected; the remaining imbalance is the right-middle leg

*Observed:* the gate table reported 2.9 cm clearance while the reward-economics tool put π_S0a's apex at
3.7 / 3.9 / 4.1 cm (q25 / 50 / 75), on the 3.8 cm target. *Root cause:* `eval_s0.py` averaged every lift-off
including sub-5 mm stutter micro-lifts. *Fix:* a swing counts only if its apex is > 5 mm; quartiles reported;
micro-lifts counted separately per leg (`logs/eval/*_v2.json`).

Re-scored: π_base barely stepped with RM and RB (0.10 / 0.11 real swings/s vs ≈ 0.8 elsewhere) — **two crutch legs
since Phase 1**. Run 3 fixed RB; RM now mostly stutters (1.85 micro-lifts/s, 0.52 real swings/s).

*Clearance gate reinterpreted with the user (2026-09-21):* π_base's 6–8 cm apex was high-stepping produced by its
tibia-origin target, so "± 10 % of π_base" does not measure the gate's intent; π_S0a's tight tracking of the 3.8 cm
target is adequate.

*Next (run 4):* kernel σ 2 → 1 cm, continued from run 3 (economics, run-3 actor: σ 1 cm walk +32.0 / stand +0.65 /
fall −0.1; stutter micro-lifts earn ≈ nothing). Gotchas found launching it: a Hydra `agent.resume=True` is silently
overwritten by `cli_args` (use `--resume`); pin `env.v0_ramp_steps=[0,1]` because the ramp is keyed to the
per-process step counter.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.13.

### 2.0.14 S0a run 4 (σ 1 cm, continued from run 3) — leg imbalance resolved; π_S0a

Run 4 (1,500 iterations from run 3, σ 1 cm; `…/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt`). No falls after
the first iteration; swing kernel 3.95 → 4.14 of 5.

Eval (256 episodes, V_0 only, AX-12A, drops, payload): falls **0 %**; contact duty **0.487** (per leg 0.43–0.54);
**every leg 1.48–1.50 real swings/s** (RM micro-lifts 1.85 → 0.06); toe apex 3.7 / 3.8 / 3.9 cm; coxa > 5 Hz 0.134;
v_x 0.291 m/s; V_0 RMSE 0.031. The RM/RB crutch legs present since Phase 1 are gone.

**π_S0a = `…/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt`**; all S0a gates met (clearance as
reinterpreted). Caveats: one seed; product of runs 3 + 4 (4,498 iterations ≈ 295 M env steps after the warm start);
servo, drop and IMU parameters provisional; flat ground at μ 0.75.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.0.14.

### 2.0.15 Phase 2.0 closure — reproduction, spec checklist, deviations, open items

**Reproducing π_S0a** (Isaac Sim / Isaac Lab as installed, rsl_rl 5.0.1, one RTX A4000):

```
# run 3 — warm start from π_base (code: commit fe78300 + logs/…/2026-09-21_15-39-59_s0a_seed42_r3/git/hexasim.diff)
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --warmstart logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt agent.run_name=s0a_seed42_r3 \
  'env.rewards.cpg_z_target.params.sigma=0.02'
# run 4 — continuation (code: commit 5face31 + logs/…/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/git/hexasim.diff)
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 1500 \
  --resume --load_run 2026-09-21_15-39-59_s0a_seed42_r3 --checkpoint model_2999.pt \
  --run_name s0a_seed42_r4_from_r3 'env.v0_ramp_steps=[0,1]'
# evaluation
python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0a-v0 \
  --ckpt logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt
```

Every run directory holds `params/env.yaml`, `params/agent.yaml` and `git/` (commit + diff); the values above were
checked against them. Runs 1 (`…14-09-06_s0a_seed42`) and 2 (`…15-11-28_s0a_seed42_r2`) are superseded (2.0.9,
2.0.10) and kept only as evidence. π_S0a comes from a two-segment lineage (σ 2 cm for 3,000 iterations, then σ 1 cm
for 1,500); the final configuration has not been run from π_base in a single segment.

**Statistics.** Gate numbers are single-seed, 256 deterministic episodes. 0 falls in 256 → 95 % upper bound on the
fall rate ≈ 1.2 % (rule of three), inside the +2 pp gate. Per spec §4.2, S0 is one shared run; seed variance is
reported from S1 onward, conditional on π_S0 — state this in the paper.

**Spec checklist (paradigm spec v2 §4.1–4.3, §12 step 1–4) for S0a**

| Item | Status |
| --- | --- |
| §4.1 audit of π_base | Done (spec table) + corrected: batch 4,096 × 16; v_x 0.159 m/s over 256 episodes (not 0.209); two crutch legs; π_base's return rested on the abs-coxa term (2.0.10) |
| §4.2 prerequisite: static torque envelope | Done in sim (2.0.3), incl. payload up to +1.0 kg; rest pose changed. Bench measurement open |
| §1 toe point, friction combine (with sim sliding test), joint-velocity pipeline, dropped reads + mask | Done (2.0.2, 2.0.5); friction test with negative control |
| §4.2 S0a: AX-12A model, drops, velocity pipeline, obs noise, V_0 substitution ramp, action clip, std floor 0.05, normaliser re-estimation | Done (2.0.5, 2.0.6) |
| §4.3 widening: function-preserving, zero-init new columns, std carried, per-block handling, 1e-6 test | Done; test passes in float64 (2.7e-7; fp32 summation order alone gives ~4e-6) |
| §8.2 gradient tests | Value-loss → actor grads None: done. PPO → encoder and L_probe/L_jepa → FAH: no encoders/FAH exist yet (S1+) |
| S0a gates | Fall, duty, coxa > 5 Hz: PASS. Clearance: reinterpreted with the user (2.0.13) |
| §4.2 from-scratch control arm | Pending — spec runs it "through S0a/S0b at the same budget", so it follows S0b |
| §12 step 1 remaining: contract P (683-D), sequence buffer, β-NLL, phase-free floors, remaining routing tests | Pending — S0b (floors) and S1 (the rest) |

**Deviations from the paradigm spec / v7, with reasons** (the spec text is unchanged; reasons in the sections cited)

| # | Deviation | Where |
| --- | --- | --- |
| D1 | Batch 4,096 × 16, not 8,192 × 16 (the base run's real batch; compute limit) | Decisions table |
| D2 | V_0 = 5-frame o_p MLP trained online in the env, output detached | 2.0.6 |
| D3 | S0a clock driven by the same ramped v_x the actor sees (spec defines the clock only from S0b) | Decisions table |
| D4 | S0a critic privileged block = 22-D subset of contract P | 2.0.6 |
| D5 | Rest pose femur/tibia +30° (spec: "offsets kept"), forced by the §4.2 torque gate; warm start folded | 2.0.3, 2.0.4 |
| D6 | Payload U[0.1, 1.0] kg brought forward into S0a (user request) | Decisions table |
| D7 | Flat-ground μ 0.75 = π_base's effective training friction (placeholder until per-surface bench) | 2.0.2 |
| D8 | Drop probability driven by servo load, not tracking error | 2.0.5 |
| D9 | Coxa excluded from `cpg_stance_rigidity` (documented, as v7 §2.6 allows) | 2.0.2 |
| D10 | S0a swing term = bounded toe-kernel reward (σ 1 cm, ×5); alive ×10; stand_still −75 (reward economics) | 2.0.11, 2.0.13 |
| D11 | PhysX `enable_external_forces_every_iteration=True` in S0a (required for explicit servo torques) | 2.0.9 |
| D12 | Duty-factor gate measured as contact duty (|F| > 1 N); clearance gate reinterpreted | 2.0.2, 2.0.13 |

**Open items carried forward**

* Bench (hardware, deferred by the user): AX-12A gain/compliance/backlash/latency, bus drop rate vs load, static
  torque envelope on the robot, mast mass, per-surface μ, IMU noise. Every sim value above is provisional.
* S0b: commands + command-referenced clock (replaces D3), terrain curriculum, phase-free floors, surface-normal
  force projection (currently world-Z, flat only), full dynamics randomisation (recommended to start here), the
  from-scratch control arm, S0b gates incl. tracking at the hardware drop rate.
* Tooling: the pre-commit `insert-license` hook needs `.github/LICENSE_HEADER.txt`; `play_debug.py:229` SIM115
  (pre-existing).

---

## Phase 2.1: S0b — commands, clock, standing, terrain, dynamics randomisation (started 2026-09-21)

Scope: paradigm spec §4.2 S0b on top of π_S0a. Output: π_S0 (warm start + KL anchor for every arm).

Decisions taken with the user before implementation:

| Topic | Decision | Spec deviation? |
| --- | --- | --- |
| Terrain | Mild mix, 10 levels: flat 15 %, rough ±0 → 1.5 cm, pyramid slopes up/down 0 → 10°, bumps 0 → 2 cm. Generator laid out for 15 levels (`difficulty_range = (0, 10/15)`), so +5 harder levels (stairs, rougher, steeper) later is a config change with levels 0–9 unchanged in distribution | No (spec: "easy-to-medium") |
| Dynamics randomisation | Full v7 §5.6 set brought into S0b, provisional ranges until the bench | Timing not fixed by the spec |
| Zero command | **Standing mode**: clock frozen (f = 0), every CPG term masked, stance floor on all legs + all-feet-contact bonus, tracking asks v = 0 | Spec silent (D13) |
| Friction / compliance | Per-env μ + contact compliance (per-patch materials are the S2/S3 experiment) | See D14 |
| Command resampling | Every 5 s (two segments per episode), same marginals (10 % zero, 30 % with v_y/yaw) | Spec says "episodes" (D15) |
| v_y / yaw ranges | ±0.05 m/s, ±0.25 rad/s | Spec: "small" |

### 2.1.1 Implementation

`Hexapod-S0b-v0` = `HexapodS0bEnv(HexapodS0aEnv)` + `HexapodS0bEnvCfg(HexapodS0aEnvCfg)`:

* **Commands** (`mdp/commands.py`, `mdp/velocity_command.py`): Isaac Lab's `UniformVelocityCommand` with
  `rel_standing_envs = 0.1`; 70 % of segments get v_y = yaw = 0. Actor obs 117-D = o_p (111) ⊕ V_0 (3) ⊕
  twist_cmd (3). Critic adds the terrain-aware 22-D privileged block and 12-D DR parameters (151-D).
* **Clock**: `f_ref = clamp(v_cmd,x / 0.20, 0.8, 2.5)` from the command (replaces D3), × 0 when standing.
  `env.cpg_active` masks every `cpg_*` term (all 1 in Flat/S0a, so those tasks are unchanged).
* **Terrain** (`mdp/terrain.py`): generator above (4 × 4 m tiles, 5 cm cells, 2 mm vertical quantum; Isaac Lab's
  random-uniform field ignores difficulty, so a wrapper scales its amplitude). A height map (2 cm grid, one
  downward raycast of the terrain mesh at startup) gives `env.terrain_z` / `env.terrain_normal`; the Flat/S0a
  env returns z = 0 / world z. Used by: termination (root height *above terrain* < 5 cm), `non_foot_body`,
  critic base height, toe heights in the swing floor, reset of the contact baseline, eval clearance.
* **Surface-normal forces** (`mdp.rewards.foot_forces_nt`): F_n = F·n(toe xy), F_t = |F − F_n n|, used by
  `cpg_force_target` and the critic. Closes the v7 §2.6 world-Z item.
* **Curriculum** `terrain_levels_cmd`: Isaac Lab's `terrain_levels_vel` promotes at tile/2 = 2 m, out of reach at
  0.05–0.20 m/s. Up: survived, commanded path ≥ 0.3 m, displacement ≥ 0.8 × commanded path. Down: fell or
  displacement < 0.4 × path. Start levels 0–2.
* **Rewards** (§2.1.3): tracking (`track_lin_vel_xy_exp` σ 0.05 ×10, `track_lin_vel_xy_l1` ×−10,
  `track_ang_vel_z_exp` σ 0.25 ×3) replace forward_velocity / lateral_velocity / stand_still (its 0.1 m/s
  threshold would punish the 0.05 m/s commands). Phase-free floors (spec §5), introduced here: `swing_clearance`
  (per foot, peak toe height above terrain over the airborne interval, min(peak/3 cm, 1) at touchdown, ×5) and
  `stance_rigidity_contact` (legs in sensor contact: toe tangential speed + 0.1·Σ|q̇| incl. coxa, ×−0.5);
  `standing_contact` (feet-in-contact fraction, standing only, ×1).
* **DR** (all provisional, bench): foot μ_s U[0.3, 1.1], μ_d = μ_s·U[0.8, 1], restitution U[0, 0.1], 50 % of
  envs compliant with k log-U[2e3, 5e4] N/m, c log-U[10, 200] N·s/m (64 material buckets); payload
  U[0.1, 1.0] kg (S0a) + base CoM x/y ±1 cm; servo stall ×U[0.9, 1.1], compliance slope {16, 32, 64},
  backlash U[0.3°, 1.0°], Coulomb joint friction U[0, 0.05] N·m (new in the AX-12A model), latency 1–2 physics
  steps (5–10 ms, as S0a); per-episode IMU bias N(0, 0.05 m/s²) / N(0, 0.01 rad/s). Servo parameters are per env,
  resampled at reset.
* **Warm start** (`rl/s0_ppo.widen_actor_state`, `train.py --warmstart <π_S0a>`): 114 → 117 by zero columns,
  identity normaliser stats for the new dims (re-estimated in the 50-iteration burn-in and folded), std carried,
  V_0 weights + optimiser carried. Critic fresh; actor frozen until iteration ≥ 300 and EV ≥ 0.8 (S0PPO).

### 2.1.2 Bugs found while building (Observed / Root cause / Fix)

**Foot materials had no effect.** *Observed:* `scripts/tests/foot_material_test.py` — every env slid at 30°
(tan 0.577) and none at 20° regardless of its sampled μ; no compliance sink. That is PhysX's default material
(μ 0.5). *Root cause:* the tibia colliders are **instance proxies** (the robot USD's mesh prims are
instanceable); `sim_utils.bind_physics_material` applies to collider descendants and silently skips proxies,
so nothing was bound. *Fix:* author the physics-purpose binding (strongerThanDescendants) on the tibia
rigid-body prim; binding resolution inherits it down to the collider (`mdp/events.py`).

**Kit crashed at start (`free(): invalid pointer`) for every task, S0a included.** *Root cause:* `train.py`
loads the task config (Hydra) before starting the simulation app; the config now imported USD (`pxr`) at module
level, twice — in `mdp/events.py` and through Isaac Lab's `UniformVelocityCommand` class. *Fix:* `pxr` imported
inside the event function; the command class lives in `mdp/velocity_command.py`, referenced lazily by the
config's `class_type` string, as Isaac Lab does.

### 2.1.3 Verification before training

* `python scripts/tests/test_s0_ppo.py` → `S0 checks OK (float64): warm-start max|Δ| = 2.73e-07, fold max|Δ| =
  1.87e-14, S0a→S0b max|Δ| = 0.00e+00` (widened actor = π_S0a over 10,000 random observations with random
  twist_cmd).
* `python scripts/tests/foot_material_test.py --headless` → **PASS** (`logs/foot_material_test.txt`), 256 envs,
  zero actions, flat terrain. 20°: 157/157 envs with μ_d > tan θ + 0.15 hold (max drift 4.5 mm). 30°: 92/92 envs
  with μ_s < tan θ − 0.05 slide, 73/73 with μ_d > tan θ + 0.15 hold. So μ_eff = sampled foot μ (combine multiply
  against the terrain's 1.0). Near the boundary (μ_d within 0.12 of tan 30°) 5 envs slip 6–11 cm: they land from
  the spawn height with tangential speed, and uphill legs unload — reported, not tuned away. Compliance:
  compliant envs sink 0.37 mm more than rigid ones on average (max 0.97 mm), Spearman(k, sink) = −0.91; effective
  per-foot stiffness ≈ 2.5 × the material's nominal k (PhysX applies k per contact point). The foam fit on the
  bench must therefore be done against measured sink in sim, not against the nominal k.
* Servo DR: compliance slopes {16, 32, 64} drawn 96/70/90 of 256.

**Reward economics** (`scripts/tests/reward_economy.py --task Hexapod-S0b-v0 --ckpt <π_S0a> --cmd … --terrain_level
0`, per step, dt-scaled; `logs/s0b_economy_v1.txt`, `_v2.txt`). "Walk" is π_S0a widened, which ignores the
command and walks 0.206 m/s at the 0.8 Hz clock (0.244 m/s at 1.0 Hz). "Walk@cmd" is estimated: measured walk,
tracking terms replaced by perfect tracking (+0.24), coxa term scaled to the commanded speed.

| cmd v_x | σ_lin | Stand | Walk (π_S0a, 0.21 m/s) | Walk@cmd (est.) | Fall |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.10 | +0.366 | +0.215 | ≈ +0.39 | −0.1 once |
| 0.05 | **0.05** | **+0.285** | +0.189 | ≈ +0.39 | −0.1 once |
| 0.10 | 0.10 | +0.275 | +0.244 | ≈ +0.41 | |
| 0.10 | **0.05** | **+0.205** | +0.207 | ≈ +0.41 | |
| 0.20 | 0.05 | +0.181 | +0.283 | — | |
| 0 (standing) | 0.10 | **+0.478** | +0.369 (walks 0.012 m/s, 2 % falls) | — | |

With σ 0.1, standing when commanded 0.05 m/s was within ≈ 5 % of walking at the command — the classic
"stand when asked to go slow" optimum. σ 0.05 widens that margin to ≈ 37 %; the L1 term keeps a gradient far
from the command, where the kernel is flat. At zero command, standing beats stepping by 0.11/step. Coxa-term
overspeed bias: its slope (≈ +20 per m/s over 3 stance legs) against the tracking terms puts the equilibrium
≈ 5 mm/s above the command — inside the 0.03 m/s gate.

Baseline, π_S0a on S0b (`eval_s0.py --task Hexapod-S0b-v0 --cmd 0.10 0 0 --terrain_level 0`, 64 envs): v_x 0.205,
cycle RMSE 0.110 m/s, stride 0.241 m (reference 0.125), falls 1.6 %, V_0 RMSE 0.092 m/s (0.031 on S0a flat — the
DR moves it off-distribution; it keeps training online in S0b). At zero command it keeps stepping (clock frozen):
0.16 m drift per episode.

### 2.1.4 S0b run 1 (seed 42)

Warm start from π_S0a, 3,000 iterations, 1.76 s/iteration (`logs/rsl_rl/hexapod_s0b/2026-09-21_19-59-42_s0b_seed42_r1/`).
The curriculum already moved π_S0a to mean level 6.4 during the frozen burn-in (the "up" rule counts overspeed as
progress).

Gates at 2,999 (`eval_s0b_gates.sh`, 256 episodes, L0 / L5): falls 0 %, cycle RMSE ≤ 0.025, drops ≈ drop-free, stride
✓ at 0.15 m/s (−4 %) but **✗ at 0.10 m/s (+18 %**, a residual 0.019 m/s overspeed). Both eval speeds sit below the clock
knee (f_ref = 0.8 Hz). The right-side imbalance was partly back (right stride 0.12–0.13 m vs left 0.17 m, RB duty 0.39).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.4.

### 2.1.5 Standing mode not learned — one leg held in the air

**Observed:** at zero command the policy stops walking (0.08–0.15 swings/s per leg) but freezes mid-gait with the
right-back leg lifted (RB contact duty 0.20) and jitters (Σ|q̇| 6.8 rad/s, 5.7 cm drift per episode).
**Root cause:** the standing incentive was too weak. `standing_contact` ×1 pays 1/6 per step per planted leg,
and `stance_rigidity_contact` only looks at legs in contact — a lifted, jittering leg costs nothing.
Measured (`reward_economy.py --ckpt <run 1> --cmd 0 0 0`): rest-pose standing beat the policy's pose by only
≈ 0.046/step (dt-scaled). **Fix:** `standing_contact` ×5 (a lifted leg now costs 0.83/step) and a new
standing-only `standing_joint_speed` penalty (−0.2·Σ|q̇|, all joints). Same measurement, new weights
(`logs/s0b_economy_v3.txt`): rest pose +0.554 vs the policy +0.446 per step on L0 (+0.543 vs +0.447 on L5), gap
0.108. Only standing segments are affected, so the walking economics above are unchanged.

The 0.10 m/s overspeed is not treated as a reward bias: it shrank with training (stride 0.167 → 0.148 m between
iterations 1,500 and 3,000) and 0.15 m/s undershoots slightly, so the policy is still moving off π_S0a's longer
stride rather than being pulled to a faster speed.

### 2.1.6 S0b run 2 (continuation of run 1, standing fix)

Run 2 (+2,000 iterations; `…/2026-09-21_21-36-27_s0b_seed42_r2_from_r1/model_4998.pt`). Gates: falls 0 %, cycle RMSE
≤ 0.021, drops ✓, stride ✓ at 0.15 (0.187 vs 0.1875 m) but **✗ at 0.10 (+16 %)** — v_x plateaued at 0.117, so §2.1.5's
"still leaving π_S0a's stride" reading was wrong. Left/right stride asymmetry gone; RB duty 0.41. Standing: legs rarely
swing but contact flickers (per-leg duty 0.33–0.73; weight on 3–4 legs), 0.27 rad/s joint jitter, 5 cm drift per 10 s.
V_0 RMSE under DR 0.047 m/s (S0a flat 0.031). Status then: π_S0 not declared; open items were the 0.10 m/s equilibrium,
standing load sharing, the from-scratch control arm and harder terrain.

**Deviations added in S0b** (spec text unchanged):

| # | Deviation | Where |
| --- | --- | --- |
| D13 | Standing mode at zero command (clock frozen, CPG terms masked, standing terms) — spec silent | 2.1.1, 2.1.5 |
| D14 | Per-env contact physics on the foot material (terrain μ 1, multiply), not a sampled ground μ: one terrain mesh carries one material. μ_eff equals the sampled value (2.1.3) | 2.1.1 |
| D15 | twist_cmd resampled every 5 s, not per episode | Decisions |
| D16 | Tracking gate on gait-cycle-averaged v_x (per-step RMSE includes the within-stride oscillation; reported alongside) | 2.1.3 |
| D17 | Full DR brought into S0b (spec does not fix its stage) | Decisions |

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.6.

### 2.1.7 Diagnosis of the 0.10 m/s stride equilibrium; standing load sharing

**Method.** `reward_economy.py --cmd 0.10 0 0 --obs_cmd <c> 0 0` scores every reward against a 0.10 m/s command
while the actor *sees* command c, which drives the run 2 policy at other speeds (the clock is 0.8 Hz for all of
them). Run 2 checkpoint, level 0, 256 envs (`logs/s0b_stride_diag.txt`), dt-scaled reward per step:

| actual v_x (episode mean) | 0.085 | 0.098 | 0.107 | 0.117 | 0.129 |
| --- | --- | --- | --- | --- | --- |
| total | 0.444 | 0.481 | **0.494** | 0.492 | 0.470 |
| tracking (exp + L1) | 0.108 | 0.134 | **0.142** | 0.136 | 0.110 |
| cpg_coxa_contact | 0.122 | 0.127 | 0.131 | 0.133 | 0.136 |
| cpg_z_target | 0.070 | 0.073 | 0.074 | 0.076 | 0.077 |

**Root cause.** The reward itself peaks near 0.11 m/s, so the policy is at the reward optimum, not stuck.
(1) Main part: both tracking terms scored the *instantaneous* body velocity. The within-stride surge is skewed,
so the exp kernel and the L1 term are maximised when the typical (modal/median) instant is at the command, which
puts the mean ≈ 7 % above it. The gate — and "walk at v" — is the mean. (2) Smaller: the coxa-sweep and swing terms
grow with speed (≈ +0.46 per m/s together).

**Fix.** One term `track_lin_vel_avg` on the body velocity averaged over about one gait period (EMA, τ = 1.25 s =
one period at the 0.8 Hz floor): exp(−|e|²/0.05²) − |e|₁, weight 10 — the same magnitudes as the per-step pair it
replaces (`track_lin_vel_xy_l1` deleted). Same sweep, new reward (`logs/s0b_stride_diag_v2.txt`): tracking now
peaks between 0.098 and 0.107 m/s episode-mean (which includes the start from rest), total reward between 0.10
and 0.11. Near the command the tracking slope (≥ 1.2 per m/s at 6 mm/s error) exceeds the speed-seeking terms'
0.46, so no change to the coxa term.

**Standing.** New standing-only `standing_load_share` = Σ_feet |F_n − W/6| / W (0 = even load, ≈ 1 on three
legs), L1 on the terrain-normal force, weight −5: a continuous signal where the 1 N contact test flickers.
Economics at zero command (`logs/s0b_stride_diag_v2.txt`, run 2 policy vs rest pose): rest pose +0.547, policy
+0.439 per step at weight −2 (load share −0.008 vs −0.039); at −5 the gap grows from 0.108 to ≈ 0.155.

### 2.1.8 S0b run 3 (continuation of run 2: averaged-velocity tracking, load sharing)

Run 3 (from run 2, `track_lin_vel_avg` + `standing_load_share`; `…/2026-09-21_23-04-11_s0b_seed42_r3_from_r2/model_6997.pt`).
**Worse:** 0.10 m/s → v_x 0.128 (stride +27 %), per-step error 0.059 (run 2: 0.031) — the within-stride surge grew
once only its average was scored. 0.15 m/s stride −4 %. Standing contact 0.56 → 0.63 but jitter up (Σ|q̇| 5.6 rad/s).
Falls 0 %. The averaged-velocity term was deleted (§2.1.9).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.8.

### 2.1.9 The real problem: the policy barely responds to slow commands

**Observed.** The run 3 policy barely responds to slow commands: shown 0.03 … 0.10 m/s it walks 0.106 … 0.128 m/s;
gate commands 0.05 / 0.10 / 0.15 → 0.110 / 0.128 / 0.146 m/s, a slope of ≈ 0.36 instead of 1. §2.1.7's "reward optimum
at 0.11" was partly this — the sweep could not drive the policy below 0.106.

**Ruled out: V_0.** `eval_s0.py --gt_velocity` gives the same speeds, and V_0's mean bias is ≤ 0.003 m/s.

**Root cause (working hypothesis).** The zero-init twist_cmd columns must be learned against command-independent terms
several times larger per step (alive 10, `cpg_z_target` ≤ 5, `cpg_coxa_contact` ≈ 5, vs tracking ≤ 10). The gain from
slowing to a 0.05 m/s command (≈ 0.15/step) is a small share of the advantage variance PPO sees; below the 0.16 m/s
clock knee only the command inputs can set stride.

**Fix (run 4).** Per-step tracking back (averaged form deleted), weight ×3 (exp σ 0.05 and L1, ×30) so tracking
dominates; branch from run 2 (the better tracker); §2.1.7's standing terms kept.

**Result.** Run 4 (`…/2026-09-22_00-18-12_s0b_seed42_r4_from_r2/model_6997.pt`): commands 0.05 / 0.10 / 0.15 → 0.062 /
0.104 / 0.148 m/s. **All four walking gates pass on L0 and L5** (stride +4–5 % at 0.10, 0 % at 0.15; cycle RMSE
≤ 0.009); the per-step error is also < 0.03, so D16's cycle averaging no longer decides the gate. Standing still
unclean (duty 0.40–0.78, Σ|q̇| 4.6 rad/s vs ≈ 0.5 at rest, drift 4.6 cm) → §2.1.10.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.9.

### 2.1.10 Standing: the actor kept obeying the frozen clock

**Observed.** Run 4 at zero command: per-leg contact duty 0.40–0.78, Σ|q̇| 4.6 rad/s, 4.6 cm drift. The reward
signal is not weak: rest pose (zero action) +0.930/step vs the policy +0.742 (gap 0.19, ≈ 25 % of the reward;
`logs/s0b_stand_diag.txt`). Dropped reads are not the cause (no drops: Σ|q̇| 4.0, duty 0.63).
**Root cause.** In standing mode the clock stops, but the actor still saw its frozen (sin, cos) values. Measured
over 256 standing envs (`logs/s0b_stand_phase.txt`): legs whose frozen phase reads "swing" are in contact 39 % of
the time, legs reading "stance" 87 % (every leg individually shows the same split, e.g. right-back 14 % vs 82 %).
The walking policy's "this leg is in swing" association survives into standing.
**Fix.** The o_p clock block is zero while the clock is stopped (`cpg_active` false) — the spec's own "clock
absent" flag (§5: (sin, cos) never equals (0, 0)), used again for the S2 clock dropout. Part of D13.

### 2.1.11 S0b run 5 (continuation of run 4, zero clock when standing)

Run 5 (from run 4). At 8,000: standing contact duty 0.40–0.78 → **0.85–0.97 on five legs**, Σ|q̇| 3.2 rad/s, drift
4.1 cm; walking unchanged (0.102 m/s, stride +3 %). The right-back leg stayed up (contact 11 %) → §2.1.12. Stopped at
8,142 (`…/2026-09-22_01-30-32_s0b_seed42_r5_from_r4/model_8100.pt`).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.11.

### 2.1.12 Right-back coxa parked beyond its servo limit — no gradient returns it

**Observed** (`logs/s0b_stand_load*.txt`, 128 standing envs, normal force per leg / fair share W/6). Rest pose,
zero action: 1.07 0.98 0.95 1.03 1.02 0.94 — the robot can stand evenly. Policy (run 5, iter 8000): 0.87 1.25
**0.12** 1.09 1.00 1.66, with the right-back coxa at **−54.9° from rest**, i.e. at the −55° action clip (servo
CW/CCW limit, `S0aActionsCfg`).
**Root cause.** Once the requested target is past the clip, the reward no longer depends on that action
component, so PPO gets no signal to bring it back — a dead zone. Right-back's weak stance in walking since S0a
(contact duty ≈ 0.40 vs ≈ 0.50 for the other legs) is plausibly the same stop.
**Fix.** `action_beyond_clip` = Σ |requested target − clipped target| (rad), weight −1, always on (zero while all
targets are inside the limits, so it does not change the economics of any in-range gait).

### 2.1.13 S0b run 6 (continuation of run 5 at 8,100, action-clip penalty)

Run 6 (from run 5 + `action_beyond_clip` −1). At 9,100: walking fine (0.0995 m/s, stride +1 %); standing RB contact
0.04; clip excess only 2.4 → 2.1 per second. The requested RB coxa target was −69° (14° past the stop) with femur
−21.5° and tibia +9°: the leg is deliberately **tucked**, not merely stuck past the clip. The coxa sign table was
re-checked against the layout probe (not a sign bug). Stopped at 9,201 (`…/2026-09-22_02-07-00_s0b_seed42_r6_from_r5/
model_9200.pt`); the clip penalty is kept.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.13.

### 2.1.14 Standing pose term; S0b run 7

The tucked pose is a local optimum: rest-pose standing pays ≈ 0.19/step more, but no small change of the tucked leg
improves the load-share / contact terms until the foot reaches the ground. Added a standing-only `standing_pose` =
Σ|q − q_rest| (rad), weight −1, a dense per-joint pull home.

Run 7 (from run 6): at 10,200 RB standing contact 0.01 — unchanged; the pose penalty stayed at ≈ −3.6…−4.0 per standing
step and the clip excess at ≈ 1.8–1.9 per second, no trend. The right-back habit is inherited from π_base's right-side
crutch legs (§2.0.14) and survives every warm start. Stopped at 10,257 (`…/2026-09-22_02-41-40_s0b_seed42_r7_from_r6/
model_10250.pt`).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.14.

### 2.1.15 S0b run 8: stronger pulls (clip ×10, standing pose ×3)

`action_beyond_clip` −1 → −10 (still zero for every in-range target), `standing_pose` −1 → −3. Run 8 (from run 7): clip
excess 1.8 → ≈ 0.28 per second, walking fine (stride +4 %), standing RB contact still 0.02. Stopped at 11,400
(`…/2026-09-22_03-15-45_s0b_seed42_r8_from_r7/model_11400.pt`).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.15.

### 2.1.16 Standing is a barely-trained input region, not a one-leg problem

`reward_economy.py --cmd 0 0 0 --pin_joints 'right_back_.*'` (`logs/s0b_stand_pin.txt`): policy +0.109/step, rest
pose +0.904, policy with right-back pinned at rest +0.147 — pinning one leg does not help, because the whole
stance is off. Posture probe (`logs/s0b_stand_posture.txt`, 128 envs, run 8 at 11,250), joint offsets from rest (°):

| | RF | RM | RB | LF | LM | LB |
| --- | --- | --- | --- | --- | --- | --- |
| coxa | +54 | −53 | −46 | +54 | +49 | +52 |
| femur | +57 | +45 | −13 | +12 | +36 | +7 |
| tibia | −67 | −75 | +6 | −11 | −19 | +2 |

(rest pose, zero action: all within 1.7°; base height 0.169 m vs 0.180 m under the policy). Almost every coxa sits
near its ±55° stop. The actor is training normally (not frozen; explained variance 0.96–0.99 in runs 5–8), so this
is data, not plumbing: standing is 10 % of segments, and with twist_cmd = 0 and the clock zeroed the input lies in a
region the network barely saw — its output there is extrapolation from walking, and 4,000 iterations of 10 %
standing data have not reshaped it.

### 2.1.17 S0b run 9: standing share 50 % for 1,000 iterations

Temporary schedule change (D18): zero-command segments 10 % → 50 % (`env.commands.base_velocity.rel_standing_envs=0.5`),
to return to the spec's 10 % before π_S0 is declared. Run 9 (1,000 iterations from run 8,
`…/2026-09-22_03-54-55_s0b_seed42_r9_from_r8_stand50/model_12399.pt`): standing unchanged (RB contact 0.05, same
contorted pose), walking fine. Five times the standing data did not move it → §2.1.18.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.17.

### 2.1.18 Root cause of the standing failures: the pose can only change by stepping

With six feet planted the legs form closed loops through the ground, so the joint angles are fixed by where the
feet are; returning to the rest pose needs a leg lifted and re-placed. Every hold term added in §2.1.5–2.1.16
(contact ×5, load share, joint speed, stance floor) charges exactly that. Wherever the robot stops mid-stride,
re-posing costs more in the short run than holding a contorted stance, so the learned standing pose is a
compromise of those stop points — which is why it contorts itself even from a rest-pose start. It also explains
why each stronger hold penalty failed: it made the remedy (stepping) dearer too.

**Fix (settle window, agreed with the user).** For the first `stand_settle_s` = 1.5 s of a stand (one gait period
at 0.8 Hz, rounded up) only `standing_pose` (the rest-pose pull) acts; `standing_contact`, `standing_load_share`
and `standing_joint_speed` apply from then on (`env.stand_hold`). The robot may take a step or two to square up,
then hold. The actor does not observe the timer; it can tell "not yet at rest" from its own joint positions.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 1500 \
  --resume --load_run 2026-09-22_03-54-55_s0b_seed42_r9_from_r8_stand50 --checkpoint model_12399.pt \
  --run_name s0b_seed42_r10_from_r9_settle env.commands.base_velocity.rel_standing_envs=0.5
```

### 2.1.19 Terrain difficulty — discussion with the user (before implementing)

Robot numbers: body 25 cm long, 13–20 cm wide; coxa 5.3 cm, femur 6.2 cm, tibia ≈ 13 cm (URDF). Belly clearance
≈ 13–14 cm (base origin ≈ 0.17 m above terrain, base plate 3.3 cm below it) — not binding. The binding limit is
**swing height ≈ 3.8 cm** (`cpg_z_target` step height; measured apex 3.7–4.0 cm). Slopes: servo torque is not binding
(femur ≈ 12 % of stall standing); friction is — at the lowest sampled μ 0.3 anything above ≈ 17° slides.

Proposal: gates also at L9, then levels 10–14 (rough ±2.25 cm, slopes 15°, bumps 3 cm) plus pyramid stairs with
1.5 → 3 cm risers, capped there — harder geometry needs a higher step target and makes blind arms fail for geometric
reasons, blurring the friction/compliance comparison (v7 §5.4: geometry ⟂ physics).

L9 check (run 4 policy, 256 episodes): all walking gates pass on the hardest built level (0.10: stride +6 %; 0.15: −1 %;
0 falls) — headroom for levels 10–14.

**Run 10** (settle window of §2.1.18; `…/2026-09-22_12-11-04_s0b_seed42_r10_from_r9_settle/model_13898.pt`) — the first
standing fix that worked: RB standing contact 0.05 → 0.57, all-leg duty 0.79–0.85, drift 4.0 → 1.8–2.0 cm, Σ|q̇| 2.3–2.6
rad/s, 0 % falls. Gates now on L0 / L5 / L9 (`logs/eval_s0b_r10_final.txt`): **all walking gates pass** (0.10 m/s stride
+0.3 / +1.6 / +2.2 %, 0.15 m/s −1.5 / −1.6 / −2.2 %, cycle RMSE ≤ 0.011, drops = drop-free) — the best walking of the
warm-start lineage. Standing posture still off rest (LB coxa at its +55° stop, per-leg load 0.45–1.82 × fair share,
Σ|q − q_rest| ≈ 5 rad vs ≈ 11 in run 8).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.19.

### 2.1.20 Decisions after run 10 (with the user, 2026-09-22)

Standing accepted as is (run 10). Next, before declaring π_S0: (1) an evaluation-only terrain stress sweep to find
where the gates break; (2) turning; (3) training on harder terrain up to the break point + 1–2 levels (capped at
level 14), the levels above it kept as a held-out suite; (4) that suite recorded in both specs as a secondary,
sim-only result marked TBD. Reasoning (agreed): training on levels beyond the robot's reach (obstacles ≥ 3 cm vs
the 3.8 cm CPG swing) wastes samples on failures and pays for lifting feet above the CPG target, distorting the
gait handed to S1/S2; the training terrain becomes an all-arm invariant at S1; levels above the training top give a
clean held-out test where F vs B2 is the fair comparison (depth arms beating blind arms on geometry is expected and
not a claim).

**Terrain sweep (eval only).** `mdp/terrain.py`: `EVAL_TERRAIN_TYPES` = the training sub-terrains + pyramid stairs
up/down (riser 0 → 3 cm at difficulty 1, tread 0.3 m; not in the training mix); `single_type_generator(name)` builds
one type over 15 levels with difficulty (ℓ + U)/15 exactly as in training, so levels 0–9 match the training rows.
`eval_s0.py --terrain_type <name>`; `scripts/rsl_rl/eval_terrain_sweep.sh <ckpt> <tag>`: 6 types × levels
{0, 3, 6, 9, 10–14} × v_x {0.10, 0.15}, 128 episodes per cell. (Naming fix: Isaac Lab's plain pyramid has the
platform on top — the robot walks *down*; the training keys `slope_up`/`slope_down` were swapped and are now
correct, order and terrain unchanged.) First attempt discarded: cells launched while the env code was being edited
crashed at import, and Kit exited 0, so `set -e` did not stop the script — cells are now judged by their JSON
(`MISSING …` otherwise).

### 2.1.21 Turning (design)

**Problem.** `cpg_coxa_contact` paid every leg for a *backward* coxa sweep. Turning needs the two sides to sweep in
opposite directions (and sideways motion needs sweeps along other directions), so the reward fought every turn —
the main reason a CPG-shaped policy resists anything but straight walking. v7 §2.6 already asks for the signed
velocity "projected on commanded heading".
**Fix.** `mdp.rewards.coxa_sweep_sign(env, twist)`: per leg, a planted toe must move at v_toe = −(v + ω ẑ × p_toe)
(body frame); +coxa moves it along t⁺ = ẑ × (p_toe − p_coxa); the leg is paid for coxa rate × sign(cos(t⁺, v_toe)),
0 where |cos| < 0.2. For straight walking it equals the old `COXA_BACKWARD_SIGN` (asserted at env build). Only S0b
uses it (`env.coxa_sign()`); Flat/S0a keep the fixed sign, so their rewards are unchanged.
**Commands (D19).** Segments: 10 % standing (independent); of the rest 15 % turn on the spot (v_x = v_y = 0,
|yaw| ~ U[0.1, 0.4] rad/s, random sign — clock at the 0.8 Hz floor, not standing), 30 % walk with v_y ±0.05 m/s
and yaw ±0.4 rad/s, 55 % straight. Lateral goals are meant to be reached by turning then walking; the spec's small
v_y stays in the mix. The evaluation conditions of the arms (v_x 0.10/0.15, v_y = yaw = 0) are unaffected.
**Yaw tracking** on the same footing as linear (§2.1.9 lesson): `track_ang_vel_z` = exp(−e²/0.1²) − |e|, ×10
(was Isaac Lab's exp σ 0.25, ×3). Gate script adds turn on the spot ±0.3 rad/s and walk 0.10 m/s + 0.2 rad/s on
L0/L5; provisional turning gates (not in the spec): yaw-rate cycle RMSE ≤ 0.05 rad/s, falls ≤ 5 %.
**Economics** (`reward_economy.py --cmd <vx vy wz> --terrain_level 0`, run 10 policy — never trained to turn —
vs zero-action standing, dt-scaled per step; `logs/s0b_economy_turn.txt`): turn on the spot 0.3 rad/s: policy +0.280
(drifts 0.032 m/s, does not turn), stand +0.696, perfect turning ≈ +1.05 (stand's terms + yaw tracking +0.26 + CPG
terms, now aligned); walk 0.10 + yaw 0.2: policy +0.687 (walks, yaw error ≈ full), stand +0.076, perfect ≈ +0.91;
straight 0.10: walk +0.751, stand +0.309. Turning > standing > falling in every command class.

### 2.1.22 Terrain sweep result — no break point within the agreed cap

`scripts/rsl_rl/eval_terrain_sweep.sh <run 10 model_13898.pt> r10` — 6 types × levels {0, 3, 6, 9, 10–14} × v_x {0.10,
0.15}, 128 episodes per cell, 108/108 cells (`logs/eval/sweep_r10_*.json`).

**0 % falls in all 108 cells.** The only gate failures are uphill at 0.15 m/s, where the robot slows: slope-up L13–14
(stride −8 … −10 %, cycle RMSE 0.032–0.038) and stairs-up L14 (−11 %). Within the agreed cap (≤ 3 cm obstacles, ≤ 15°)
the policy has no break point despite training on levels 0–9 only, so step 3 of the plan (train to break point + 1–2,
hold out the rest) was re-decided with the user (§2.1.24).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.22.

### 2.1.23 S0b run 11: turning (continuation of run 10)

Run 11: turning (§2.1.21) from run 10, standing share back at the spec's 10 %; `eval_s0.py` now reports `wz_mean` and
`wz_rmse_cycle` for yaw commands (stride only when a translation is commanded). Results in §2.1.24.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.23.

### 2.1.24 Held-out levels 15–20 (decision: train 0–14, hold out 15–20)

Decided with the user: train on levels 0–14 (the agreed cap); keep 15–20 — beyond the cap, eval-only — as the held-out
suite for the secondary result. `single_type_generator` now builds 21 levels with difficulty (ℓ + U)/15 (levels 0–14
unchanged; level 20: rough ±3.0 cm, slopes ≈ 19.7°, bumps and stairs 4.0 cm, at/over the 3.8 cm foot lift).

Held-out sweep of run 10 (`eval_terrain_sweep.sh <run 10> r10x 15 … 20`, 72/72 cells, `logs/eval/sweep_r10x_*.json`):
falls stay ≈ 0 (one cell 1 %) even at 4 cm stairs and ≈ 20° slopes. Beyond the cap the policy degrades in **speed and
stride accuracy** (uphill and rough first; L20 stairs-up at 0.15 m/s −30 %), not in survival → the held-out suite must be
scored on tracking error and stride, not only falls.

Run 11 result (`…/2026-09-22_16-11-22_s0b_seed42_r11_from_r10_turning/model_15897.pt`, 256 episodes): 0 % falls; straight
walking passes every gate (0.10 stride +6–8 %, 0.15 ≈ +1 %); walk 0.10 + yaw 0.2 → ω̄_z 0.153, yaw RMSE 0.049–0.054.
**Turning is lopsided:** turn on the spot −0.3 → −0.228 (≈ 75 %), +0.3 → +0.081 (27 %). The coxa sign is mirror-
symmetric by construction, so the asymmetry is the policy's. Standing contact slipped 0.85 → 0.70 with the standing
share back at 10 %.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.24.

### 2.1.25 S0b run 12: training terrain = levels 0–14 + stairs; turning continues

`S0B_TERRAIN_CFG`: 15 rows, difficulty (0, 1); mix flat 0.15, rough 0.20, slope down 0.15, slope up 0.15, bumps 0.15,
stairs down 0.10, stairs up 0.10 (stairs: 0 → 3 cm risers, tread 0.3 m). Levels 0–9 keep their per-level parameters,
but the mix changed (stairs added), so their training distribution is not identical to runs ≤ 11.

Run 12 (from run 11, 3,000 iterations planned). Its first launch crashed in Kit start-up (the OpenBLAS at-fork race
found in §2.2.5); the relaunch worked. At 17,400: curriculum mean level 9.3 of 14, 0 falls; turn on the spot +0.3 →
+0.108 rad/s, −0.3 → −0.272: left turns still barely work → §2.1.26.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.25.

### 2.1.26 Lopsided turning: the coxa stroke term was farmable by tucked toes

**Observed** (`logs/s0b_turn_probe2.txt`, run 12 at 17,400, turn on the spot). The needed coxa sign
(`coxa_sweep_sign`) during stance: left turn — left-middle and left-back get the *opposite* sign to the other legs
99 % / 94 % of stance steps, and the policy sweeps them that way at 2.0–2.2 rad/s; yaw rate only +0.05 of +0.3.
Right turn — only right-front flipped (54 % of steps); yaw −0.26 of −0.3.
**Root cause.** The sign is kinematically right: the needed coxa rate is θ̇ = (r × v_toe)_z / |r|² = −ω (r·p_toe)/|r|²
for a pure turn (r = toe − coxa, p_toe = toe − base), so it flips when a toe sits inside its coxa relative to the
base. But in this tall stance the horizontal toe reach |r_xy| is only 5.5–8 cm (`logs/s0b_sign_debug.txt`), and
the term paid coxa *angular* rate: with a toe tucked near/under its coxa, a large coxa spin barely moves the toe but
earns full reward. The policy tucked the left-middle/left-back toes and spun them — a reward exploit that fights
left turns. (Geometry at rest was checked: all six signs −1 for a left turn.)
**Fix.** `coxa_stroke_weight` = sign × |r_xy| / (that leg's rest-pose reach, measured at env build): the term pays
toe stroke speed, not coxa spin; tucked toes earn ≈ 0. Flat/S0a keep the fixed sign. (§2.1.35 later found the rest
reach was mis-measured, so from run 13 on the term ran at ≈ 0.32 × its intended strength.)

**Result.** Run 13 (from run 12 at 17,600; `…/2026-09-22_18-39-41_s0b_seed42_r13_from_r12_coxareach/model_20099.pt`):
the spin exploit is gone (tucked legs stroke only 0.1–0.3 rad/s), but the **tucked posture remained** — RF, LF, LM, LB
toes at 3–10 % of their rest reach; only RM and RB reach out. Final gates (256 episodes, training terrain now levels
0–14): 0 falls; straight gates pass on L0–L9 (stride −1.4 … −3 %, cycle RMSE ≤ 0.008); walk-and-turn passes the
provisional yaw gate (RMSE 0.044–0.048); turn right on the spot −0.294 ✓; **turn left +0.114 (38 %) ✗** — unchanged
across runs 11–13. Discussed with the user → §2.1.27.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.26.

### 2.1.27 Left–right mirror augmentation (user decision, D21)

**Why.** The policy has been left/right lopsided since π_base (right-side crutch legs, 2.0.14; right-back tucked
when standing, 2.1.12; left turns on the spot at 38 %, §2.1.26) while every reward is mirror-symmetric. Each reward
fix removed a cause but not the inherited habit. Decision with the user: train with mirror-symmetric data
augmentation (Mittal et al. 2024, rsl_rl's `symmetry_cfg`), then move on to the from-scratch control run. This is a
training-algorithm change shared by every arm; it must stay on from S1 onward (paradigm spec note pending).

**Mapping** (`mdp/symmetry.py`): reflection y → −y; legs RF↔LF, RM↔LM, RB↔LB; coxa angle, rate, tracking error, last
action and action flip sign (+coxa moves right toes forward, left toes backward), femur/tibia keep it; polar vectors
(IMU accel, gravity, V_0, true velocity, CoM offset) negate y; angular velocity (ω_x, ω_y, ω_z) → (−ω_x, ω_y, −ω_z);
twist_cmd (v_x, v_y, ω_z) → (v_x, −v_y, −ω_z); per-leg clock, read mask, foot forces and contacts are permuted with
the legs (the tripods swap = the clock shifted by π, a state the phase jitter already produces); height, foot
material, mass and servo scalars unchanged. `rsl_rl` trains every mini-batch on original + mirrored samples.
**Tests.** `python …/mdp/symmetry.py` — the mirror is an involution on all three layouts. `scripts/tests/mirror_test.py`
(gravity off, floating, 16 env pairs holding q and mirror(q), servo randomisation off): toe positions mirror to a
median 0.91 mm, max 3.6 mm; joint-observation block to ≤ 0.010 rad → **PASS** (`logs/mirror_test.txt`). With servo
randomisation on the error was 7 mm / 0.02 rad (backlash, gains); a wrong joint sign would show as cm / 0.3 rad.

### 2.1.28 S0b run 14: mirror augmentation (continuation of run 13)

Run 14 (2,000 iterations from run 13 with mirror augmentation, 2.1 s/iteration;
`…/2026-09-22_20-30-48_s0b_seed42_r14_from_r13_mirror/model_22098.pt`). rsl_rl's symmetry loss (logged only) 0.87 → 0.75.

Against run 13 (256 episodes, `logs/eval_s0b_r14_final.txt`): straight stride −0.1 … −3.5 %, cycle RMSE ≤ 0.008;
walk-and-turn yaw RMSE 0.044 → **0.030**; turn left on the spot 38 % → **52 %** of the command (+0.155), right −0.278;
standing contact 0.68 → **0.80**. 0 falls; drops = drop-free. Mirror augmentation helped every asymmetric measure without
costing straight walking but did not close the left/right gap within the agreed 1–2 h → from-scratch control arm next,
which also tests whether the asymmetry is inherited from the π_base lineage.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.28.

### 2.1.29 π_S0 (provisional) and the from-scratch control arm

**π_S0 = `logs/rsl_rl/hexapod_s0b/2026-09-22_20-30-48_s0b_seed42_r14_from_r13_mirror/model_22098.pt`** (seed 42,
4,096 envs × 16 steps, 22,098 S0b iterations on top of π_S0a's 4,498; declared with the user, provisional).
Spec §4.2 S0b gates, 256 deterministic episodes per condition, terrain levels 0/5/9 (`logs/eval_s0b_r14_final.txt`):
v_x tracking RMSE ≤ 0.008 (limit 0.03) ✓; stride within −0.1 … −3.5 % (limit ±10 %) ✓; falls 0 % (limit 5 %) ✓;
tracking at the provisional drop rate identical to drop-free ✓. Our own (not in the spec) turning gates: walk-and-turn
✓, turn on the spot right ✓, **turn on the spot left ✗ (52 % of a 0.3 rad/s command)** — the open caveat, carried
into the paper's honesty section unless the from-scratch arm or later stages close it. Standing: 80 % contact duty,
1.7 cm drift per 10 s, no falls.

**From-scratch control arm (spec §4.2), decisions with the user:** budget = the warm-start reconditioning effort,
S0a + S0b = 4,498 + 22,098 = 26,596 iterations ≈ 1.74 B env steps (Phase 1's π_base is treated as sunk cost);
the control arm gets 1.2× ≈ 32,000 iterations. Setup: the **final** S0b task from a random actor (init std 1.0,
floored at 0.05), same env, rewards, DR, terrain curriculum, mirror augmentation and batch as π_S0's last run, with
the S0PPO actor freeze disabled (it exists to protect a warm-started gait). Gates evaluated at 4k, 8k, 16k, 24k and
32k iterations; if it passes the S0b gates within the budget, the warm start and the KL anchor (§4.4) are dropped
for every arm.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 32000 \
  --run_name s0_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
```

**Deviations added in S0b stage 2** (spec text updated where noted):

| # | Deviation | Where |
| --- | --- | --- |
| D18 | Standing share raised to 50 % for runs 9–10 (training schedule), back to the spec's 10 % from run 11 | 2.1.17 |
| D19 | Commands: 15 % of non-standing segments turn on the spot (\|yaw\| 0.1–0.4 rad/s), yaw range ±0.4 (was ±0.25); `cpg_coxa_contact` pays the stroke the commanded twist needs (v7 §2.6's "projected on commanded heading"), reach-weighted | 2.1.21, 2.1.26 |
| D20 | Training terrain = levels 0–14 of the 15-level generator + stairs (0 → 3 cm risers); levels 15–20 eval-only | 2.1.24, 2.1.25; paradigm spec §1 terrain row and §10.4 updated |
| D21 | Left–right mirror data augmentation in PPO, every arm from S0b | 2.1.27; paradigm spec §1 training-algorithm row |
| D22 | Held-out harder-terrain suite recorded as a secondary result marked TBD | paradigm spec §10.4, v7 §5.3 |

### 2.1.30 From-scratch control arm — progress

Gates of `logs/rsl_rl/hexapod_s0b/2026-09-22_22-50-43_s0_fromscratch_seed42/` checkpoints (256 episodes per condition,
`logs/eval_fs_it*.txt`):

| Iteration (of 32,000) | falls | 0.10: v_x / stride err / RMSE | 0.15: v_x / stride err / RMSE | turn ±0.3 | standing duty / drift |
| --- | --- | --- | --- | --- | --- |
| 4,000 | 0 % (L0/L5/L9) | 0.095 / **−36 %** / 0.006 | 0.144 / **−29 %** / 0.007 | **+0.064 / −0.063** | 0.90 / 1.5 cm |
| 8,000 | 0 % (L0/L5/L9) | 0.094 / −4.9 % / 0.006 | 0.140 / −6.8 % / 0.009 | +0.187 / −0.191 | 0.97 / 1.0 cm |
| 18,750 | 0 % (L0/L5/L9) | 0.097 / −1.8 % / 0.004 | 0.146 / −1.8 % / 0.003 | +0.257 / −0.254 (RMSE 0.040 / 0.042) | 0.95 / 0.8 cm |

* At 4,000 it already walks at the commanded speed with 0 falls and stands better than π_S0, but takes shorter, faster
  steps than the 0.8 Hz clock (stride −29 … −36 %) and barely turns — **symmetrically** (+0.064 / −0.063), evidence
  that the left/right asymmetry of §2.1.26 is inherited from the π_base lineage.
* **At 8,000 iterations — 0.30 × the warm-start budget — it passes all four spec §4.2 S0b gates** (falls 0 %, cycle
  RMSE ≤ 0.011, stride −4.6 … −8.3 %, drops = drop-free), well inside the 1.2 × rule, and beats π_S0 on standing (0.97
  vs 0.80), walk-and-turn (yaw RMSE 0.024 vs 0.030) and symmetric turning. Per §4.2 **the warm start and the KL anchor
  (§4.4) are dropped for every arm**; training continued to the full budget for quality.
* At 18,750 (0.70 ×) it passes every gate including both provisional turning gates and beats π_S0 on all of them.
* **Stopped at 20,749** (user: return plateaued at ≈ 400 for ~2,000 iterations) → `model_20700.pt` = 0.78 × the
  warm-start budget. Its 0.10 m/s stride error (−1.8 %) exceeds the warm-started one (−0.1 %) only because stride =
  v_x / f_actual measures speed accuracy (0.0966 m/s at 0.787 Hz vs 0.0984 at 0.800 Hz); ≈ 2 mm per step, far inside
  ±10 %.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.30.

### 2.1.31 π_S0 — final (from scratch, 2026-09-23)

**π_S0 = `logs/rsl_rl/hexapod_s0b/2026-09-22_22-50-43_s0_fromscratch_seed42/model_20700.pt`** (seed 42, 4,096 envs ×
16 steps, 20,749 iterations ≈ 1.36 B env steps, no warm start, mirror augmentation on). Reproduce:

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 32000 \
  --run_name s0_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
# stopped at 20,749 (return plateaued ≈ 400 for ~2,000 iterations)
scripts/rsl_rl/eval_s0b_gates.sh <ckpt> pi_s0_final 1024
```

Final gates, **1,024 episodes per condition**, 17 conditions = 17,408 episodes, `logs/eval_pi_s0_final.txt`:

| Condition | L0 | L5 | L9 |
| --- | --- | --- | --- |
| 0.10 m/s: v_x / stride (ref 0.125) / cycle RMSE | 0.1024 / 0.1299 (+3.9 %) / 0.0041 | 0.1022 / 0.1303 (+4.2 %) / 0.0043 | 0.1021 / 0.1307 (+4.6 %) / 0.0050 |
| 0.15 m/s: v_x / stride (ref 0.1875) / cycle RMSE | 0.1508 / 0.1902 (+1.4 %) / 0.0031 | 0.1502 / 0.1898 (+1.2 %) / 0.0035 | 0.1495 / 0.1888 (+0.7 %) / 0.0047 |
| turn +0.3 / −0.3: ω̄_z, yaw RMSE | +0.258, 0.038 / −0.264, 0.032 | +0.258, 0.039 / −0.264, 0.032 | — |
| walk 0.10 + yaw 0.2: ω̄_z, yaw RMSE | 0.202, 0.011 | 0.201, 0.012 | — |
| standing: contact duty / drift | 0.95 / 0.85 cm | 0.87 / 1.03 cm | 0.84 / 1.24 cm |
| falls | 0 % | 0 % | 0 % |
| 0.10 / 0.15 with vs without dropped reads | 0.0041 vs 0.0043 / 0.0031 vs 0.0032 | — | — |

**All spec §4.2 S0b gates pass**, with zero falls in 17,408 episodes (95 % upper bound 0.29 % per condition), and both
of our own turning gates pass symmetrically. Toe clearance 3.9–4.2 cm walking. The spec §4.2 control-arm rule fired:
the warm start and the KL anchor are dropped for every arm (paradigm spec §1, §4.2, §4.4 updated 2026-09-23).

### 2.1.32 Demonstration script (keyboard teleoperation + video)

`scripts/rsl_rl/demo_teleop.py` (needs the Isaac Sim GUI for key input): one difficulty (`--level`, 0–20) with the
seven terrain types as columns — flat, rough, slope down, slope up, bumps, stairs down, stairs up — three rows deep,
so walking sideways crosses types. Arrow keys drive (v_x 0–0.20, v_y ±0.05), Z/X turn (±0.4 rad/s), no key is the
standing mode, R starts/stops a video take (`logs/videos/demo/*.mp4`, Isaac Lab's RecordVideo), N resets with a fresh
dynamics-randomisation draw. Commands are clamped to the trained ranges; DR, sensor noise and dropped reads stay on;
the camera follows the robot. `--selftest N` runs N scripted steps headless (no keyboard) and writes one take —
used to verify the script: level 12, 300 steps, base 0.169 m above terrain, `s0_demo_L12_take1.mp4` written.

### 2.1.33 π_S0 terrain characterisation (180 cells)

`scripts/rsl_rl/eval_terrain_sweep.sh <π_S0> pis0 0 3 6 9 10 … 20` (180 cells, 128 episodes each,
`logs/eval/sweep_pis0_*.json`). **Every cell of the training range (levels 0–14) passes, with zero falls** — the
warm-started policy already failed uphill at L13 (§2.1.22). Falls stay at 0 % through the held-out levels too;
degradation there is in speed and stride: first failures at L17 on rough and stairs-up at 0.15 m/s (−12 … −14 %),
large by L20 (stairs-up 0.15 −38 %, rough 0.15 −24 %, stairs-up 0.10 −26 %); 52 / 72 held-out cells pass. This is the
headroom §10.4 needs and confirms the suite must be scored on tracking and stride, not falls.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.33.

### 2.1.34 Gait geometry: the feet migrated under the hips (open question)

The user saw in the teleop demo and `play.py` that the tibia points inwards. `scripts/rsl_rl/record_gait.py` (new) walks
a checkpoint at 0.15 m/s on flat ground, records a video (`logs/videos/gait/*.mp4`) and prints per-leg joint angles, toe
position, **reach** (toe distance from its coxa joint along the hip's outward direction; rest ≈ 0.06 m) and
**tibia_lat** (+ = shin outboard) (`logs/gait_geometry.txt`).

**Finding.** The middle-leg reach shrank along the lineage — π_base 0.184 m, π_S0a 0.102, S0b runs 0.097–0.136 — to
**0.011–0.018 m in π_S0 (from scratch)**, with tibia +34…+39°, tibia_lat −0.047…−0.062 and stance half-width 0.108 m
(π_S0a 0.161). The four middle/back feet stand almost under their hips, knees splayed, shins slanting inward; front legs
near normal (reach 0.08–0.10); left/right mirror to ≈ 2 mm.
**Why:** static joint torque scales with the horizontal joint–foot distance, so tucking minimises torque excess, servo
sag and stance-rigidity cost; nothing in the reward asks for stance width, so the cheapest posture wins (and a narrow
stance helps turning on the spot). S0a inherited π_base's wide stance; the from-scratch run optimised straight to the
torque optimum.
**Risk:** all gates pass, but the support polygon is ~25 % narrower than the rest-pose footprint and the ground contact
sits near the edge of the foot rather than the rubber tip — a sim2real contact-geometry risk (v7 §5.1). S1–S3 keep this
reward minus the CPG terms, so it would not resolve itself.

**Swing-phase check** (user: "unnecessary tibia motion in the air"; per-swing excursion, path/excursion and direction
reversals, `logs/swing_probe.txt`). π_S0's tibia is not jittery — the smoothest of π_S0a / S0b r14 / π_S0 (path /
excursion 1.6–2.0; π_S0a up to 3.1 with 3–4 reversals). What looks unnecessary is **amplitude**: middle-leg coxa arcs of
95° (π_S0a ≈ 50°), close to the ±55° limits, because toe travel = coxa angle × reach and the middle toes sit ≈ 1 cm
outboard of their hips. The inward shins and the large arcs are one problem. Proposed: a stance-footprint term (§2.1.35).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.34.

### 2.1.35 Rest-reach bug (runs 13–14 and π_S0) and the stance-footprint term

**Observed.** Sizing the footprint term on π_S0 (`reward_economy.py --set rewards.stance_footprint.weight=-1`):
motionless rest-pose standing showed a *larger* footprint deviation (0.84 m summed) than the tucked walking policy
(0.47 m) — impossible if the target were the rest pose. `env.coxa_rest_reach` read **0.2079 m for every leg**;
standing at rest the actual reach is 0.054–0.093 m (`logs/reach_debug.txt`).
**Root cause.** `HexapodEnv.load_managers` measured it while the articulation was still in the USD's zero pose
(all joints 0 → legs straight out: 0.0529 + 0.06 + 0.095 = 0.2079 m); the configured rest pose is only applied at
the first reset. **Consequence for trained policies:** `coxa_stroke_weight` (§2.1.26) divided the toe reach by 0.208
instead of the rest reach, so from run 13 onward — including π_S0 — the coxa stroke term ran at ≈ 0.32 × its
intended strength (sign logic unaffected). Plausibly it contributed to the tucking: the coxa term is the one that pays
more for extended legs.
**Fix.** At build, write the rest pose to the joints, take one physics step so PhysX refreshes the link transforms,
then measure; assert reach < 0.15 m. Now **0.0669 m for every leg** (`logs/reach_debug2.txt`). The coxa term on π_S0
reads 0.077/step instead of 0.025 (×3.1 as expected); near the command the tracking slope (≈ 5 per m/s) still dwarfs
the coxa term's (≈ 1.2), so no speed bias.

**Stance-footprint term.** `stance_footprint_penalty` = Σ over feet in contact of | |r_xy| − rest reach | (m); the
horizontal distance from the coxa axis is invariant to the coxa sweep, so it restores extension without touching the
stride. Economics (π_S0, L0, per step dt-scaled at weight −1): walking 0.15 → 0.0015, turning → 0.0030, rest-pose
standing → 0.0007 (servo-sag floor); torque excess ≈ 0, so the tucking was not bought with a reward saving — it was
simply unopposed. **Weight −20** (tucked posture ≈ 3 % of the walking reward).

**Result** (2,000 iterations from π_S0, `…/2026-09-23_14-50-13_s0_fromscratch_seed42_footprint/model_22699.pt`). Gates
(1,024 episodes) all pass, 0 falls, slightly better than π_S0 (0.10 m/s stride +1.4 / +2.3 / +2.9 %; turn +0.278 /
−0.272; standing 0.97). Posture: middle reach 0.012 → 0.029 m, back 0.018 → 0.034 m (rest 0.067), half-width 0.108 →
0.122 m — but the middle/back tibias bent further (+34…+39° → **+43…+44°**), shins still slant inward, middle coxa arc
unchanged at 95°. The policy met the target partway and the cheap way. By eye "more or less the same" → no sweep.
Options put to the user: tibia-slant penalty, footprint weight ≈ −60, or accept.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.35.

### 2.1.36 Mid-air toe in/out motion: detection (no training yet)

**Observed Behavior.** By eye (user) the four middle/back legs of π_S0 and of the footprint continuation swing the toe
in and out mid-air ("slants more, then less just before landing"); the front pair and π_S0a look fine.

**Detection.** `scripts/tests/swing_radial_probe.py` (flat, 0.15 m/s, 128 envs × 300 steps after a 100-step settle): per
swing (≥ 5 airborne steps) the toe's horizontal distance from its coxa axis |r_xy|, tangential sweep, lift and tibia
angle, plus the mean profile over 10 normalised swing bins (`logs/swing_radial_probe.txt`).

**Findings.**
1. The motion is real: π_S0's middle/back legs lift by *folding the tibia* (+16°, toe pulled ≈ 5 cm toward the hip in
   the first 30 % of the swing), unfold it before landing, dip in again, and land ≈ 5 cm closer to the hip than they
   lifted off. π_S0a lifts with the femur: the toe arcs ≈ 7 cm *outward* and lands where it lifted off (tibia ±6°). The
   footprint term made it slightly worse (+20°).
2. **The amount of radial motion does not separate the policies** (radial path per swing π_S0a 12.6–19.0 cm vs π_S0
   ≈ 20 cm); the difference is direction and mechanism (inward via the tibia with a reversal, vs outward via the femur
   once) and the stride (middle/back sweep 4–6 cm vs 11–15 cm). A plain |Δ|r_xy|| swing penalty would charge π_S0a's
   liked gait about as much, so per the agreed gate it was **not** implemented; re-discussed with the user (§2.1.37).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.36.

### 2.1.37 Coordinated swing lift: swing radial motion + touchdown reach

**Root cause (geometry, not the policies).** A joint rotation moves the toe at right angles to the joint→toe line:
lift per radian = horizontal joint–toe distance, in/out per radian = vertical distance. At the S0a rest pose
(femur +30°, tibia +30°, reach 6.7 cm; coxa 5.3, femur 6.0, shin→toe 13.1 cm) the toe sits almost under the femur
joint and slightly inside the knee (a planar estimate): femur-up ≈ 0.25 mm lift and 1.7 mm **out** per degree;
tibia-fold ≈ 0.65 mm lift and 2.2 mm **in** per degree. A single-joint lift therefore throws the toe radially
(π_S0a out, π_S0 in); a vertical lift needs femur up and tibia fold together (≈ 0.75 : 1). This pose was chosen for
the torque gate (§2.0.3), so the trade-off is stance torque vs swing kinematics. Swing energy is negligible
(≈ 0.2 kg leg × 4 cm ≈ 0.08 J per swing, < 1 W total) against stance holding, so the case for the change is
reliability: a controllable landing point (needed by the foothold arms), support width and a narrow toe path.
User decision: the baseline uses a coordinated lift. This is a reward change inherited by every arm, so the final
π_S0 for the paper is retrained from scratch after a pilot.

**Fix.** `mdp.rewards.swing_radial_motion` (Σ airborne |Δ|r_xy||/dt, m/s; free: coxa sweep and vertical lift) and
`mdp.rewards.touchdown_reach` (Σ over legs landing this step of | |r_xy| − rest reach |, m). Both use contact > 1 N,
reward-only, × cpg_active. They replace `stance_footprint` (§2.1.35), which π_S0 met by bending the tibia.

**Economics** (`logs/s0b_economy_coordlift.txt`, π_S0, L0, dt-scaled per step at weight −1): walking 0.15 →
radial 0.0200, touchdown 0.00008; turning on the spot 0.3 → 0.0139 / 0.0001; standing ≈ 0.0002 / 0.
Weights: **radial −1** (≈ 2 % of the 0.97 walking reward), **touchdown −300** (≈ 0.024/step, ≈ 2.5 %; ≈ −0.22 per
landing at π_S0's 3.7 cm landing error). Lowering the lift to dodge the radial cost is unprofitable: at 1.5 cm the
radial saving (≤ 0.010) is below the `swing_clearance` (0.005) plus `cpg_z_target` (0.067 total, largely lift)
losses, while a coordinated lift saves the full 0.020 at no cost. Once the clock terms leave in S2, only
swing_clearance defends the lift; by then the radial cost of a coordinated lift is ≈ 0. Walk (0.97) > stand
(0.27 at 0.15 command) > fall (−0.1) unchanged.

**Pilot from π_S0** (radial −1 / touchdown −300, stopped by the user at 1,350 iterations on a plateau,
`…/2026-09-23_19-07-33_s0_pilot_coordlift/model_22050.pt`): gates pass (1,024 episodes, 0 falls) but **the gait did not
change** — middle/back radial path 17.7–20.8 cm per swing, landing 4.7 cm inside lift-off, tibia fold +19°. Both terms
stayed flat: the converged policy (std at the 0.05 floor) found no cheaper lift. The front legs dominate the radial
cost (23 cm per swing), which the user considers fine.

**5× check from π_S0** (radial −5, touchdown −1,500, 1,700 iterations, `…/2026-09-23_20-32-32_s0_pilot_coordlift_x5/
model_22399.pt`): the terms moved, the wrong way — middle radial path 19.9 → 15.3 cm by *shrinking the sweep* (5.3 →
3.9 cm) and *holding the tibia folded* (+9° at touchdown); landing still 4.5 cm inside; gates pass but 0.10 m/s
overshoots +3…+3.6 %. **Conclusion: fine-tuning a converged π_S0 does not reach a coordinated lift; the tucked-tibia
basin is deep.**

**From scratch with the new terms** (user: 18,000-iteration cap), weights fixed at 2× the pilot: **radial −2,
touchdown −600** (on π_S0's behaviour ≈ 9 % of the walking reward; ≈ 0 for a coordinated lift; walk 0.88 ≫ stand 0.27
≫ fall), all six legs, everything else as π_S0 (§2.1.31). At 1,750 iterations the return was ≈ 300 vs ≈ 238 for the
old run at the same point: ≈ 2/3 real tracking (track_lin_vel 16.8 → 19.9, track_yaw 1.6 → 3.2), ≈ 1/3 bookkeeping
(the rest-reach fix makes the coxa term ≈ 3× larger). Middle/back radial path 6.0–6.3 cm per swing (π_S0 final ≈ 20),
sweep 8–13 cm, tibia folds ≈ 44° while |r_xy| moves ≈ 1.5 cm: **a coordinated lift**. Standing not learned yet at that
point. Stopped at ≈ 2,700 and replaced by §2.1.38's run.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.37.

### 2.1.38 Leg-to-leg clearance

**Observed Behavior.** The coordinated-lift run (§2.1.37, stopped at ≈ 2,700 iterations,
`…/2026-09-23_22-28-40_s0_coordlift_fromscratch_seed42/model_2650.pt`) looks right, but by eye (user) the
front/middle and middle/back toes almost touch while walking — a hardware risk.

**Root cause.** Self-collision is off (`enabled_self_collisions=False`, `assets/robot.py`), so legs can pass through
each other in sim and nothing charges the policy for it. Measurement (`scripts/tests/leg_gap_probe.py`,
`logs/leg_gap_probe_capsule.txt`): links modelled as capsules fitted to the right-front USD meshes — tibia from the
knee-servo top (−0.01, 0, 0.035) to the toe, r 3.0 cm (99 % of mesh points; bounding 3.6 cm; the true half-width across
the shin is 2.3 cm, so the capsule is conservative sideways); femur (−0.01…0.075 m on x), r 4.0 cm. Surface gap =
closest segment distance − radii (Ericson segment–segment, checked against brute force on 20,000 random pairs). The
new run overlaps down to −3.8 cm at 0.15 m/s (17–19 % of steps under 2 cm), −2.7 at 0.20, −3.0 walking-and-turning;
turning on the spot ≥ +2.8 cm, standing +7 cm. π_S0 was worse (−6.3 cm, ≈ 22 % of steps). At closest approach the
middle coxa sits at ±55°, the action clip limit, while its neighbour is swung the opposite way (tripod anti-phase).

**Fix.** Coxa limits alone were rejected: collisions depend on a *pair* of angles, so a per-joint limit must assume
both legs at their extremes, which shortens the stride everywhere. Instead `mdp.rewards.leg_proximity_penalty` =
Σ over same-side neighbour pairs max(0, 2 cm − capsule gap) (L1, m, zero with room to spare; all four capsule
combinations, tibia/femur). Hard per-leg coxa CW/CCW limits on the real servos remain the deployment backstop, set just
outside the trained envelope. Self-collision stays off in training (the foot contact sensor would count leg–leg
contact as ground contact) and is a candidate eval-only check.

**Economics** (`logs/s0b_economy_proximity.txt`, model_2650, L0, dt-scaled at weight −1): walking 0.15 → 0.00036,
walk-and-turn 0.10 + 0.2 → 0.0001, turning on the spot and standing 0. **Weight −400**: 0.15 m/s overlaps cost
≈ 0.14/step (≈ 17 % of the 0.84 walking reward) — a safety term, heavier than the ≈ 9 % style terms and far below
tracking; walk 0.70 ≫ stand 0.27 ≫ fall. Clearance can be bought without losing stride by moving the middle legs'
sweep centre (standing clearance is +7 cm).

**Run** (from scratch, replaces the stopped §2.1.37 run; user: review every 2,000 iterations with video + probes, stop
early on a plateau, cap 18,000):
```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 18000 \
  --run_name s0_coordlift_prox_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
scripts/rsl_rl/milestone_review.sh <run_dir> <iter>   # every 2,000 iterations → logs/milestones/<run>/
```

Milestones (`logs/milestones/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/`; without the term the worst gap
was −3.8 cm):

| Iteration | return (gain / 2,000 it) | worst capsule gap | middle/back radial path per swing | lift | half-width |
| --- | --- | --- | --- | --- | --- |
| 2,000 | 291.6 | +0.5 cm | 7.3–7.9 cm | 3.4–4.1 cm | 0.12 m |
| 4,000 | 337.3 (+15.7 %) | +0.6 cm | 7.8–8.1 cm | 3.6–4.0 cm | 0.12–0.13 m |
| 6,000 | 365.3 (+8.3 %) | +0.2 cm | 7.8–8.0 cm | 3.9–4.0 cm | 0.13 m |
| 8,000 | 382.9 (+4.8 %) | +0.7 cm | 6.8–8.4 cm | 4.1–4.4 cm | 0.14 m |
| 10,000 | 389.5 (+1.7 %) | +0.9 cm | 5.9–8.2 cm | 4.1–4.6 cm | 0.14 m |
| 12,000 | 395.3 (+1.5 %: plateau) | +1.3 cm | 5.7–7.9 cm | 4.3–4.4 cm | 0.14 m |

Middle/back stance reach settled at the rest reach (0.067–0.071 m); the front legs sit forward (coxa ±22…26°) and
narrower. Training stopped at 12,2xx; final checkpoint
`logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt`.

**Gates, 1,024 episodes per condition** (`logs/eval_coordlift_prox_final.txt`,
`scripts/rsl_rl/eval_s0b_gates.sh <ckpt> coordlift_prox_final 1024`): all pass, 0 falls in 17,408 episodes.
Stride error at 0.10 m/s +1.0 / +0.3 / −0.7 % (L0 / L5 / L9; π_S0 +3.9 / +4.2 / +4.6 %), at 0.15 m/s −0.2 / −0.6 /
−1.4 %; v_x cycle RMSE ≤ 0.0046; drops vs no drops equal; turn on the spot +0.321 / −0.311 rad/s of ±0.3 (RMSE
0.039–0.048; π_S0 ±0.26); walk-and-turn v_x 0.100, yaw 0.216 of 0.2, stride +4.6 %; standing contact 0.97, drift
0.8 cm, joint speed 0.92.

**Terrain sweep** (user accepted the gait; nearest legs 3–4 cm by eye): `scripts/rsl_rl/eval_terrain_sweep.sh
<model_12200> pis0cl 0 3 6 9 10 … 20` (128 episodes per cell, `logs/eval/sweep_pis0cl_*.json`, `logs/sweep_pis0cl.txt`;
180/180 cells). **All 108 training-range cells (L0–14) pass; 0 falls in all 180 cells.** Held-out L15–20: 50/72 pass
(π_S0 52/72); failures are stride shortfalls, never falls: rough from L16 at both speeds (−11 … −18 %, the new
policy's weak spot), slope-up at 0.15 from L18, stairs-up from L18/19, stairs-down at L19–20.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.1.38.

### 2.1.39 π_S0 superseded: the gait-corrected policy (comparison)

Same task, seed, envs, mirror augmentation, DR and curriculum; differences are only the §2.1.35–38 reward changes
(rest-reach fix, footprint term removed, swing radial, touchdown reach, leg proximity).

| | old π_S0 (`…22-50-43…/model_20700`) | new π_S0 (`…00-35-51…/model_12200`) |
| --- | --- | --- |
| Training to the stop | 20,749 iterations, 12.9 h | **12,214 iterations, 7.3 h** (0.59×) |
| track_lin_vel reaches 95 % of its final value | iteration 7,283 | **iteration 3,943** |
| Gates, stride error at 0.10 m/s (L0/5/9) | +3.9 / +4.2 / +4.6 % | **+1.0 / +0.3 / −0.7 %** |
| Gates, stride error at 0.15 m/s | ≈ 0 | −0.2 / −0.6 / −1.4 % |
| v_x cycle RMSE | ≤ 0.005 | ≤ 0.0046 |
| Turn on the spot (±0.3 cmd) | ±0.26 (RMSE 0.032–0.038) | **+0.321 / −0.311** (RMSE 0.039–0.048) |
| Standing contact / drift | 0.95 / 0.9 cm | 0.97 / 0.8 cm |
| Sweep L0–14 (108 cells) | 108 pass, 0 falls; mean \|stride err\| 2.4 %, RMSE 0.0054 | 108 pass, 0 falls; 2.1 %, 0.0052 |
| Sweep held-out L15–20 (72 cells) | 52 pass, 0 falls; 6.9 %, RMSE 0.0200 | 50 pass, 0 falls; 7.7 %, **RMSE 0.0158** |
| Middle/back radial toe path per swing | ≈ 20 cm (tibia-fold lift) | **5.7–7.9 cm** (coordinated) |
| Landing vs lift-off (radial) | 5.5 cm inside | ≈ 1 cm outside |
| Middle/back stance reach (rest 6.7 cm) / half-width | 1.8–2.1 cm / 0.11 m | ≈ 6.7–7.1 cm / 0.14 m |
| Worst leg-to-leg capsule gap | −6.3 cm (legs overlap in sim) | **+1.3 cm** (3–4 cm by eye) |

**Verdict.** The visually motivated corrections produced a policy that is better or equal on every training-range
task metric, has a physically safe, coordinated gait (no leg overlap, feet placed at the rest reach, lift by femur +
tibia together), and got there in 0.59× the iterations and wall time. On the held-out suite it is mixed: lower
tracking RMSE and much better on the hardest stairs (L20 stairs-up 0.15: −26 % vs −38 % stride), weaker on high-level
rough terrain (fails from L16). Lesson for later phases: the rest-reach bug and the missing collision/landing terms
were silently shaping π_S0; gate metrics alone did not reveal them — the swing and gap probes did.

---

## Phase 2.2: S1 — blind trunk (started 2026-09-24)

Scope: paradigm spec §6 / §12 step 5 on top of π_S0 (`…/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/
model_12200.pt`). Plan agreed with the user 2026-09-24: terrain materials before S1, 8 S1 seeds (S1-K later), V_0 → V
by the user's widen-and-anneal strategy, per-env drop slope. S1b is blocked on the N_v benchmark (spec §6.2).

### 2.2.1 Spatial terrain materials — feasibility spike

Until S1 the terrain was one rigid μ-1 mesh and every env had one foot material for the whole run (§2.1.1). P's
576-D strips, the touchdown-material probe targets and V's slip test need materials that vary over space.
`scripts/tests/material_spike.py` (Hexapod-S0b-v0, flat, zero actions; `logs/material_spike*.txt`) tested writing
per-foot materials every step through the PhysX tensor API (`root_view.set_material_properties`):

| Test | Result |
| --- | --- |
| T1 per-shape writes, gravity tilted 30°: feet at μ 0.2 slide, μ 1.0 hold | PASS (≥ 9.7 m vs ≤ 4.6 mm) |
| T2 mid-contact switch 1.0 → 0.2 → 1.0 | Takes effect both ways (6.4 cm in 10 steps after the switch; odd envs ≤ 0.5 mm). 90 of 128 robots sliding at ≈ 0.6 m/s tumble when grip returns at 30° — physical; still sliding at μ 0.2 would have been ≈ 4.5 m |
| T3 friction write keeps the USD compliance | PASS (compliant − rigid sink 0.37 → 0.42 mm) |
| T5 stiffness via the restitution slot (PhysX: restitution < 0 ⇒ compliant, k = −restitution) | Works, also on feet that were rigid in USD: k 2e3 sinks 0.97 mm, k 5e4 −0.20 mm (rigid −0.26); stable |
| T4 cost | ≈ 55 µs per env written: all 4,096 envs 226 ms per call vs env.step 110 ms; 400 envs 18–22 ms; quantising values barely helps |
| T6 damping on the ground (compliant ground k_g 1e6, c_g ∈ {10, 100, 1000}) against feet with k written per step | FAIL: a compliant ground overrides the foot's written k (every foot sinks −0.26 mm = rigid), and c_g has no visible effect (landing v_z RMS 0.0017 for all three) |

### 2.2.2 Design — hybrid of per-entry submeshes and per-foot writes (user decision)

* **Field** (`mdp/materials.py`): per 4 m cell, Voronoi sites carrying library entries — 1 site (uniform, 25 %),
  2 (one straight boundary, 35 %), 3 collinear (stripes, 15 %), 4–8 (mosaic, 25 %); transition width per cell
  U[0, 0.8] m (the user asked for gradual transitions of 60–80 cm, a little larger than the robot; the sharp end is
  kept because spec §5.3 requires sharp and gradual conditions with continuously randomised sharpness). Second-site
  weight ½·(1 − clamp(2b/w, 0, 1)), b = distance to the bisector. Stored on a 2 cm grid, static, seed fixed
  (`material_field_seed = 0`) so every arm and seed sees one layout; independent of the height field (v7 §5.4).
* **Library** (approved by the user, provisional until the bench): concrete, rubber mat, PTFE (μ_s 0.2–0.3), talc/
  graphite, dry dust, foam, wet tile, wet rubber, glossy grit, polished coating; matte/shiny tag per entry; μ_d =
  μ_s·U[0.8, 1]; rigid = k 1e6. Each env redraws its values inside every entry's range at reset (per-episode palette),
  so a patch always looks like its entry while its exact values vary.
* **USD** (static, renderable — for S2 depth dropout, S3 appearance, O's M*): one visual submesh per entry (triangle →
  dominant entry at its centroid, every face assigned exactly once, semantic label per entry). The source mesh stays
  the single collider (hidden; also the HeightMap raycast target): rigid, μ 1, multiply. (First version: per-entry
  colliders — dropped for cost, §2.2.4.)
* **Per-foot physics**: μ_s, μ_d and k written to the tibia shapes from the field blend and the palette, only for envs
  where a foot that is low (≤ 2 cm) or in contact has a target that moved by > 0.01 (μ) or 0.05 (log k).
* **Damping — D25**: spike T6 failed, so per the user's fallback the damping stays on the foot, per env, from the USD
  buckets (now all compliant, c log-U[10, 200] N·s/m; fixed per run). P's and the probes' log c channel is the env's.
* Known ceiling: gradual physical transitions render as a sharp visual edge (dominant entry per triangle) until S3
  adds blended band materials from the stored weights.

Check, `scripts/tests/material_field_test.py --headless` (512 envs, flat tiles, `logs/material_field_test.txt`):
PASS — read-back = value in effect (1.2e-7), in effect within one quantum of the field target; θ = 0 stable
(max 5 mm/s, rigid entries k 1e6); tilted 20° (tan 0.364): 82/82 envs standing only on μ_s < 0.30 slide, 225/225 on
μ_d > 0.55 hold. (The first version classified footing at the end of the run and flagged envs that slid onto a grippy
patch; fixed to start-and-end footing.)

### 2.2.3 S1 implementation

| Piece | Where |
| --- | --- |
| Layouts, single source: actor 1,125 = π_S0's 117 (o_p, V_0, twist_cmd) ⊕ z_p 64 ⊕ ê 15 ⊕ zero-held M* 576, z_s/z_F 20, tokens 192, v_ℓ⊙m_ℓ 128, m_ℓ 2, σ̂_ℓ 8, rem 3; contract P 683; mirror rule per block | `mdp/layout.py` |
| E_p: causal dilated TCN, 2 convs per residual block, k 3, d {1,2,4,8} (RF 61; W 75 → +16, RF 125), hidden 128 → 64 + LayerNorm; RF ≥ W asserted | `rl/s1/tcn.py` |
| FK from `models/hexapod.urdf` (vs sim toe point 0.008 mm) and v_kin (§2.1 stance weights, no switches) | `rl/s1/kinematics.py` |
| V = v_kin + Δv̂(z_p), f̂ (6 × F_n/F_t); probes (current 84 on z_p, touchdown 102 on z_p — the O/B0 heads), β-NLL β 0.5, log σ² ∈ [−6, 4], touchdown material ×2, constant weight; VICReg var + cov; slip-weighted Δv Huber; encoder LR 0.3 × heads | `rl/s1/aux.py`, `losses.py` |
| Sequence buffer L 128: W-step history within one episode; touchdown targets = first touchdown strictly after t, back-filled, Δt^TD and body-frame p^TD formed per row at sample time | `rl/s1/seq_buffer.py` |
| Env `Hexapod-S1-v0`: materials, drop slope × U[0.5, 2] per env and episode (D24), aux owned by the env (the V_0 pattern), labels, touchdown events, contract P (nominal foothold = rest-pose toe from FK), z_p/ê on the mirrored history for the augmentation | `hexapod_env.py:HexapodS1Env`, `mdp/observations.py` |
| S1PPO: S0PPO + aux update per iteration, frozen normaliser dims (V_0 at π_S0's stats, zero-held slots at identity, ê while gated), V tests every 100 iterations → ê gate, V_0 retirement schedule, aux state in the checkpoint | `rl/s1_ppo.py`, `agents/rsl_rl_ppo_cfg.py:PPORunnerCfg_S1` |
| Loader shared by eval/probe scripts (task from actor width, aux window, env state before reset) | `rl/load_policy.py`; `eval_s0.py`, `record_gait.py`, `swing_radial_probe.py`, `leg_gap_probe.py`; gate/sweep scripts take `TASK=` |

**V_0 → V switch (user's strategy, corrected; D23).** V_0 stays in the old columns, frozen at π_S0's weights (no
online training) with its normaliser statistics frozen, so its dropout does not rescale it. ê is appended with
zero-init columns behind a binary gate: while the gate is shut the block input is exactly 0, so its columns get no
gradient (tested). When V passes v7 §4.3 on three consecutive checks (slip-stratified error < 3 × rigid; zeroing the
9 IMU dims raises the error ≥ 1.5 ×, the spec's "materially", provisional), ê's normaliser statistics are set from
aux samples while its columns are still zero (function-preserving) and the gate opens; the zero-init columns give the
gradual onset. 1,000 iterations later V_0 is dropped per episode (never per step), probability 0 → 1 over 1,000
iterations; zeros are not an unambiguous flag for a velocity (v = 0 is a real reading), which is why the end state is
permanently zero and twist_cmd separates standing. V_0's three columns remain in the layout, always zero after S1.

**Merge-gating tests** — `python scripts/tests/test_s1.py` → `S1 checks OK: widening max|Δ| = 0.00e+00 (float64);
routing, TCN RF, β-NLL, buffer, mirror`: widened actor = π_S0 over 10,000 random observations with random values in
every new block; value-loss backward leaves the actor's grads None; PPO backward leaves every aux grad None; aux
backward leaves actor and critic None; gated ê and zero-held columns get exactly zero gradient; TCN RF; β-NLL gradient
= −(y − μ)/σ; buffer history and touchdown semantics on a synthetic episode with a reset; mirror involution on all S1
groups and equality with the S0b mirror on the old 117 dims. Wiring in sim, `scripts/tests/s1_env_check.py`
(`logs/s1_env_check.txt`): PASS (shapes, zero-held and gated slots exactly 0, FK 0.008 mm, P strip/scan exact,
buffer and aux run). Training smoke test (128 envs, 24 iterations, thresholds lowered so the gate and the V_0
schedule fire, `logs/s1_smoke.txt`): runs end to end, checkpoint carries aux and schedule state.

**v_kin constants — D26** (`scripts/tests/vkin_calibrate.py`, π_S0 on Hexapod-S1-v0, 512 envs × 400 steps,
`logs/vkin_calibrate.txt`): τ_e 0.005 rad, s 0.003, δ −0.005 m → v_kin RMSE 0.063 m/s on walking steps (v_true RMS
0.138; the first guess 0.02/0.01/0.005 gave 0.087); the weights agree with the contact sensor on 90 % of foot-steps
(mean w 0.74 in stance, 0.07 in swing). The grid optimum (0, 0.001, −0.02; RMSE 0.058) weights only the lowest feet
(58 % agreement, stance w 0.25) and was rejected: Σw → 0 on uneven ground would pull v_kin toward zero. Provisional
until the bench.

**Deviations added in Phase 2.2**

| # | Deviation | Where |
| --- | --- | --- |
| D23 | V_0 kept as a frozen old block in S1 and retired by per-episode dropout (not in the §2.9 table; always zero after S1) | 2.2.3 |
| D24 | Drop slope randomised per env and episode, × U[0.5, 2] (P's drop entry) | 2.2.3 |
| D25 | Damping per env on the foot, fixed per run (spike T6: a compliant ground breaks the per-foot stiffness) | 2.2.1–2 |
| D26 | v_kin stance constants calibrated in sim | 2.2.3 |
| D27 | Material realisation: static per-entry submeshes + per-foot tensor writes; per-episode palette inside each entry's range | 2.2.2 |

### 2.2.4 Collection cost (first pilot launch stopped at ≈ 40 iterations, user decision)

The first pilot launch (`…/2026-09-25_00-27-59_s1_pilot_seed42`, superseded) ran 5.9 s per iteration (collection
4.6 s) against π_S0's 2.15 s; the user asked whether fewer envs would help. They would not: the batch (envs × steps)
is an S1 invariant shared by every seed (D1: 4,096 × 16), and fewer envs only shrink the data per iteration. The
overhead was ours. `scripts/tests/s1_profile.py --headless` (4,096 envs, `logs/s1_profile.txt`), ms per env step:

| | S1 as launched | S1 fixed | S0b |
| --- | --- | --- | --- |
| env.step | 293 | 198 | 115 |
| physics + managers (excl. S1 obs) | ≈ 175 | ≈ 102 | 115 |
| E_p + V inference, own + mirrored history | 70 (fp32) | 46 (fp16) | — |
| foot material writes (≈ 260 envs per step) | 25 | 24 | — |
| P, labels, mirror frame, buffer | ≈ 8 | ≈ 8 | — |

* **Per-entry colliders were the physics cost.** Each of the 10 entry submeshes spans the whole terrain, so every
  foot was tested against 10 terrain-wide triangle meshes (+60 ms per step; unchanged with the writer off or half the
  feet rigid). Since T6 failed the ground's physics material is uniform anyway, so the submeshes are now visual +
  semantic only and the original mesh stays the single collider (invisible). Physics back at the S0b level.
* **fp16 inference** for the rollout latents (1.6×; max |Δz| 3.5e-3 on unit-scale latents; bf16 3.3e-2 rejected).
  A receptive-cone TCN evaluation (only the positions feeding the last output) was tried and reverted: 1.5× in fp32,
  no gain over fp16, slower to train.
* **Aux training in fp16 mixed precision** (autocast + loss scaling): ≈ 485 ms per PPO iteration for 8 × 4,096
  windows (≈ 840 ms in fp32).

Result: 4.0 s per iteration at 4,096 × 16 (ETA 10,000 iterations ≈ 11 h), batch unchanged. `test_s1.py` and a
128-env training smoke test pass after the changes; the material field test was re-run on the single-collider
terrain (below).

### 2.2.5 S1 pilot (seed 42, W 50, 10,000 iterations; user request: document every 2,000 iterations)

```
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 10000 \
  --warmstart logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt \
  --run_name s1_pilot_seed42
scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0.pt>   # π_S0 baseline gates on the S1 env, then every 2,000 iterations:
                                                     # videos, probes, reward slopes, gates (256 episodes) with V tests
```

Bugs fixed before the pilot of record (launches 1–2 superseded):

* **Launch 2 crashed at ≈ 400** — `RuntimeError: Inplace update to inference tensor outside InferenceMode` in
  `S1PPO._open_gate`. The ê gate opened (V passed v7 §4.3 at 100/200/300) and its normaliser stats were written outside
  inference mode, but rsl_rl recreates `_std` inside the rollout's inference mode. *Fix:* `s1_ppo.set_block_stats` (and
  `_restore`) write inside `torch.inference_mode()`; regression check in `test_s1.py`. V at the crash already beat v_kin
  (RMSE 0.038–0.040 vs 0.060–0.064; slip / rigid 1.9–2.0 ×; IMU-zeroed 2.1–2.6 ×).
* **Probe statistics** were frozen from the first buffer fill (curriculum at levels 0–2), so `probe_td` rose 0.35 → 2.9
  as the curriculum climbed. *Fix:* fixed statistics from a π_S0 rollout covering the training distribution
  (`scripts/tests/s1_probe_stats.py`, 2,048 envs at uniform levels 0–14, 262,144 samples) →
  `hexasim/rl/s1/probe_stats_pi_s0.pt`, the default `AuxCfg.stats_path`, one file for every S1 seed and arm;
  `aux.update()` refuses to run without it; target std floored at 5e-3.
* **Kit start-up crash — root cause (not telemetry).** After launch 2 every `train.py` start crashed ≈ 1 s into Kit
  (exit 139) or hung. The native crash report showed Kit's `omni.platforminfo` calling `fork()`, whose at-fork handler
  in numpy's bundled OpenBLAS (`blas_thread_shutdown_`) crashes joining its worker threads — a race, and the real cause
  of the "telemetry flake" of §2.1.25. *Fix:* `OPENBLAS_NUM_THREADS=1` (top of `train.py` / `eval_s0.py`, exported by
  `s1_milestones.sh`); 4 of 4 clean starts after.
* **Evals next to a running S1 training** died with `Failed to get DOF velocities` after CUDA OOM (PhysX pre-allocates
  ≈ 1 GB of 4,096-env buffers; training holds 10.7–13 GB). *Fix:* `hexasim.rl.load_policy.eval_physx_buffers` sizes the
  buffers for ≤ 1,024 envs in `eval_s0.py` and the probe scripts.
* Material field test re-run on the single-collider terrain: PASS (74/74 slide, 231/231 hold).

**Pilot of record (launch 3):** `logs/rsl_rl/hexapod_s1/2026-09-25_03-28-11_s1_pilot_seed42`. Schedule: actor unfrozen
at 300; ê gate open at 399 (three consecutive V-test passes); V_0 per-episode dropout 1.0 from ≈ 2,400 (deployment
layout from then on); terrain level ≈ 9.4 (peaked 11.6 at 400).

Milestones (`logs/milestones/2026-09-25_03-28-11_s1_pilot_seed42/`, `logs/eval/s1p_it*_*.json`; gates 256 episodes on
Hexapod-S1-v0, against π_S0 on the same env):

| | π_S0 on S1 env | it 2,000 | it 4,000 | it 6,000 | it 8,000 |
| --- | --- | --- | --- | --- | --- |
| Reward (gain / 2,000 it) | — | — | 389 (+1.9 %) | 394 (+1.2 %) | 400 (+1.4 %) |
| Falls, 17 conditions | 0 % | 0 % | 0 % | 0 % | 0 % |
| Stride err 0.10 m/s L0 / L5 / L9 | +0.2 / −1.0 / −1.4 % | +2.7 / +2.5 / +2.9 % | −0.2 / −0.2 / +0.2 % | −0.8 / −0.6 / −0.1 % | +0.3 / +0.4 / +0.9 % |
| Stride err 0.15 m/s L0 / L5 / L9 | −0.9 / −1.9 / −2.1 % | +0.3 / −0.6 / −0.4 % | −1.9 / −2.2 / −2.1 % | −2.8 / −2.6 / −2.3 % | −2.1 / −2.2 / −2.0 % |
| Cycle RMSE (max walking) | ≤ 0.0050 | ≤ 0.0062 | ≤ 0.0057 | ≤ 0.0053 | ≤ 0.0050 |
| Turn ±0.3 (L0): ω̄_z | +0.320 / −0.309 | +0.317 / −0.319 | +0.306 / −0.302 | +0.307 / −0.307 | +0.291 / −0.300 |
| Walk-and-turn yaw RMSE | 0.021 | 0.008 | 0.015 | 0.012 | 0.011 |
| Standing contact L0 / L5 / L9 | 0.97 / 0.84 / 0.82 | 0.96 / 0.90 / 0.87 | 0.97 / 0.92 / 0.89 | 0.98 / 0.94 / 0.90 | 0.98 / 0.95 / 0.91 |
| V: v̂ RMSE walking; slip / rigid; IMU zeroed | — | 0.019–0.025; 1.6–2.5 ×; ≈ 6 × | 0.018–0.027; 1.5–2.0 ×; ≈ 5 × | 0.023–0.029; 1.5–1.8 ×; ≈ 5 × | 0.019–0.025; 1.7–2.2 ×; ≈ 5 × |
| Worst leg capsule gap | +1.3 cm | −0.5 cm | 0.0 cm | −0.6 cm | +0.3 cm |
| Swing lift | 4.0–4.5 cm | 4.0–4.5 cm | 4.0–4.5 cm | 3.8–4.2 cm | 3.5–4.0 cm |

Readings: every spec §4.2 gate passes at every milestone, drops within tolerance of drop-free. Once V_0 is retired,
tracking is at or better than π_S0, turning is more symmetric and standing on rough terrain improves. V is ≈ 2.5 × more
accurate than the frozen V_0 it replaces (`v0_rmse` 0.050–0.057); its training RMSE converged at ≈ 0.035 by ≈ 2,000
(v_kin alone ≈ 0.061), and its error grows with terrain level and slip share as expected. Probe NLL (β-NLL,
standardised) is flat / creeping down, rising with terrain level. **Two gait regressions:** leg clearance oscillates
around 0 (up to 7.5 % of steps under 2 cm at 0.15 m/s) and swing lift drifts down toward `cpg_z_target`'s 3.8 cm (the
spec §5 early sign of shuffling once the clock terms go). **Pilot stopped at 8,650 (user)** to diagnose them (§2.2.6).

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.2.5.

### 2.2.6 Diagnosis: swing lift and leg-to-leg clearance

**Observed Behavior.** Over the pilot, the worst same-side leg capsule gap went +1.3 (π_S0) → −0.5 / 0.0 / −0.6 /
+0.3 cm (it 2k / 4k / 6k / 8k), with 2–7.5 % of steps under 2 cm at 0.15 m/s. Swing lift went 4.0–4.5 → 3.5–4.2 cm, and
`action_beyond_clip` grew ×10.

**Hypothesis 1, rejected: friction.** Probe `scripts/tests/gait_by_material.py --cmd 0.15` (512 envs, flat; even envs
all-slippery μ 0.25 / 0.22, odd all-grippy μ 0.9 / 0.85, all rigid; `logs/gait_by_material_015.txt`). Slippery and
grippy groups have the same gait for every policy (slip speed 0.02 vs 0.01 m/s; lift, coxa range and gaps within
noise). The change is in the policy, over training:

| Policy | lift f / m / b (cm) | coxa range f / m / b (°) | worst gap / < 2 cm | clip |
| --- | --- | --- | --- | --- |
| π_S0 | 4.0 / 4.4 / 4.5 | 48 / 59 / 55 | +1.4 cm / 0.1 % | 0.015 |
| S1 2,000 | 4.2 / 4.7 / 4.5 | 50 / 75 / 51 | −0.1 / 3.5 % | 0.015 |
| S1 4,000 | 4.3 / 4.6 / 4.6 | 60 / 66 / 61 | +0.7 / 1.7 % | 0.016 |
| S1 8,650 | 3.7 / 4.0 / 4.1 | 76 / 64 / 64 | −0.1 / 6.2 % | 0.019 |

**Mechanism.** Stance geometry at 0.15 m/s (milestone `record_gait` output): reach front / middle / back 0.079 / 0.082 /
0.088 m (π_S0) → 0.053 / 0.066 / 0.085 m (S1 8,000). Middle-leg mean coxa angle ±10° → ±28° (swept back toward the
back legs), half-width 0.139 → 0.124 m. The stride is fixed by v / f_ref, and stride ≈ coxa arc × reach, so toes pulled
toward the hips need larger coxa arcs. Neighbouring legs then close in (RM–RB worst), and the coxa targets run into
the servo clip.

**Root cause: the reward's own optimum.** Per-term reward on the same probe (weighted, per second, 0.15 m/s,
`logs/gait_by_material_015b.txt`):

| Term | π_S0 | S1 8,650 | Δ |
| --- | --- | --- | --- |
| touchdown_reach | −1.29 | −0.59 | +0.70 |
| cpg_z_target | 4.14 | 4.70 | +0.55 |
| cpg_stance_rigidity | −2.14 | −1.83 | +0.32 |
| swing_radial | −0.75 | −0.68 | +0.06 |
| leg_proximity | −0.001 | −0.61 | −0.61 |
| action_beyond_clip | 0 | −0.016 | −0.02 |
| landing radius f / m / b (cm; target 6.7) | 12.0 / 7.7 / 6.2 | 9.1 / 6.8 / 6.2 | |

`touchdown_reach` (§2.1.37) is two-sided: it pays for landing at the single rest reach (6.7 cm), so it also pulls the
front legs in from their natural 12 cm. That reach gain (+0.70 / s) is bigger than the proximity cost (−0.61 / s), so
overlapping legs are profitable. S1 did not create this: π_S0 was stopped on a return plateau while `touchdown_reach`
was still improving (−1.19 per episode-second at its end), and S1 continued optimising the same S0b reward. The lift
trend is `cpg_z_target` converging onto its own 3.8 cm half-sine: π_S0 overshot at 4.0–4.5 cm, and the term pays
+0.55 / s more for the lower swing. It becomes a real problem in S2: once the clock terms leave, only
`swing_clearance` defends lift, and it saturates at 3 cm (§2.1.1).

**Fix (user decisions, 2026-09-25).** Three changes to the S0b reward, shared by every arm:

1. `touchdown_reach` made **one-sided**: Σ at landing of max(0, rest reach − |r_xy|). It still penalises tucked
   landings (§2.1.37), but no longer pulls outward landings inward.
2. `swing_clearance` saturation height **3 → 4 cm**, so the phase-free floor holds the lift once the clock terms leave
   in S2. `cpg_z_target` still peaks at 3.8 cm; the two disagree by ≤ 4 % of the kernel at the apex, which is
   negligible.
3. `leg_proximity` **×400 → ×800**.

Economics on the same probe after changes 1 and 2 (`logs/gait_by_material_newreward.txt`): `touchdown_reach` is equal
for both postures (π_S0 −0.093, S1 8,650 −0.099 per s), so the +0.70 / s pull is gone. The S1 posture was still
+0.31 / s ahead overall (cpg_z_target +0.55, cpg_stance_rigidity +0.32, tracking and radial +0.23; leg_proximity
−0.61, other −0.18). Those gains may be separable from the posture, but to be safe the proximity weight was doubled:
overlap then nets −0.30 / s, while gaits with 2 cm of room pay 0. The 4 cm floor barely changes `swing_clearance` on
these policies (0.479 vs 0.468 per s: both lift ≈ 4 cm).

`reward_economy.py --task Hexapod-S0b-v0 --ckpt <π_S0> --cmd 0.15 0 0 --terrain_level 0`
(`logs/s0b_economy_onesided.txt`): walk +1.013 ≫ stand +0.267 ≫ fall −0.10 per step. Deviation D28: S0b reward changed
after π_S0 (these three terms); π_S0 is retrained from scratch under it.

**S0 retrained from scratch** (user decision), same task, seed, envs, mirror augmentation, DR and curriculum as
§2.1.38:
```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 18000 \
  --run_name s0_onesided_reach_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
# run dir logs/rsl_rl/hexapod_s0b/2026-09-25_22-23-35_s0_onesided_reach_fromscratch_seed42
scripts/rsl_rl/milestone_review.sh <run_dir> <iter>   # every 2,000 iterations
```

**The one-sided retrain sprawled; abandoned at 6,250 (user decision).** With the outward half of `touchdown_reach`
gone, the uncapped reach weighting in `cpg_coxa_contact` (`coxa_stroke_weight` = sign × |r_xy| / rest reach,
§2.1.26) paid extension. At iteration 2,000 its ledger against the old run showed cpg_coxa_contact +1.33 and
touchdown_reach +1.27 per episode-second (cpg_z_target −1.13, track_yaw −1.09, standing_pose −0.90). The legs
straightened out and the back legs stopped lifting:

| | reach f / m / b (m) | half-width | back-leg lift | worst gap |
| --- | --- | --- | --- | --- |
| it 2,000 | 0.16 / 0.18 / 0.18 | 0.228 m | 0.7 cm | −0.6 cm |
| it 4,000 | 0.14 / 0.18 / 0.16 | 0.225 m | 0.0–0.1 cm | +0.8 cm |
| it 6,000 | 0.13 / 0.18 / 0.16 | 0.218 m | 0.5–0.6 cm | +1.0 cm |

Run kept as a record: `logs/rsl_rl/hexapod_s0b/2026-09-25_22-23-35_s0_onesided_reach_fromscratch_seed42`. The two-sided
`touchdown_reach` had been holding the coxa term's extension incentive in check. It also produced the coordinated lift,
so it stays.

### 2.2.7 S1 pilot v2 — clearance via leg_proximity, lift floor 4 cm, plateau-based budget

Decisions with the user (2026-09-26):

- π_S0 (`…/2026-09-24_00-35-51…/model_12200.pt`) stays.
- The S0b reward is reverted to exactly π_S0's form: `rewards.py` restored from HEAD; `swing_clearance` 3 cm and
  `leg_proximity` −400 in `S0bRewardsCfg`.
- Two S1-only overrides (`HexapodS1EnvCfg.__post_init__`, deviation **D28**: S1 rewards = S0b except these, shared by
  every arm): `leg_proximity` **−1,600**; `swing_clearance` height **4 cm**.
- Budget: stop at the first 2,000-iteration milestone ≥ 4,000 whose `Train/mean_reward` gain is < 1.5 % over the last
  2,000 iterations; cap 12,000. The stop iteration becomes the S1 budget for every seed.

With the two-sided reach kept, the overlap posture's advantage was ≈ +1.0 / s, so ×800 would still have left it
≈ +0.4 / s ahead; ×1,600 was chosen for margin. Economics under the S1 v2 reward (`gait_by_material.py --cmd 0.15`,
π_S0 vs pilot-1 model_8650, `logs/gait_by_material_s1v2.txt`): the overlap posture now nets **−0.69 / s** against π_S0
(leg_proximity −2.42 outweighs cpg_z_target +0.55, touchdown_reach +0.69, stance rigidity +0.33, tracking +0.22).
`reward_economy.py --task Hexapod-S1-v0` (`logs/s1_economy_v2.txt`): walk +0.982 ≫ stand +0.266 ≫ fall −0.10 per step
(the script now recognises S1 as a command task). `s1_milestones.sh <run> <π_S0> 12000 4000 1.5` adds
`gait_by_material.py` to the review and performs the plateau stop (`BUDGET` file).

```
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 12000 \
  --warmstart logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt \
  --run_name s1_pilot2_seed42        # run dir logs/rsl_rl/hexapod_s1/2026-09-26_02-33-23_s1_pilot2_seed42
```
Run config verified: leg_proximity weight −1600.0, swing_clearance height 0.04.

Milestones (`logs/milestones/2026-09-26_02-33-23_s1_pilot2_seed42/`; schedule identical to pilot 1):

| | π_S0 (S1 env) | it 2,000 | it 4,000 | it 6,000 | it 8,000 |
| --- | --- | --- | --- | --- | --- |
| Reward gain / 2,000 it | — | — (380) | +1.54 % | +2.17 % | **+1.47 % → plateau** |
| Falls, 17 conditions | 0 % | 0 % | 0 % | 0 % | 0 % |
| Stride err 0.10 / 0.15 m/s (L0) | +0.2 / −0.9 % | +1.0 / +0.2 % | −0.7 / −2.3 % | −0.1 / −2.8 % | +1.3 / −2.3 % |
| Cycle RMSE (max walking) | ≤ 0.0050 | ≤ 0.0063 | ≤ 0.0055 | ≤ 0.0052 | ≤ 0.0048 |
| Turn ±0.3: ω̄_z | +0.320 / −0.309 | +0.311 / −0.315 | +0.306 / −0.307 | +0.297 / −0.296 | +0.310 / −0.313 |
| Standing contact L0 / L5 / L9 | 0.97 / 0.84 / 0.82 | 0.97 / 0.90 / 0.88 | 0.97 / 0.92 / 0.89 | 0.98 / 0.95 / 0.91 | 0.98 / 0.95 / 0.91 |
| V: v̂ RMSE; slip / rigid; IMU zeroed | — | 0.021–0.027; ≤ 1.8 ×; ≈ 7 × | 0.019–0.027; ≤ 2.0 ×; ≈ 7 × | 0.020–0.026; ≤ 2.0 ×; ≈ 8 × | 0.019–0.025; ≤ 2.2 ×; ≈ 8–9 × |
| **Worst leg gap; < 2 cm at 0.15 / 0.20** | +1.3 cm; ≈ 0 % | +1.0 cm; ≤ 1.4 / ≤ 1.0 % | +0.8 cm; ≤ 1.2 / 0 % | +1.0 cm; ≤ 1.3 / ≤ 0.1 % | **+1.3 cm; ≤ 2.7 / ≤ 0.1 %** |
| **Lift f / m / b (swing probe)** | 4.0–4.5 cm | 4.0 / 4.7 / 4.6 | 3.8 / 4.1 / 4.7 | 3.65 / 3.7 / 4.4 | **3.5–3.7 / 3.8–3.9 / 4.1–4.2** |
| Front stance reach; front coxa range | 0.079 m; 48° | 0.071–0.076 m; 50° | 0.07 m; 59° | 0.061–0.064 m; 68° | 0.052 m; 72° |

Clearance held at π_S0's level for the whole run (pilot 1 reached −0.6 cm). Lift converged from above onto
`cpg_z_target`'s 3.8 cm half-sine peak: the 4 cm floor (×5, ≈ 0.48 / s at saturation) is too small to hold it above
the clock term's target while that term is present. The front legs crept inward again (reach 0.079 → 0.052 m), held
off only by the proximity weight.

**S1 budget = 8,000 iterations** (≈ 524 M env steps at 4,096 × 16), fixed for the W sweep and every S1 seed. Pilot-2
checkpoint `logs/rsl_rl/hexapod_s1/2026-09-26_02-33-23_s1_pilot2_seed42/model_8000.pt`. The wrapper compared the
*rounded* printed gain ("+1.5") and did not fire; training was stopped by hand at 8,500 and `BUDGET` written. The
parser now computes the gain from the raw values (re-checked: 4,000 1.543 % continue, 6,000 2.169 % continue, 8,000
1.473 % stop).

**S1 pilot verdict** against the spec §6.1 S1 gate: performance ≥ S0 on the S1 env (every S0b gate, 0 falls; turning
and rough-terrain standing better than π_S0); V passes v7 §4.3 (gate open at ≈ 400; slip / rigid ≤ 2.2 ×; IMU dropout
≈ 8–9 ×); probe NLL flat (current −0.10, touchdown rising only with terrain level); V_0 retired at ≈ 2,400; return
plateaued at 8,000.

**Residual watch items, carried into S2 design:**

1. The reach pull on the front legs persists: reach 0.052 m, coxa range 72°, landing 9.5 cm against 12 at π_S0.
   Clearance holds only because of the proximity weight.
2. Front and middle lift converge onto `cpg_z_target`'s 3.8 cm peak (front 3.5–3.7 cm on the swing probe, 3.9 on
   the material probe). Once the clock term leaves in S2, only the 4 cm floor (×5, ≈ 0.48 / s) holds lift. Its weight
   should be re-checked with `reward_economy.py` against the S2 reward before S2.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.2.7.

### 2.2.7a S1 pilot v2 — terrain-type gait videos at level 13 (renumbered from a duplicate §2.2.8)

`scripts/rsl_rl/record_gait.py` gained `--terrain_type` (one of `mdp.terrain.EVAL_TERRAIN_TYPES`) and `--terrain_level`:
the generator is replaced with `single_type_generator(name)` (21 levels × 4 columns, same difficulty curve as training
and the sweep) and `pin_terrain_level` uses the requested level. The flat/no-flag path is unchanged (regression check:
mean v_x 0.151 m/s at 0.15).

Five 15 s side-view clips of pilot-2 `model_8000.pt` at v_x 0.15 m/s, level 13, one per terrain type (mean v_x: rough
0.142, bumps 0.145, slope_up 0.146, slope_down 0.151, stairs_up 0.151 m/s), in
`logs/milestones/2026-09-26_02-33-23_s1_pilot2_seed42/it8000_<type>_L13_side-step-0.mp4` (+ `it8000_terrain_L13.txt`).
No falls; DR, dropped reads and the S1 material field at training defaults; frames confirmed to show each terrain.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.2.7a.

### 2.2.8 S1 pilot 3 — cpg_z_target 4 cm, per-leg touchdown_reach (user decisions, 2026-09-26)

Pilot 2 (§2.2.7) fixed clearance but left two residuals: lift converging onto `cpg_z_target`'s 3.8 cm peak (the term
pays 4.6 / s against the 4 cm floor's 0.5 / s), and the front legs pulled inward by `touchdown_reach`'s single 6.7 cm
target (front reach 0.079 → 0.052 m, coxa range 48° → 72°, landing 12 → 9.5 cm). Changes, S1-only (deviation **D29**),
on top of pilot 2's (`leg_proximity` −1,600, `swing_clearance` 4 cm):

1. `cpg_z_target` `step_height` **0.038 → 0.040**: the clock term and the floor now target the same height.
2. `touchdown_reach` **per-leg, two-sided**: front legs' landing target `front_reach` = **10 cm** (π_S0 lands them at
   ≈ 12 cm; the user asked for "a couple cm less"), middle/back legs keep the rest reach (6.7 cm). Weight unchanged
   (−600). Kept two-sided on purpose: a one-sided form let the coxa term sprawl the legs (§2.2.6). `front_reach: None`
   reproduces the S0b term (S0b/π_S0 unchanged). Not raised: more weight would pull harder toward the target.
3. Budget: 8,000 iterations (pilot 2's plateau, §2.2.7), reviews at 4,000 then every 2,000; no plateau stop.

**Economics** (`gait_by_material.py --cmd 0.15`, `logs/gait_by_material_pilot3_econ.txt`; `logs/s1_economy_v3.txt`):
the new `touchdown_reach` costs π_S0 −0.66 / s (was −1.29 at the single target) and pilot 2's posture −0.32 / s;
walk +0.997 ≫ stand +0.264 ≫ fall −0.10 per step. Prediction: the front legs settle at ≈ 10 cm (L1 minimum).

**Regression watch** (any breach → stop and report): worst same-side leg gap ≥ 0 cm and < 2 cm on ≤ 3 % of steps at
0.15 / 0.20 m/s; lift ≥ 3.8 cm on every leg type; front landing reach and coxa range (pilot 2: 9.5 cm, 72°); coxa clip
term ≤ 0.02; all S0b gates; V within v7 §4.3.

```
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 8000 \
  --warmstart logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt \
  --run_name s1_pilot3_seed42      # run dir logs/rsl_rl/hexapod_s1/2026-09-26_13-50-10_s1_pilot3_seed42
scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0> 8000 4000
# stopped at 4,500 (watch breach, below), then resumed:
python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3500 \
  --resume --load_run 2026-09-26_13-50-10_s1_pilot3_seed42 --checkpoint model_4500.pt \
  --run_name s1_pilot3_seed42_resume4500     # run dir logs/rsl_rl/hexapod_s1/2026-09-26_19-41-49_s1_pilot3_seed42_resume4500
scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0> 8000 6000
```
Run config verified: `cpg_z_target.step_height` 0.04, `touchdown_reach.front_reach` 0.1, `leg_proximity` −1,600,
`swing_clearance.height` 0.04.

**Milestone 4,000 → stopped at 4,500 on a watch breach.** Gates pass, V passes, worst gap +0.7 cm. Two breaches: the
coxa clip term reached 0.025 (> 0.02) — all of the increase is the **back-leg coxa** hitting the ±55° stop (0.0037 vs
0.0009 rad/step in pilot 2), because both pilots drift the coxa stance centre rearward (right-side mean coxa
−22…−27° vs −1…−11° in π_S0), pilot 3 faster; and lift overshot the target (4.8 / 4.8 / 5.1 cm for 4 cm), which looked
like the early part of pilot 2's own overshoot-then-converge. Front reach did what the economics predicted (landing
11.7–11.8 cm, coxa range 55–58°). The cause of the back-coxa drift (4 cm target, proximity −1,600 or front reach) was
not established. **User: resume to 8,000, decide at 6,000** (continue if clip < 0.02 and lift falling). The resume path
was verified first (ê gate open, V_0 retired, actor unfrozen; 128-env smoke); differences from an uninterrupted run:
RNG restarts at seed 42 and the aux buffer refills. `max_iterations` counts from the loaded iteration;
`milestone_review.sh` now merges every event file in a run dir so the reward slope spans the join.

**Milestone 6,000:** gates pass; lift 4.2 / 4.5 / 4.4 cm (converging from above); clip term 0.0206, back coxa 0.0016 —
0.0006 over the line but falling at every check (0.025 → 0.022 → 0.0206), judged as a trend and continued. Front landing
11.4 cm, front coxa range 55° (pilot 2 at 6,000: 10.0 cm, 68°): **the inward drift is gone**. New watch item: speed
under-tracking (0.15 m/s stride −4.1 %, v_x 0.145), inside the gate.

→ Full log (commands, per-run tables): `phase2-run-logs.md` §2.2.8.

#### Pilot 3 — milestone 8,000 (`…_resume4500/model_7999.pt`; probe files named `it7999_*`, gates and material probe `it8000_*`)

Training finished at 8,000 (budget). Plateau test: 394.8 → 399.6 = +1.2 % (< 1.5 %). Terrain level 9.38.

| | π_S0 | pilot 2 it 8,000 | pilot 3 it 6,000 | **pilot 3 it 8,000** |
| --- | --- | --- | --- | --- |
| Falls, 17 conditions | 0 % | 0 % | 0 % | 0 % |
| Stride err 0.10 (L0 / L5 / L9) | +0.2 / −1.0 / −1.4 % | +1.3 / +1.5 / +2.0 % | −2.2 / −2.2 / −1.7 % | **+0.5 / +0.2 / +0.7 %** |
| Stride err 0.15 (L0 / L5 / L9); v_x L0 | −0.9 / −1.9 / −2.1 % | −2.3 / −2.4 / −2.2 %; 0.148 | −4.1 / −4.2 / −3.9 %; 0.145 | **−2.6 / −2.9 / −2.7 %; 0.1475** |
| Cycle RMSE (max walking); drops vs no drops | ≤ 0.0050; ≈ equal | ≤ 0.0048; equal | ≤ 0.0062; equal | ≤ 0.0051; 0.0034 / 0.0035, 0.0035 / 0.0035 |
| Turn ±0.3 ω̄_z, RMSE; walk-and-turn yaw RMSE | +0.320 / −0.309, 0.037–0.045; 0.021 | +0.310 / −0.313, 0.019–0.020; 0.007 | +0.303 / −0.298, 0.017–0.019; 0.009 | +0.309 / −0.306, 0.017; 0.007 |
| Standing contact L0 / L5 / L9 | 0.97 / 0.84 / 0.82 | 0.98 / 0.95 / 0.91 | 0.98 / 0.94 / 0.90 | 0.98 / 0.95 / 0.91 |
| V: v̂ RMSE; slip / rigid; IMU zeroed | — | 0.019–0.025; ≤ 2.2 ×; ≈ 8–9 × | 0.022–0.026; ≤ 2.0 ×; ≈ 9 × | 0.019–0.025; ≤ 2.2 ×; ≈ 8–9 × |
| Worst leg gap; < 2 cm at 0.15 / 0.20 | +1.3 cm; ≈ 0 % | +1.3 cm; ≤ 2.7 / ≤ 0.1 % | +1.1 cm; ≤ 0.4 / 0 % | **+0.5 cm; ≤ 1.8 / ≤ 0.1 %** |
| Lift f / m / b: swing probe (material probe) | 4.0–4.5 cm | 3.5–3.7 / 3.8–3.9 / 4.1–4.2 (3.9 / 4.0 / 4.2) | 3.9 / 4.3 / 4.4 (4.2 / 4.5 / 4.4) | **3.8–3.9 / 4.3–4.4 / 4.3 (4.1 / 4.5 / 4.3) cm** |
| Gate apex clearance L0 / L9 | ≈ 4.0 / 4.4 cm | 3.8 / 4.2 cm | 4.1 / 4.5 cm | 4.1 / 4.4 cm |
| Landing reach f / m / b (cm); front coxa range | 11.9 / 7.7 / 6.2; 49° | 9.5 / 7.4 / 6.4; 72° | 11.4 / 6.3 / 5.9; 55° | **11.2 / 6.5 / 6.2; 60°** |
| **Coxa clip term (rad/step)** | 0.015 | 0.022 | 0.0206 | **0.026–0.029** |
| — back / middle coxa | 0.0008 / 0.0008 | — | 0.0016 / 0.0013 | **0.0031 / 0.0019** |
| Weighted `action_beyond_clip` (per s) | −0.01 | −0.017 | −0.011 | **−0.051** |

**Regression watch verdict.**

- Clean: all 17 gate conditions (0 falls, RMSE ≤ 0.03, stride ±10 %, drops within tolerance), V within v7 §4.3, standing
  and turning better than π_S0, speed under-tracking recovered (−4.1 → −2.7 %).
- Improved over pilot 2: lift converged near 4 cm (front 3.8–4.1 rather than 3.5–3.7), front legs kept out of the
  inward drift (landing 11.2 cm vs 9.5; coxa range 60° vs 72°), touchdown_reach cost −0.37 / s.
- Open: the **coxa clip term rose again after 6,000** (0.0206 → 0.026–0.029), back coxa 0.0016 → 0.0031 and middle coxa
  0.0013 → 0.0019, against 0.022 in pilot 2 at the same iteration and 0.015 in π_S0. That is above the 0.02 line and
  moving up, which under the rule set at the 6,000 decision means the `action_beyond_clip` fix. The term's
  contribution to the reward is small (−0.05 / s of ≈ 40 / s), and requested-beyond-limit targets are clipped by the
  servo, not executed. Leg gap +0.5 cm is the lowest of the runs since pilot 1, with 1.8 % of steps under 2 cm for the
  LM–LB pair at 0.15 m/s (limit 3 %).

**Decision (user, 2026-09-27): accept pilot 3 as the S1 configuration; no `action_beyond_clip` change (D30).**

The S1 reward is final: S0b's reward with `leg_proximity` −1,600, `swing_clearance` height 4 cm, `cpg_z_target`
`step_height` 4 cm and `touchdown_reach` with `front_reach` 10 cm (deviations D28, D29; two-sided, weight −600 unchanged).
Budget 8,000 iterations. The coxa clip term (0.026–0.029 rad/step at 8,000 against 0.015 in π_S0) is recorded as a known
S1 imperfection: −0.05 / s of ≈ 40 / s, requested-beyond-limit targets are clipped by the servo, no gate is affected.
Revisit it in the S2 reward design, where the coxa terms anneal. Pilot 3 counts as seed 42 / W = 50 for the paired
branching: its code is the final S1 code, with one difference to a clean run (resumed at 4,500 with a restarted RNG and a
refilled aux buffer). Pilot-3 checkpoint: `logs/rsl_rl/hexapod_s1/2026-09-26_19-41-49_s1_pilot3_seed42_resume4500/model_7999.pt`
(first 4,500 iterations: `…2026-09-26_13-50-10_s1_pilot3_seed42/model_4500.pt`).

**Remaining S1 steps (not started):** the W sweep (spec §9.1) at seed 42 and 8,000 iterations, W = 25 and 75
(`env.aux.window=25|75`; the aux frame statistics do not depend on W); the smallest W within noise of the best on V's
slip-stratified error and the current-probe NLL is fixed for every arm. Then the 8 S1 seeds at the chosen W.

### 2.2.9 Audit: demo / Play-env changes did not touch training or evaluation (2026-09-29)

Changes made around pilot 3 for gait demos (`research/playenv-teleop-updates.md`): `Hexapod-S1-Play-v0` and
`episode_length_s = 10000` in every `*_PLAY` config; the S1 Play config reverts `cpg_z_target` (3.8 cm) and
`touchdown_reach.front_reach` (None) to pilot-2 values; `demo_teleop.py` and `record_gait.py` switched to the Play env;
`S0B_TERRAIN_CFG` gained `color_scheme="random"`. Timeline: pilot 3's first half started 09-26 13:50, the config edits
landed 14:52–15:13, the resume started 19:41, and the W sweep and seeds 43–46 ran after.

| Check | Result |
| --- | --- |
| Training config `HexapodS1EnvCfg` (git diff vs HEAD) | Only the pilot-3 overrides; every demo change is confined to `*_PLAY` classes and the new registration; no training or eval class inherits from a Play class |
| Saved `params/env.yaml` / `agent.yaml` of pilot 3 (both halves), W 25 / 75, seeds 43–46 | Identical to pilot 3's first half except each run's intended seed / window and `color_scheme` |
| Terrain geometry, `color_scheme` none vs random (same global seed) | **Bit-identical** (1,673,248 vertices and faces): Isaac Lab colours the mesh after all sub-terrains are generated; material submeshes and the HeightMap use the USD points, not the colours |
| Evals (`eval_s0.py`, `gait_by_material.py`) | Use `Hexapod-S1-v0`, never the Play env |
| `record_gait.py` (milestone videos, geometry tables) → Play env | No measurable effect: 10 s recordings already fitted the 10 s training episode; the reverted reward values do not enter the observation or the actions |

**Found while auditing (pre-existing, not a regression):** Isaac Lab's rough and bump sub-terrains draw from numpy's
*global* RNG (`hf_terrains.py`), not the generator's seeded one, so the terrain geometry depends on the run's seed
(seed 43 builds 1,673,241 vertices, seed 42 1,673,248; same distribution). Every S1 seed therefore trains on its own
terrain instance, which is part of the measured seed variance. Arms branching at S2 from the same S1 seed with the same
seed share the terrain (pairing holds); evaluations must keep the eval seed fixed across arms (`eval_s0.py --seed`).

Doc notes: `playenv-teleop-updates.md` says `front_reach` "defaults to 10 cm" — the function default is `None` (the
S0b term); 10 cm is set by `HexapodS1EnvCfg`. The S1 Play config is not the accepted S1 reward and must not be used for
reportable numbers.

### 2.2.10 W sweep (spec §9.1): W = 50 kept

W = 25 and W = 75 trained at seed 42 for 8,000 iterations with the final S1 code and config (driver
`scripts/rsl_rl/run_s1_sweep_and_seeds.sh sweep`; only difference `env.aux.window`), evaluated like the seeds (256
episodes × 17 gate conditions, material probe, leg-gap / swing probes). Summary by
`scripts/rsl_rl/summarize_s1_runs.py` (`logs/s1_sweep_summary.md`); full table in `phase2-run-logs.md` §2.2.10.

| | W 25 | W 50, seed 42 (pilot 3, resumed) | W 75 | **W 50, seeds 43–46 range** (clean runs) |
| --- | --- | --- | --- | --- |
| Gates (17 conditions) | all pass, 0 falls | all pass, 0 falls | all pass, 0 falls | all pass, 0 falls |
| V slip RMSE, training 6k–8k (m/s) | **0.0527** | 0.0564 | 0.0597 | 0.0550–0.0566 |
| V rigid RMSE, training 6k–8k | 0.0269 | 0.0284 | 0.0294 | 0.0281–0.0283 |
| Current-probe NLL, training 6k–8k | −0.0904 | −0.0999 | −0.1034 | −0.1132 … −0.1069 |
| Touchdown-probe NLL | 0.0863 | 0.0742 | 0.0660 | 0.0569–0.0667 |
| Gait (lift, reach, gap, coxa range) | equivalent | equivalent | equivalent | — |

The noise yardstick is the spread of seeds **43–46**, not 42: pilot 3 was resumed at 4,500 with a restarted RNG and a
refilled aux buffer, and it is the outlier on the probe NLL (revised 2026-09-30 after the external S1 review; the
first version of this section used 42–45).

**What the §9.1 rule gives.** The rule is "the smallest W within noise of the best, on V's slip-stratified error and the
current-probe NLL". Against the 43–46 range:

- **W 25** lies outside the range on both criteria, in opposite directions: better on V (0.0527 < 0.0550, ≈ 4 seed sd),
  worse on the current probe (−0.090 > −0.107) and on the touchdown probe (0.086 > 0.067). It is *not* within noise of
  the best on the probe criterion, so the rule does not select it (and the seeds need no rerun).
- **W 75** is worse than every clean W 50 seed on V (0.0597) *and* on the current probe (−0.103 > −0.107); only its
  touchdown NLL (0.066) is inside the range. Dominated.
- **W 50** is not the best on V either, so the rule, applied literally, selects **no** window: the two criteria disagree.

**Decision: W = 50 for every arm — a deviation (D32), not a sweep result.** Reason: blind S1 scoring is biased toward
short windows. Both §9.1 criteria are measured on the S1 limit-cycle gait, where one gait period of history suffices;
what a longer window buys — state estimation on off-limit-cycle gaits once the clock leaves in S2, touchdown timing
(W 25 = 0.5 s is below one stance at the 0.8 Hz floor, 31 steps) — is invisible to a blind S1 metric. The weaker W 25
touchdown probe is the one S1-measurable signal of that capacity. Revisit only if an S2 pilot shows no W effect.

Caveat: one run per W (seed 42, not resumed for W 25 / 75). Gait effects of W are within seed-to-seed scatter.

### 2.2.11 S1 seeds (W 50): the result is robust across seeds

Seeds 43–46 trained with the final S1 config (8,000 iterations each, ≈ 9.2–9.7 h, no resumes, every first attempt
clean; driver `scripts/rsl_rl/run_s1_sweep_and_seeds.sh seeds 50`, per-seed logs `logs/s1_w50_seed<N>.log`); seed 42 is
pilot 3. All five evaluated identically (gates on `Hexapod-S1-v0`, 256 episodes × 17 conditions; material probe;
leg-gap / swing probes); summary `logs/s1_seeds_summary_5seeds.md` (4-seed interim
`logs/s1_seeds_summary_4seeds.md`), full per-seed table in `phase2-run-logs.md` §2.2.11.

Across all five seeds 42–46 (mean ± sd [min, max]):

| Metric | Across seeds | π_S0 on the S1 env |
| --- | --- | --- |
| Falls, 85 gate cells (21,760 episodes) | **0 in every cell** | 0 |
| Every S0b gate (RMSE ≤ 0.03, stride ±10 %, drops within +5 %) | **pass on every seed** | pass |
| Cycle RMSE, worst walking cell (m/s) | 0.0055 ± 0.0008 [0.0048, 0.0067] | ≤ 0.0050 |
| Stride err 0.10 / 0.15 m/s, L0 | +1.2 ± 1.1 % [+0.3, +2.9]; −1.8 ± 1.1 % [−2.6, +0.1] | +0.2 %; −0.9 % |
| Turn on the spot ±0.3 rad/s | +0.302 ± 0.004 / −0.301 ± 0.005 | +0.320 / −0.309 |
| Standing contact, L9 | 0.918 ± 0.010 | 0.82 |
| V slip RMSE (training 6k–8k) | 0.0561 ± 0.0006 | — |
| V slip / rigid (eval, worst cell); IMU-zero ratio (eval, weakest cell) | 2.24 ± 0.08 ×; 8.3 ± 2.3 × | — |
| Probe NLL current / touchdown | −0.107 ± 0.005 / 0.065 ± 0.006 | — |
| Mean reward (training 6k–8k) | 397.6 ± 1.6 | — |
| Worst leg capsule gap; < 2 cm share at 0.15 | +0.88 ± 0.37 cm [+0.5, +1.4]; 1.6 ± 1.0 % | +1.3 cm; ≈ 0 % |
| Front lift (material probe); front landing reach; front coxa range | 4.35 ± 0.24 cm; 11.1 ± 0.1 cm; 59.7 ± 1.0° | 4.0 cm; 11.9 cm; 49° |
| Coxa clip term (rad/step) | 0.020 ± 0.004 (seed 42 0.026; fresh seeds 0.018–0.019) | 0.015 |

**Reading.** All five seeds pass every gate with zero falls, and the spread of every task, estimator and gait metric is
small next to its gate margin: the S1 result is a property of the configuration, not of seed 42, so it is **robust
rather than lucky**. Each seed also trained on its own terrain instance (§2.2.9), so the result is robust to the
terrain draw as well. Adding seed 46 to the four-seed set moved no aggregate materially (e.g. V slip RMSE
0.0561 ± 0.0007 → 0.0561 ± 0.0006; worst leg gap +0.95 → +0.88 cm), which is itself evidence the spread is
converged. Seed-specific differences are all inside the gates: seed 44 calibrates speed slightly high (+2.9 % stride at
0.10 m/s, +0.1 % at 0.15) and has the highest cycle RMSE (0.0067); seed 45 shows the strongest IMU dependence in V
(12.1 × vs 6.5–8.9 × elsewhere). Seed 42 (pilot 3) is the outlier on two metrics, both likely from its resume at 4,500:
the weakest probe NLL (−0.0999 vs −0.107 … −0.113) and the highest coxa clip term. **D30 is therefore mostly a pilot-3
artefact**; the four fresh seeds sit at 0.0182–0.0193 rad/step, still above π_S0's 0.015.

Seed-level numbers are the input to the paired statistics of spec §10.1: each S2 arm branches from one of these five
trunks, so these five are the pairing units (D31: five trunks cap B2 / F at the spec's hard floor).

### 2.2.12 Phase 2.2 (S1) status: what is left, deviations, lessons, readiness

**Left in S1:**

1. ~~Seed 46~~: **done** (finished 2026-09-30 03:00, evaluated; §2.2.11 covers all five seeds).
2. **S1-K ×5** (B1-K's own trunk, spec §10.1): not started, by design. B1-K is the first arm to cut if compute is short
   (spec §10.2 cut order B3 → B1-K to 3 seeds → B1-M to 3), and it branches from its own trunk (KiVi kinesthetic
   module, 10-step history, z_K VAE), not from these S1 seeds. It can be trained any time before B1-K's S2.
3. S1b (blind batch transition at N_v × T_v, spec §6.0): blocked on the N_v benchmark (spec §6.2).

**Deviations added in Phase 2.2:** D23–D30 (§2.2.3–2.2.8, see `research_progress.md`), and

| # | Deviation | Why |
| --- | --- | --- |
| D31 | S1 trunk seeds: 5 (42–46), not 8 | User decision (compute). Spec: B2 / F target 8 paired seeds, hard floor 5; every arm except B1-K branches from an S1 seed, so 5 trunks cap B2 / F at the floor unless three more trunks are trained |
| D32 | W 50 kept as a deviation: the §9.1 rule selects no window (W 25 better on V, worse on both probes, vs the clean seeds 43–46) | Blind S1 metrics cannot see what a longer window buys in S2 / S3 (§2.2.10) |

**Keep in mind for S2 and later:**

1. **Longer training exposes the reward's optimum.** Pilot 1's overlap and lift drift, and pilot 3's clip term, appeared
   only after 4,000–8,000 iterations. S2 / S3 are several times longer: expect every remaining reward imperfection to
   surface there. Carry the regression watch (leg gap, lift per leg type, landing reach, coxa range, clip term) into
   every S2 / S3 milestone.
2. **Known S1 reward imperfections to settle in the S2 reward design (with recorded deviations):**
   - the coxa clip term (D30, 0.018–0.026 vs 0.015);
   - the front-leg reach target (10 cm; legs settle at 11 cm);
   - lift has only the 4 cm floor after the clock anneal (×5 ≈ 0.48 / s: check with `reward_economy.py` against the S2
     reward);
   - `touchdown_reach` is not a CPG term and survives the anneal, while `cpg_coxa_contact` (whose reach weighting pays
     extension) leaves. The balance that holds the posture changes at the anneal: re-run the §2.2.6 ledger on the S2
     reward before training.
3. **Reward changes need the per-term ledger first** (`gait_by_material.py` + `reward_economy.py`). The one-sided
   `touchdown_reach` was an unforced error: the ledger for it was run on one posture only, and the removed term had been
   holding back another term's incentive (sprawl, §2.2.6).
4. **Terrain depends on the seed** (§2.2.9). Keep eval seeds fixed across arms; for exact common random numbers,
   generate the evaluation terrain once and reuse it.
5. **Operational:** `OPENBLAS_NUM_THREADS=1` for every Kit process; evals next to training need
   `load_policy.eval_physx_buffers`; resuming writes a new run dir (copy the event files for slope reviews); Play
   configs are for demos only.

**Ready for the next phase?** S1 is complete as a trunk (pending seed 46's evaluation, expected to agree), with the gates,
the V tests, the probe checks and the four correctness items (critic separation, SSL decoupling, λ_priv constant,
critic at the nominal foothold) implemented and tested (`scripts/tests/test_s1.py`). The spec's next steps
(§12 steps 5–8) come before any vision arm:

1. **N_v benchmark** (spec §6.2): the largest env count with the S3 stack (renderer, E_v, FAH, P, critic, buffer)
   within VRAM and ≥ 90 % of the best throughput, on the A4000. Blocked design point; decides T_v and the S2 / S3 batch
   for every arm, the blind ones included. Needs at least a rendering + E_v prototype.
2. **S1b** at N_v × T_v, per seed, until returns plateau.
3. **Blind O vs B0 headroom test** (S2 / S3 without rendering): O gets the M* material strips (576 zero-held columns go
   live), both lose the clock over the first 30 % of S2 with the clock-phase reward terms. If O ≈ B0 on transition
   terrain, anticipation has no value on this task and every vision run is wasted (spec §11) — the gate for all
   vision arms.
4. Only then the vision arms (B2, F; then B1-M, B1-K from S1-K; B3 optional), after the bench items that gate S2 / S3
   (Z_min, mount pitch, bus drop rate vs torque, latent_age p99, renderer pinning and temporal-artifact tests,
   visual-detectability gate).

So: ready to leave S1 once seed 46 is evaluated; the next work is infrastructure (N_v benchmark, S2 reward design with
the clock anneal, O's M* path) and the blind headroom test, not the vision arms yet.

**Review items resolved (2026-09-29):**

1. **Ground-contact penalty never faded — D33.** `non_foot_body` (femur links < 5 cm above the terrain, weight −30) was
   meant to anneal to 0 by Phase 1 step 2,000 (`LinearAnnealRewardWeight`). `HexapodFlatResumeEnvCfg` clears the
   curriculum and hand-applies the final weights of the other annealed terms, but not this one, so it has been live at
   −30 in S0a, S0b and S1 (seed 43's `params/env.yaml`: `weight: -30.0`). Effect so far ≈ 0: in every per-term
   ledger it reads 0.000 / s (−0.007 only in the sprawled one-sided run, §2.2.6), and it is identical for every arm, so
   it is not a confound. Left unchanged for the S1 trunks (no mid-seed change). Decide in the S2 reward design whether it
   stays as a permanent safety term (it guards a real failure: a femur dragging on rough terrain) or goes; either way it
   is recorded.
2. **Seed counts reconciled in the paradigm spec.** Arm table B1-K 3 → "5, first cut to 3" (§10.1–10.2), stage table
   "8 seeds + 3 S1-K" → 5 S1 seeds (the 2026-09-28 decision, D31; 8 would be needed for 8 paired B2 / F seeds) + 5 S1-K
   (3 if B1-K is cut).
3. **KL anchor marked void wherever it still read as active** (spec §0.1 technique table, §0.5 precedence row, §4 intro,
   §6.0 S1b, stage-table KL row, §8.2 routing) and in CLAUDE.md's paradigm rules. §4.4 stays as the record.
4. **Play config guarded.** `eval_s0.py` now refuses `*-Play-v0` tasks, and the S1 Play config carries a comment that
   its pilot-2 reward values are demo-only (§2.2.9).

## Phase 2.3: S1 review response — sim-only corrections before the N_v benchmark (started 2026-09-30)

An external review of the S1 results raised eight items plus spec drift. User decisions (2026-09-30): the hardware bench
is deferred until the sim-only S1 results are presented (D34); in the vision stages mirror augmentation stays, with the
colour optic on the sagittal plane and depth corruption after mirroring (D35); the blind O vs B0 headroom pilot is built
and run now at 4,096 × 16 as a non-reportable go/no-go.

### 2.3.1 Review items already resolved or stale

- Seed counts and the KL anchor were reconciled in the spec on 2026-09-29 (§2.2.12); nothing left.
- The W comparison was written up (§2.2.10) but used seeds 42–45 as the noise yardstick and was framed as a sweep
  result. Revised: yardstick = clean seeds 43–46; W 25 lies outside that range on **both** criteria in opposite
  directions (better V, worse probes), so the §9.1 rule selects no window and the seeds need no rerun; W 50 is recorded
  as deviation D32 with the reason that blind S1 metrics are biased toward short windows.

### 2.3.2 Spec drift fixed (paradigm spec v2)

Batch 8,192 → 4,096 × 16 (D1) in §1, V1-12, stage table, §6.2, §8.1, §8.3 memory note; §0.1 symmetry row (adopted,
D21/D35); dropped-reads row (load, D8); §9.1 stale command-distribution note resolved + sweep outcome; §10.2 primary
test made **one-sided** paired t-test with the seed as the unit; §10.3 sign test secondary; §12 steps 1–5 marked, torque
envelope "sim done, bench pending", bench keep/rerun rule pre-registered (step 7); §7.3 corruption after mirroring. v7
camera-mount item: colour optic on the sagittal plane.

### 2.3.3 Deviations added

| # | Deviation | Why |
| --- | --- | --- |
| D34 | Servo/bus bench deferred past S1; trunks trained on datasheet servo values | User timeline (sim-only results first). Keep/rerun rule pre-registered in spec §12 step 7; v_kin recalibrated on bench data regardless |
| D35 | Mirror augmentation kept in S2/S3: D435 colour optic on the sagittal plane, depth corruption after mirroring | A mirrored frame must correspond to a camera the robot has; the one-sided occlusion shadow must stay on the physical side |
| D36 | Latency DR 1–3 physics steps (5–15 ms) from S1b for every arm | Spec says 4–12 ms; trained 5–10 ms; trunks hold every gate at 15 ms (§2.3.5) |
| D37 | Second oracle O-geo (H*, 144-D) beside O-mat | Spec O excludes height, so it cannot test shape foresight (§2.3.6) |
| D38 | Primary behaviour metrics = transition-window task metrics + first-contact impact; stride descriptive | Under the shared reward stride shortening is not the profitable anticipatory response (§2.3.4) |

### 2.3.4 Does the reward pay for anticipation? (review item 3)

`scripts/tests/anticipation_economy.py` (new): transition course `mdp.materials.strip_course` — one 0.5 m rigid low-μ
strip across the path (μ_d = 0.88 μ_s), μ 0.9 / 0.85 elsewhere; spawn at the tile centre, yaw 0; command 0.15 m/s,
256 envs × one 10 s episode, deterministic. `walk` = the policy as is; `cautious` = scripted stride shortening (the
actor sees v_x × 0.73 from 0.3 m before the strip to its end; clock and rewards keep 0.15). Reward summed over the
episode, dt-scaled; S2 = S1 minus the four clock-phase terms. Policy: S1 seed 43 (`…_s1_w50_seed43/model_7999.pt`).

| Course | walk: falls; v_x / slip on strip (m/s) | walk reward S1 | cautious − walk, S1 / S2 | biggest term of the difference |
| --- | --- | --- | --- | --- |
| flat, μ 0.25 | 0; 0.149 / 0.015 | 504.5 | −94.8 / −92.3 | track_lin_vel −98.2 |
| flat, μ 0.10 | 0; 0.146 / 0.025 | 491.2 | −89.6 / −87.8 | track_lin_vel |
| flat, μ 0.05 | 0; 0.143 / 0.035 | 474.2 | −83.2 / −81.4 | track_lin_vel −88.9 |
| slope down 15°, μ 0.9 (control) | 0; 0.149 / 0.019 | 463.5 | −85.8 / −86.3 | track_lin_vel |
| slope down 15°, μ 0.25 | 0; **0.188** / 0.067 | 384.4 | −22.3 / −23.6 | track_lin_vel −32.9 |
| slope up 15°, μ 0.9 (control) | 0; 0.142 / 0.012 | 428.8 | −55.3 / −58.6 | track_lin_vel |
| slope up 15°, μ 0.25 | 0; **0.101** / 0.074 | 338.7 | −29.8 / −34.5 | track_lin_vel −52.3 |

Reading:

1. **On flat ground friction does not matter** for this 3.2 kg hexapod at 0.15 m/s: even μ 0.05 (below the library's
   0.2 minimum) costs the unwarned walker 30 reward units in 10 s, with no falls and 3.5 cm/s toe slip. There is nothing
   to anticipate on flat transitions; any headroom must come from slopes (and steps) × low μ.
2. **On 15° slopes a μ 0.25 strip is expensive** (−79 downhill: the robot runs away at 0.188 m/s; −90 uphill: it
   stalls at 0.101 m/s) — so a policy that knows the strip is coming *can* gain.
3. **But the gain is not in stride shortening.** Shortening cut slip only 0.067 → 0.055 m/s and still lost 22–35 units,
   almost all through `track_lin_vel` (per-step exp σ 0.05 + L1, ×30). The slip-related term
   (`stance_rigidity_contact`) moved by +0.5. `touchdown_reach` taxes it little (+2.4 in favour of cautious). The
   reviewer's risk is confirmed in a sharper form: under this reward the optimal anticipatory response is to *hold the
   commanded speed* on the strip, not to shorten the stride, so stride shortening as the primary behaviour metric
   (spec §10.3) would likely read "no effect" for every arm.
4. The per-step tracking term is the tax: slowing by 27 % for ≈ 5 s costs ≈ 90 units anywhere, while the worst
   physical outcome of not slowing costs ≤ 90 and never a fall.

### 2.3.5 Robustness, held-out layout, compliance and probe skill (review items 2 sim proxy, 5, 6, 8)

Driver `scripts/rsl_rl/eval_s1_review.sh` (all five trunks, 256 deterministic episodes per cell; JSON
`logs/eval/review_s4<N>_*.json`), new `eval_s0.py` flags `--drop_k_mult`, `--layout_seed` and the `skill_*` outputs of
`S1Aux.probe_skill()`; `scripts/rsl_rl/eval_s0b_gates.sh` takes `EXTRA="…"` for every condition.

```
scripts/rsl_rl/eval_s1_review.sh s42=<pilot3 model_7999> s43=… s44=… s45=… s46=…
python scripts/tests/gait_by_material.py --headless --split compliance --cmd {0.10,0.15} s43=<ckpt> s46=<ckpt>
```

**Latency 15 ms + 2× drop slope** (`--servo min_delay=3 max_delay=3 --drop_k_mult 2`, i.e. above the trained 5–10 ms
and at drop slopes up to 4× nominal; 83 of 85 cells — s42 `stand_L0` and `v0.15_L0` did not write output, Kit exit):
**0 falls in every cell**, every S0b gate still passes: stride error 0.10 m/s −3.2 … +0.4 %, 0.15 m/s −5.5 … −2.5 %
(nominal −2.6 … +0.1 %), worst cycle RMSE 0.0077 m/s (gate 0.03), turn on the spot +0.280 … +0.290 / −0.277 … −0.285
(nominal ≈ ±0.30), standing drift ≤ 1.5 cm. Latency costs ≈ 2–3 % of stride at 0.15 m/s and ≈ 5 % of turn rate, nothing
else. Against the pre-registered bench rule (spec §12 step 7) this is reassurance, not a substitute: if the bench reads
a latency above 10 ms the rule still says rerun — but the trunks show no cliff up to 15 ms.

**Held-out material layout** (layout seed 1 vs the training layout 0; walking 0.10 / 0.15 on levels 0, 5, 9, five
seeds, 30 cells each): indistinguishable — 0 falls both; stride −3.1 … +2.9 % vs −3.0 … +2.6 %; V slip / rigid RMSE
0.036 / 0.018 both; probe skill identical to the second decimal (below). **Nothing is memorised.** From now on every
frozen evaluation set uses layout seed ≠ 0 (training keeps 0).

**Probe skill vs the mean predictor** (R² = 1 − SSE/SST against the batch mean; contact: Brier skill; mean over the 30
cells, layout 0 → 1):

| Probe | R² | Reading |
| --- | --- | --- |
| contact (Brier skill) | 0.989 → 0.990 | |
| current μ_s, front legs / rear four legs | 0.639 → 0.628 / 0.701 → 0.698 | **the trailing-leg half exists**: z_p knows the friction under the feet that have touched it |
| current log k, rear legs | 0.566 → 0.568 | compliance is sensed too, less well |
| touchdown μ_s / log k | 0.642 → 0.644 / 0.558 → 0.557 | |
| touchdown Δt | 0.922 → 0.927 | |
| damping (per env, D25) | ≈ 0 | expected: it is constant within an episode and barely observable |

**Compliance signature at matched friction** (μ 0.8 / 0.75 both groups, rigid k 1e6 vs foam k 3.2e3; seeds 43, 46;
0.10 and 0.15 m/s): foam toes sit **1.6 mm lower** in stance (+0.01 … +0.02 → −0.14 … −0.15 cm), femur + tibia stance
tracking error rises **5–12 %** (e.g. 7.03 → 7.81°, 5.40 → 5.77°), swing lift drops 0.1–0.2 cm, speed, slip and gaps
unchanged. Consistent in sign across seeds and speeds, and the log-k probes are well above the mean predictor, so the
proprioceptive signature exists — but it is small (PhysX's effective foot stiffness ≈ 2.5 × nominal, §2.1). The
compliance half of the claim is therefore **kept as secondary**; whether foresight of it buys anything is answered by
O-mat on the compliance-only transitions of the headroom pilot (§2.3.7), and if O-mat ≈ B0 there, the headline narrows
to friction before S2.

### 2.3.6 Decisions (user, 2026-10-02)

1. **Option A — keep the reward, change the primary metric.** S2 reward = S1 reward; the clock-phase terms leave with
   the clock. The primary behaviour metrics become task metrics in a window around each transition (tracking error,
   slip, falls) plus first-contact impact on geometric transitions (the user's case: a policy that knows a bump is
   coming touches it gently; a reactive one collides first and then adapts); reaction lead time and peak GRF headline;
   stride shortening descriptive (spec §10.3 revised). **D38.**
2. **Two oracles.** The spec's O sees material only (height deliberately excluded), so it cannot test the bump case.
   O-mat (= spec O, M*) and **O-geo** (H*: terrain height at the same strip points relative to the nominal-foothold
   ground, 144-D) both branch with B0 from one trunk (spec §2.2, §2.9, §11 updated). **D37.**
3. Bench deferred (D34), mirror kept for the vision stages with the colour optic on the sagittal plane (D35) — §2.3.3.

Deviations added: D36 latency DR widened to 1–3 physics steps (5–15 ms) from S1b for every arm (the trunks hold at 15 ms,
§2.3.5; the spec's 4–12 ms range was not covered); D37 O-geo; D38 primary metrics.

### 2.3.7 The O vs B0 headroom pilot — exactly what will be run (not started; implemented in a separate session)

**Question.** Would perfect foresight of the ground ahead help a blind walker at all? O-mat answers it for material
(friction, softness — the S3 claim), O-geo for shape (bumps, steps, rough ground — the S2 claim). It is *not* a test of
latent prediction (that is F vs B2). If neither oracle beats B0, no camera-based method can either, and the vision runs
stop (spec §11).

**Arms** (one seed: S1 seed 43, a fresh run, not resumed; non-reportable go/no-go at 4,096 × 16 — the reportable
version runs later at N_v × T_v):

| Arm | Actor input on top of B0 | What it isolates |
| --- | --- | --- |
| B0 | — (M*, H* held at exactly zero) | reactive baseline |
| O-mat | M*: (μ_s, μ_d, log k, log c) at 6 legs × 24 points from −6 to +15 cm around each nominal foothold (contract P's material strips) | material foresight |
| O-geo | H*: terrain height at the same points minus the nominal-foothold ground | shape foresight |

**Training, identical for all three:** `train.py --task Hexapod-S2-v0 --branch_from <seed-43 model_7999.pt>`
(Hydra `env.arm=…`), S1 reward, S1 terrain (levels 0–14 incl. stairs, slopes ≤ 15°, bumps; materials layout 0), S1 DR,
mirror augmentation, 6,000 iterations (≈ 393 M env steps), per-episode clock dropout 0 → 1 over the first 1,800
iterations (30 %), actor frozen for 50 re-burn-in iterations at the branch, milestone review every 2,000 iterations
(gait regression watch: leg gap, lift per leg type, landing reach, coxa range, clip term). Run one at a time on tmux 0:2,
≈ 7 h each.

**Evaluation** (clock fully dropped = deployment; layout seed 1; fixed eval seeds; common random numbers across arms;
≥ 256 episodes per condition; deterministic):

| Condition | Course | Expected if foresight helps |
| --- | --- | --- |
| Material, gravity-loaded | `strip_course` μ 0.25 strip on 15° slope up / down (+ μ 0.9 control) | O-mat < B0 on tracking error and slip in the window |
| Material, flat (negative control) | μ 0.25 and foam strips on flat | O-mat ≈ B0 (§2.3.4: flat friction is harmless) |
| Compliance only | foam strip at matched μ, flat and slope | O-mat < B0 on sink-driven tracking error, if at all |
| Shape | bumps, stairs up / down, rough — training level 9 and 14, held-out 17 | O-geo < B0 on first-contact impact and stumbles |
| All | the 17 S0b gate conditions | no arm worse than B0 (gates still pass) |

Window metrics: ±0.5 m around the transition: cycle-averaged v_x error, stance slip, falls; per touchdown: peak normal
force in the first 40 ms after contact ÷ static leg load (L1, never squared), toe speed at touchdown, stumbles
(touchdowns before 60 % of the median swing duration); reaction lead time (first gait change vs the first front-leg
touchdown on the transition); stride length descriptive.

**Pre-registered go / no-go** (per oracle, against B0, episode-bootstrap 95 % CI over the evaluation episodes):
- **Go** if the oracle improves its target metric by ≥ 10 % with the CI excluding 0 on its own conditions (O-mat: slope
  × μ 0.25 tracking error or slip; O-geo: first-contact impact or stumble rate on bumps / stairs), and passes every gate.
- **No-go** for both → stop before N_v (spec §11, first row). O-geo only → keep the S2 depth arms, narrow the claim to
  geometry before S3 (spec §11, new row). O-mat on the flat negative control ≠ B0 → investigate (shortcut).

**Code state for the separate session.** Already written this session (uncommitted, S1 behaviour unchanged,
`scripts/tests/test_s1.py` passes): `env.clock_on` / `env.clock_live` (clock observation and the four clock-phase terms
share one per-episode flag); `cfg.arm` with O's M* = contract P's strips; `S1PPO` branch logic (`clock_anneal_iters`,
`reburn_iters`, M* normaliser statistics set from the first batch and unfrozen); checkpoint fields `s2_start`,
`clock_drop_p`, `arm` (read by `load_policy`); `train.py --branch_from`; `Hexapod-S2-v0` + `PPORunnerCfg_S2`
(`hexapod_s2`, anneal 1,800); `mdp.materials.strip_course`; `scripts/tests/anticipation_economy.py`. **Not run in sim
yet.** Still to do: (1) H* block `h_star` (144, appended last; mirror = strip with one value per point) and the O-geo
fill; (2) widen the 1,125-D trunk to the new layout at the branch for every arm — actor and the critic's first layer
(zero columns inserted after the actor block, before P), normaliser identity on the new dims — with a widening test
< 1e-6 in `test_s1.py`; (3) a 20-iteration branch smoke test per arm (clock_drop_p ramps, actor frozen 50 iterations,
O-mat M* statistics set, B0 / O-geo M* columns get zero gradient); (4) the headroom evaluation (window and impact
metrics above, transition courses with fixed seeds); (5) the three runs and the evaluation.

*Update 2026-10-02 (later session): items (1)–(4) are done and checked — §2.3.8–2.3.10. Arm names in code are `B0`,
`O-mat`, `O-geo` (`env.arm=…`); the earlier `O` is gone (nothing was run with it).*

### 2.3.8 Soft-ground solver check (D39 step 1) — passed at nominal k 4e2 N/m

`scripts/tests/gait_by_material.py` gained `--k` (soft-group stiffness of the compliance split) and a solver line
(deepest toe sink, stance joint speed, falls) plus per-group reward per second with the clock-free ("S2") total.
S1 seed 43 trunk, flat, 512 envs (256 rigid k 1e6 / 256 soft, both μ 0.8 / 0.75), payload as trained (0.25–1.15 kg
incl. mast), 5 ms physics step. Output: `logs/eval/soft_solver_check.txt`.

```
for K in 400 600 1000 2000; do for V in 0 0.15; do
  python scripts/tests/gait_by_material.py --headless --split compliance --k $K --cmd $V s43=<seed-43 model_7999.pt>
done; done
```

| nominal k (N/m) | standing: mean stance sink / deepest (mm) | walking 0.15: mean / deepest (mm) | v_x (m/s) | stance slip (m/s) | femur + tibia stance error (°) | falls |
| --- | --- | --- | --- | --- | --- | --- |
| rigid 1e6 | −0.3 / 0.4 | −0.1 / 0.7 | 0.149 | 0.011 | 6.8 | 0 |
| 2e3 | 1.1 / 2.7 | 2.3 / 5.2 | 0.147 | 0.013 | 7.8 | 0 |
| 1e3 | 2.4 / 4.0 | 4.1 / 8.3 | 0.147 | 0.015 | 8.5 | 0 |
| 6e2 | 3.7 / 5.7 | 6.2 / 12.7 | 0.151 | 0.018 | 9.2 | 0 |
| 4e2 | 5.0 / 8.1 | 8.5 / 17.5 | 0.153 | 0.020 | 10.1 | 0 |

Reading: the solver holds down to k 4e2 — no falls, no jitter (standing stance joint speed 0.07 rad/s soft vs 0.06 rigid,
standing v_x 0.000), no runaway penetration (deepest single-foot sink 17.5 mm against the 90 mm toe offset). Standing
sink × k ≈ 2.0–2.4 N against a static leg load of ≈ 6–7 N: the ground acts ≈ 3 × stiffer than nominal (per-contact-point
stiffness; §2.3.5 estimated 2.5 ×). **The 5–10 mm target is reached while walking at k 4e2–1e3 (4.1–8.5 mm mean) and
standing only at k 4e2.** What soft ground does to the blind trunk at k 4e2: front lift 4.3 → 3.0 cm (middle 4.9 → 4.1,
back 4.2 → 3.6), worst leg gap +0.8 → +0.3 cm (< 2 cm share 0.8 → 2.1 %), stance tracking error +49 %, slip ×1.8 — and
the commanded speed is still held. The provisional ranges stand: `soft_soil` k 4e2–1e3 (walking sink 4–8.5 mm),
`wet_clay` k 6e2–2e3 (2.3–6.2 mm).

### 2.3.9 Headroom pilot implementation (H*, widening, soft entries, evaluation) and the bugs the smoke tests found

**Design (smallest change that keeps S1 reproducible).**
- *Layout.* `Hexapod-S1-v0` keeps the 1,125-D actor. `mdp.layout.S2_ACTOR_BLOCKS` = S1 blocks ⊕ `h_star` (144, mirror
  `strip:1`); `Hexapod-S2-v0` is 1,269-D (`cfg.actor_dim`). S1 checkpoints, S1 evaluations and every script that builds
  a 1,125-D actor are untouched; `load_policy.TASK_BY_DIM[1269]` routes S2 checkpoints. `layout.ARM_BLOCK` =
  {B0: none, O-mat: `m_star`, O-geo: `h_star`} is the single place an arm's extra block is named.
- *H\*.* `terrain_z(strip_xy) − terrain_z(anchor_w)` per leg: the 24 contract-P strip points minus the terrain height at
  that leg's nominal foothold (independent of body height and swing phase). Mirror: legs swapped, lateral order reversed
  (`strip:k`, the same rule as the material strips; self-check in `mdp.symmetry`).
- *Widening at the branch.* `s1_ppo.widen_checkpoint`: zero first-layer columns + identity normaliser statistics appended
  to the actor and inserted after the actor block of the critic input (before P), the two matching Adam moment tensors
  padded with zeros, new columns added to the frozen (zero-held) set. Called in `S1PPO.load` when a 1,125-D checkpoint
  meets a 1,269-D model — every arm, B0 included. `test_s1.py` check 11: widened actor **and** critic of the seed-43
  trunk equal the trunk over 10,000 random observations with random H*, max |Δ| 2.3e-12 (float64).
- *Runtime invariant.* `S1PPO.update` asserts every iteration that the first-layer columns of the arm's zero-held blocks
  are exactly zero — an arm reading a block it is not given now stops the run instead of confounding it.
- *Soft entries.* `soft_soil` (μ 0.4–0.7, k 4e2–1e3) and `wet_clay` (μ 0.25–0.5, k 6e2–2e3, shiny) appended to
  `mdp.materials.LIBRARY`; a task uses the first `cfg.material_entries` (S1 10, S2 12). **Deviation D40:** with 12
  entries the field's random draws differ, so S2's layout seed 0 is a new layout, not the one the trunks trained on
  (accepted: §2.3.5 showed a held-out layout is indistinguishable). Probe target statistics unchanged (D39 step 3).
- *Evaluation.* `scripts/rsl_rl/eval_headroom.py` (one condition → JSON with per-episode values) and
  `scripts/rsl_rl/eval_headroom.sh` (the §2.3.7 table + soft conditions + the 17 gates, then oracle − B0 with an
  episode-bootstrap 95 % CI and the pre-registered verdict). Metrics: fall, window |v̄_x − cmd|, stance slip, first-contact
  impact (peak |F| over the first 40 ms ÷ static tripod leg load, force norm — no world-Z question on slopes), toe speed
  at touchdown, short-swing share (touchdowns after < 60 % of the median swing: stumbles and touchdown bounces), stance
  sink, base height, pitch, stride (descriptive). **Not implemented: reaction lead time and touchdown-timing error**
  (user decision: not in the go rule; added only if the pilot says go).

**Bugs (Observed / Root cause / Fix).**

1. *O-mat: critic explained variance 0.3–0.6 after the branch (B0, O-geo: 0.98), actor never released.*
   Root cause: the critic input starts with sg[actor obs], so M* going live also enters the critic. There its normaliser
   statistics are those of a constant zero (variance 0 → gain 1/ε = 100) under a count of 5.2e8 that no longer moves, and
   its first-layer columns are untrained random initialisation (they never saw a gradient). M* arrived as noise × 100.
   Fix: when a block goes live, set the critic's statistics for it from the same first batch as the actor's and zero the
   critic's columns (function-preserving: the inputs were zero). The critic loses nothing — P already holds the strips.
2. *O-mat: mean reward 42 → −62 within 14 iterations of the actor being released.* Root cause: the new first-layer
   columns have zero Adam moments but share the layer's step count (154,040), so there is no bias correction and the
   first updates are ≈ 3.2 × lr × sign(g) whatever the gradient (√(1 − β₂)/(1 − β₁) scaling of m/√v). In one iteration
   (20 steps) every M* weight reached ≈ 0.0036; M* is 576 strongly correlated unit-variance inputs, so the common mode
   shifted the hidden layer by O(1) and destroyed the gait. O-geo (144 inputs, |w| 0.003) survived the same kick.
   Fix: seed the second moment of a newly live block's columns (actor and critic) with the median second moment of the
   trained proprioceptive columns — steps become proportional to the gradient, as for every trained weight. After the
   fix O-mat follows B0 (table below). The same seeding applies to any block going live later (vision arms).
3. *`reward_economy.py` on the S1 trunk: walk at 0.088 m/s under a 0.15 command.* Root cause: the script loaded only V_0,
   not the aux learner and the ê gate, so the actor ran on an untrained estimator. Fix: `load_policy.load_env_state`.
4. *`material_field_test.py --task Hexapod-S2-v0` failed twice.* (a) Run next to a 4,096-env training: PhysX CUDA
   out-of-memory in the second env of the process, nothing moved — not a physics result; run it alone. (b) Alone: all
   52 low-μ envs slide, 6 of 221 "hold" envs drifted 2–7 cm — all six on `soft_soil`. They settle elastically: 16
   soft-footing envs drift 1.4 cm median (7.4 max) in the first 2 s and **0.0 cm median (0.4 max) in the next 2 s**. Soft
   footing is now reported separately; the firm-ground assertions are unchanged and pass.

Noted, not a bug: the first iteration after any branch shows value loss ≈ 1.4e4 and EV ≈ −12 (S1 resume: 263, −1.0) —
all 4,096 envs start an episode together and 17 % of the ground is new material; EV is back at 0.97 after 5 iterations,
inside the 50-iteration actor-frozen re-burn-in. **For S1b:** the critic's normaliser count (5.2e8) is effectively
frozen and at least one contract-P `dr` dim has variance 0 — widening the latency range (D36) will enter the critic at
gain 100 unless those statistics are reset at the S1b branch (same mechanism as bug 1).

### 2.3.10 Pre-training checks (all on the S1 seed-43 trunk; outputs in `logs/eval/pre/`)

**Reward economy** (`reward_economy.py --task Hexapod-S1-v0 --cmd 0.15 0 0`, 256 × 10 s; "S2" = minus the four
clock-phase terms, which leave with the clock):

| Behaviour under a 0.15 m/s command | reward / step, S1 | S2 | discounted return |
| --- | --- | --- | --- |
| walk (trunk, v_x 0.148, 0 falls) | +1.015 | +0.897 | +92.5 |
| stand (zero action) | +0.261 | +0.302 | +23.7 |
| fall on step 0 | — | — | −0.10 |

Walk ≫ stand ≫ fall also without the clock terms (they net +0.117 / step for walking and −0.041 for standing: removing
them narrows the walk − stand margin from 0.754 to 0.595 / step, still 3 × the standing reward). On soft ground the
walking reward barely moves: S2 total 45.7 / s rigid vs 44.7 (k 1e3) and 44.9 (k 2e3) (`soft_solver_check.txt`).

**V on soft ground** (v7 §4.3 tests on a 256-episode rollout, flat, 0.15 m/s, whole ground one material at μ 0.8 / 0.75;
`eval_headroom.py --strip 0.8 0.75 <log10 k> --base 0.8 0.75 6 --start -100 --length 1000`):

| Ground | V RMSE (m/s) | slip / rigid strata (limit < 3) | IMU-zero ratio (limit ≥ 1.5) | short-swing share | toe speed at touchdown (m/s) | impact (× leg load) |
| --- | --- | --- | --- | --- | --- | --- |
| rigid | 0.020 | 1.67 | 8.6 | 0.09 | 0.17 | 2.20 |
| k 2e3 | 0.027 | 1.32 | 6.4 | 0.17 | 0.29 | 1.75 |
| k 1e3 | 0.029 | 1.20 | 5.8 | 0.18 | 0.31 | 1.41 |
| k 6e2 | 0.031 | 1.48 | 5.5 | 0.20 | 0.31 | 1.07 |
| k 4e2 | 0.034 | 1.64 | 4.8 | 0.20 | 0.31 | 0.77 |

**V passes both tests on every soft ground**; its error grows 1.7 × at k 4e2 (never seen in training). The blind trunk
holds the speed (|v̄_x − cmd| ≤ 0.004 m/s) with 0 falls, so on *flat* soft ground there is little for foresight to win
on tracking; what changes is the touchdown (toe speed ×1.8, twice as many short swings) — the soft conditions of the
pilot are therefore read on sink-driven tracking, slip and the touchdown metrics, on slopes as well as flat.

**Material field with 12 entries** (`material_field_test.py --task Hexapod-S2-v0`): PASS — read-back exact, 52 / 52
low-μ envs slide and 205 / 205 firm high-μ envs hold at 20°, soft footing settles (bug 4).

**Merge-gating tests** (`scripts/tests/test_s1.py`): pass, incl. S2 widening 2.3e-12 and the H* mirror.

**Branch smoke tests** (`train.py --task Hexapod-S2-v0 --num_envs 4096 --seed 43 --branch_from <trunk> env.arm=<arm>
agent.algorithm.reburn_iters=5`, 20 iterations B0, 40 iterations O-mat / O-geo after the two fixes; logs
`logs/eval/pre/smoke*.log`, run dirs `logs/rsl_rl/hexapod_s2/*smoke*`):

| Check | B0 | O-mat | O-geo |
| --- | --- | --- | --- |
| branch iteration (`s2_start`), `clock_drop_p` after 20 iterations | 8,000; 0.0111 (= 20 / 1,800) | same | same |
| actor frozen for `reburn_iters`, then released | yes | yes (was stuck before fix 1) | yes |
| EV at iteration 5 / 20 after the branch | 0.97 / 0.99 | 0.97 / 0.99 (0.29 / 0.60 before fix 1) | 0.97 / 0.99 |
| mean reward at iteration 20 (episodes still filling) | 199.9 | 197.0 (−61.9 before fix 2) | 198.8 |
| mean reward at iteration 40 | — | 354.9 | 400.6 |
| live block statistics set (normaliser std) | — | M* 0.51 | H* 0.008 (floored at 0.01) |
| first-layer max \|w\| in zero-held blocks | 0 (all) | 0 (all but M*) | 0 (all but H*) |
| frozen-set size | 1,076 | 500 | 932 |

The first six iterations are identical across arms to the printed digit (0.79, 6.30, 15.28, 22.64, 32.30, 42.71): the
widened, frozen actor is the trunk in every arm. Iteration time 4.4–5.5 s → 6,000 iterations ≈ 7.5–9 h per arm. The
terrain curriculum restarts from its initial levels at the branch (levels are not checkpointed) — identical for the
three arms.

**Stop point (user decision):** runs B0 → O-mat → O-geo start after the user's OK on this section.

### 2.3.11 Fast vs slow on the new terrain before training (the §2.3.4 check for soft ground, bumps, rough, stairs)

Asked by the user before the runs: the §2.3.4 economy (trunk at full speed vs scripted stride shortening, actor sees
v_x × 0.73) had only been done for friction. `anticipation_economy.py` gained `--k` (strip stiffness), `--base_mu`
(matched friction) and `--length`; S1 seed-43 trunk, 0.15 m/s, 256 envs × one 10 s episode, deterministic. Soft strips:
0.5 m long, μ 0.8 on and off the strip (stiffness is the only change). Shape terrains at difficulty 1 (3 cm bumps and
risers, ± 2.25 cm rough): slow for the whole episode (`--start -100 --length 1000`). Output
`logs/eval/pre/anticip_soft_shape.txt`; blind-trunk baselines from `eval_headroom.py` in `logs/eval/pre/trunk_*.json`.

```
python scripts/tests/anticipation_economy.py --headless --ckpt <trunk> --terrain {flat,slope_up,slope_down} --mu 0.8 --base_mu 0.8 --k {400,1000}
python scripts/tests/anticipation_economy.py --headless --ckpt <trunk> --terrain {bumps,rough,stairs_up,stairs_down} --mu 0.9 --start -100 --length 1000
```

| Course | walk: falls; v_x / slip on strip (m/s) | walk reward S1 | cautious − walk, S1 / S2 |
| --- | --- | --- | --- |
| flat, soft k 4e2 | 0; 0.148 / 0.018 | 476.3 | −87.8 / −83.2 |
| flat, soft k 1e3 | 0; 0.146 / 0.013 | 496.7 | −99.9 / −96.2 |
| slope up 15°, soft k 4e2 | 0; **0.049 / 0.084** | **138.9** | +0.6 / −1.8 |
| slope up 15°, soft k 1e3 | 0; 0.131 / 0.019 | 414.2 | −62.1 / −66.4 |
| slope down 15°, soft k 4e2 | 1.6 %; **0.063 / 0.089** | **141.0** | +10.1 / +5.1 |
| slope down 15°, soft k 1e3 | 0; 0.144 / 0.027 | 416.5 | −56.7 / −60.8 |
| bumps | 0; 0.145 / 0.011 | 477.1 | −108.4 / −109.2 |
| rough | 0; 0.143 / 0.014 | 441.1 | −119.9 / −113.1 |
| stairs up | 0; 0.146 / 0.014 | 429.8 | −94.2 / −94.9 |
| stairs down | 0; 0.147 / 0.017 | 446.9 | −108.7 / −106.7 |

Blind trunk per condition (`eval_headroom.py`, window metrics; shape at level 14):

| Condition | falls | \|v̄_x − cmd\| (m/s) | slip (m/s) | impact (× leg load) | short-swing share | sink (mm) | stride (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat, rigid strip (control) | 0 | 0.003 | 0.011 | 2.22 | 0.11 | 0 | 0.148 |
| flat, soft k 4e2 strip | 0 | 0.004 | 0.017 | 1.56 | 0.11 | 3.8 | 0.146 |
| slope up, rigid (control) | 0 | 0.008 | 0.015 | 2.43 | 0.15 | 0 | 0.142 |
| slope up, soft k 4e2 | 0 | **0.072** | **0.066** | 2.31 | **0.50** | 12.4 | **0.067** |
| slope down, rigid (control) | 0 | 0.002 | 0.014 | 2.26 | 0.12 | 0 | 0.149 |
| slope down, soft k 4e2 | **3.1 %** | **0.059** | **0.065** | 2.00 | **0.52** | 15.9 | **0.078** |
| bumps L14 | 0 | 0.004 | 0.016 | 2.29 | 0.12 | — | 0.146 |
| rough L14 | 0 | 0.009 | 0.027 | 2.21 | 0.17 | — | 0.141 |
| stairs up L14 | 0 | 0.006 | 0.021 | 2.37 | 0.14 | — | 0.144 |
| stairs down L14 | 0 | 0.004 | 0.022 | 2.21 | 0.14 | — | 0.148 |

Reading:

1. **Flat soft ground, bumps, rough and stairs: nothing to anticipate with speed.** The blind trunk holds the command
   (error ≤ 0.009 m/s, 0 falls) and slowing costs 83–113 reward units in 10 s — the `track_lin_vel` tax of §2.3.4 again.
   As there, this says the *slow-down* response is unprofitable; it does not test the response O-geo is meant to find
   (a gentler or higher step), which has no command knob. On shape terrain the blind numbers leave room only on the
   touchdown metrics (impact ≈ 2.2–2.4 × leg load, 12–17 % short swings), not on tracking.
2. **Soft k 4e2 on a 15° slope is the one condition where the blind trunk fails:** it stalls (0.05–0.06 m/s under a 0.15
   command), slips 6 × more, half its swings are short, stride halves, 1.6–3.1 % fall downhill, and the reward drops
   from ≈ 415 to ≈ 140. Slowing is then free (+0.6 … +10 S1, −1.8 … +5 S2). At k 1e3 on the same slope the loss is small
   (414–417, 0 falls) — the cliff lies between k 4e2 and 1e3.
3. **Caveat before reading (2) as headroom.** The stall happens *on* the strip, so it shows that the trunk — never
   trained on k < 2e3 — cannot walk there, not that warning helps. All three arms train on soft ground from the branch;
   if B0 learns to cross it reactively, the oracle gains nothing. Also the sink there (12–16 mm mean in the window) is
   beyond the 5–10 mm target: on a slope the load shifts to fewer legs. Whether soft-on-slope at k 4e2 is physically
   meaningful or a compliant-contact shear artefact (§2.3.9 bug 4: elastic downhill settling) is open.

Consequences for the pilot (unchanged plan, sharper expectations): the compliance go/no-go will be decided by the
soft-on-slope cells; flat soft and the flat friction control should read oracle ≈ B0; O-geo can only win on impact and
short-swing share.

*Evaluation change (user, 2026-10-03, before any arm was evaluated):* the shape cells of `eval_headroom.sh` cover every
held-out level 15–20 as well as training levels 9 and 14 (bumps, stairs up / down, rough). The O-geo go / no-go still
uses only the pre-registered cells (bumps and stairs at levels 9, 14, 17); levels 15, 16, 18–20 are reported as a
secondary "held-out" block (impact, short-swing share, falls) and cannot turn a no-go into a go.

---

## Phase 2.4: Showcase rendering for the demo videos (2026-10-03)

Presentation work, not an experiment: nothing here is in any arm's training or evaluation path, and no number in
this document changes because of it. It is recorded because the demo videos shown alongside the paper are produced
by this code and must be reproducible from it.

### 2.4.1 Why

The videos in `logs/videos/gait/` are hard to read: terrain and background are the same value, so a viewer cannot
tell where the ground is. Cause, from the code rather than from taste:

| what | state before |
|---|---|
| renderer | never configured anywhere in the repo — no `RenderCfg`, no `rendering_mode`. AppLauncher's default `balanced`, and `apps/isaaclab.python.headless.rendering.kit` explicitly disables translucency, reflections, indirect diffuse and ambient occlusion |
| sky | none. One `DomeLightCfg(intensity=750)` with no texture + one `DistantLightCfg` pointing straight down (`hexapod_env_cfg.py:101-109`), so the background is flat grey and nothing casts a raking shadow |
| terrain | two colours in total for ten materials — `(0.25,0.25,0.3)` for `shiny`, `(0.45,0.42,0.38)` otherwise (`mdp/materials.py`), with the `# S3 replaces with MDL` note already standing there |
| robot | one `UsdPreviewSurface` bound to a 1.9 kB flat-black texture |
| resolution | 1280 × 720 at 50 fps, from the untouched `video_recorder` default |

### 2.4.2 What was added

`source/hexasim/hexasim/tasks/manager_based/hexasim/demo_look.py` — `apply(cfg, …)`, called by the demo scripts on
an **already-parsed** cfg, before `gym.make`:

- `cfg.sim.render = RenderCfg(...)`. `quality="high"` is the RT2 preset (`rendering_mode="quality"`, GI with
  `max_bounces=3`, 4 spp, DLAA, AO denoiser mode 0); `quality="mid"` is RT1 (balanced, 2 spp, DLSS quality) and is
  the one to use next to a training job.
- `cfg.video_recorder.window_width/height`. This is the only knob that works: `cfg.viewer.resolution` is deprecated
  and, on this code path, consumed by nothing — the render product is allocated from `video_recorder_cfg.py` via
  `isaacsim_kit_perspective_video.py`.
- An HDRI sky dome (`DomeLightCfg.texture_file`, keys in `demo_look.SKIES`, default `Clear/noon_grass_4k.hdr`), a
  raking sun (yaw 35°, pitch −55°, `angle=1.2` so contact shadows are not razor-sharp) and a **rim light** from low
  behind. The sky is both the background and the ambient light, which is the single change that separates terrain
  from background.
- `color_scheme="none"` on the terrain generator. Safe: `TerrainGenerator` colours the mesh *after* all geometry is
  generated, from its own `np_rng`, and nothing is drawn from that generator afterwards — the geometry is
  bit-identical. (The global numpy RNG, which the terrain layout depends on per §2.2.9, is never touched by it.)
- One MDL per material-library entry, through a new `materials.VISUAL_OVERRIDE` hook (default `None`, so training
  and eval are untouched): `create_prim_from_mesh` already forwards `visual_material`, so this is two lines in
  `materials.py`. `project_uvw=True` is mandatory — the per-entry submeshes are built from raw triangles and carry
  no UVs. Side effect worth having: the friction/compliance patchwork the policy walks over is now *visible*.
- `robot_mdl=` (default off) re-skins the robot. It has to be a spawn-time `UsdFileCfg.visual_material`, not a stage
  edit: the robot's material is bound directly on each of its 12 visual meshes **inside instance prototypes**, so it
  is not editable on the composed stage, and an ancestor binding only wins because `spawn_from_usd` binds with
  `strongerThanDescendants` (`from_files.py:405-416`). Off by default — the PhantomX is matte black, and a
  silhouette against lit ground is both legible and honest.

`scripts/rsl_rl/record_gait.py`: `--pretty --sky --quality --res --fps --robot_mdl --smooth`, two new `--view`
presets (`hero`, `orbit`), and an EMA on the chase-camera anchor (the eye was pinned to the raw root position, so
body bob and yaw shook the whole frame). `--fps` goes to `RecordVideo(fps=…)` and only changes the muxed frame
rate: native is 50 (= 1/`step_dt`), and `--fps 25` is a clean 2× slow motion. **30 fps is not reachable natively** —
control is 50 Hz and raising `render_interval` would desync render from step; transcode instead.

`scripts/rsl_rl/demo_teleop.py`: the same flags plus `--take_seconds` (auto-stop a take).

`scripts/rsl_rl/record_showcase.sh`: the runner. Cell A is the equivalence check, then six 15 s takes, then
`ffprobe` on each output and stills every 3 s for inspection.

### 2.4.3 Why this cannot be a confound

- The submeshes whose material changes are created with `CollisionAPI … CreateCollisionEnabledAttr(False)`; the
  single collider is the (hidden) source mesh. Material is geometry-free.
- DR, observation noise and dropped reads are left exactly as in training — there is deliberately no "clean" flag,
  so the demo shows the trained condition.
- `materials.VISUAL_OVERRIDE` is `None` unless `demo_look.apply` is called, which only the two demo scripts do.
- `scripts/rsl_rl/record_showcase.sh` cell A measures it. An *exact* comparison turned out to be impossible:
  PhysX on the GPU is not bit-reproducible, so two runs of the same command at the same seed already disagree
  (§2.4.6). The cell therefore runs the plain command three times to get the simulator's own envelope, runs it
  once more with `--pretty`, and `scripts/rsl_rl/equiv_check.py` checks that the pretty run falls inside that
  envelope.

### 2.4.4 Status

Code complete and checked without the GPU (imports, every `RenderCfg` field name against isaaclab 6.1.17,
`apply()` on real parsed `Hexapod-S1-Play-v0` and `Hexapod-Flat-Play-v0` cfgs, the light-quaternion helper against
its intended emission directions, the `materials.py` self-check, ruff clean on the new code). Every HDRI and MDL
URL was checked to exist on the Isaac asset root. **The renders themselves have not been run** — the GPU was busy
with training and the headroom evaluations (user, 2026-10-03). Run:

```
bash scripts/rsl_rl/record_showcase.sh                       # defaults: π_S1 seed 43 (2026-09-28_13-48-22 …/model_7999.pt), quality high, 1920x1080
bash scripts/rsl_rl/record_showcase.sh <ckpt> mid 1280x720   # alongside a training job
```

on an idle GPU, then write the equivalence result and the chosen takes in here. Note the first run streams the sky
and the MDLs from the Isaac asset root (the local Omniverse hub cache is empty), so it starts slowly once.

### 2.4.5 What the first renders changed (2026-10-03, verified in frames)

The static checks passed but the first real renders were wrong in four ways. Each was found by extracting stills
and looking at them, not from the logs.

1. **The sky looked "upside down".** It was not a texture orientation problem. The generated terrain is a small
   island, and a camera 0.45 m off the ground looks straight past its edge into the sky dome's **lower**
   hemisphere — in a latlong HDRI that is the ground around the original photographer, and seen from inside the
   dome it reads as an inverted landscape hanging above the horizon. Rotating the dome (90 / 180 / 270 were all
   tried in renders) only moved which part of it showed. Fix: `backdrop`, a visual-only ground slab (default
   400 m) under the terrain that extends the ground to the horizon. It carries no collider — `collision_props` is
   left `None`, which is what decides it (`shapes.py:296`) — so it cannot touch the robot. `sky_tilt` stays as a
   flag (default 270) for a differently-authored HDRI.
2. **Flat brown blobs over the terrain texture.** The slab's top started at z = −0.01 while rough terrain at level
   13 dips to about −0.023, so the plane cut through the height field and its intersection contours rendered as
   smooth untextured patches. Fix: `backdrop_z` (default −0.05, which clears flat, rough and bumps). Stairs and
   slopes descend far further — a 15° slope drops ≈ 0.5 m across a 4 m tile — so those takes pass −0.7.
3. **Heavily underexposed.** Dome 900 / sun 2500 / rim 600 with `dome_light_upper_lower_strategy=4` rendered dark.
   Now 1500 / 4000 / 900, full IBL (`strategy=0`) on `quality="high"` and the approximation kept only for `"mid"`,
   plus a `light_gain` multiplier for the darker HDRIs.
4. **The robot was a black hole, then white.** `diffuse_color` is linear and renders much lighter than the number
   suggests under this rig: 0.45 came out near white. The `ROBOT_SURFACE` presets are now calibrated against the
   dark grey the robot has in the existing `logs/videos/gait` takes — default `gunmetal` = (0.11, 0.115, 0.125),
   roughness 0.45, metallic 0.15.

Also from this round: the per-material patchwork reads as a patchwork on video, so `terrain_mdl` now puts one
material over the whole terrain (default `dirt`, see `demo_look.GROUND_MDL`), with `"multi"` keeping the 12-entry
version that shows the physics. `talc_graphite` no longer maps to `Metals/Steel_Carbon.mdl`, which rendered as
dark metal rather than ground. `--orbit_turns` (default 0.5) sets revolutions per take, so the orbit rate follows
`--seconds`. Every footage take except `hero_flat` and the in-place spin now orbits, and the default checkpoint is
B0 final (`logs/rsl_rl/hexapod_s2/2026-10-02_18-56-36_s2_headroom_B0_seed43/model_13998.pt`).

Verification renders for the above: `logs/videos/skychk/skychk{,2..6}-step-0.mp4`, each
`record_gait.py --headless --pretty --quality high --res 1280x720 --terrain_type rough --terrain_level 13
--cmd 0.15 --view orbit --orbit_turns 0.5 --seconds 10` on the B0 checkpoint, differing only in the fix under
test. The equivalence check (§2.4.3) has **not** been re-run since these changes.

### 2.4.6 The equivalence test had to change: the simulator is not reproducible run to run

First run of the finished script reported the gait tables differing between the plain and the `--pretty` take, so
the showcase rendering looked like it was changing behaviour. It was not.

- `record_gait.py` never set `cfg.seed`, and Isaac Lab warns about exactly this ("Seed not set for the
  environment"). Every DR, noise and dropped-read draw, and the terrain layout, were therefore different between
  any two runs. Fixed: `--seed` (default 42), set on the cfg before `gym.make`.
- With the seed fixed the tables **still** differed — and so did two runs of the *same* plain command. PhysX on
  the GPU is not bit-reproducible. Measured over three plain repeats at seed 42 (rough, level 14, 4 s, the O-mat
  checkpoint): the widest spread across the 51 numbers of the gait table is **1.7** (the units are degrees for
  the joint-angle columns), i.e. the simulator disagrees with itself by up to ~1.7° on a mean joint angle.

So the test is now relative, not exact (`scripts/rsl_rl/equiv_check.py`): three plain repeats define a per-number
envelope, and the `--pretty` run must fall inside it, with slack `max(0.5 × envelope width, 0.05)`.

Result, 2026-10-03, O-mat final at `--quality mid`:

```
plain repeats:            3
their envelope, widest:   1.7000   (the simulator disagreeing with itself)
pretty outside it, worst: 0.0010   in 1 of 51 numbers
budget = max(0.5 x envelope, 0.05) = 0.8500
EQUIVALENCE OK
```

The pretty run misses the plain envelope by 0.001 on one of 51 numbers, against a run-to-run spread of 1.7. There
is no evidence the showcase rendering changes behaviour, which is what the code structure already implies
(§2.4.3). Worth keeping in mind beyond this section: **the same nondeterminism applies to every eval script in
the repo**, so any single-run number here carries a run-to-run spread of this order.

### 2.4.7 Mid-take resets: the episode was timing out, not the robot falling

A 30 s take showed the robot snapping back to its spawn twice. Cause: `Hexapod-S2-v0` has **no Play variant**, so
`record_gait.py` runs the training cfg, whose `episode_length_s` is 10.0 (`hexapod_env_cfg.py:645`); the Play
cfgs that set 10,000 s (lines 660, 776, 989, 1079) do not cover S2. Every take longer than 10 s was therefore
cut by a `time_out` termination. Fixed in `record_gait.py`:
`cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)`. A fall (`base_contact`, root below
0.05 m) still resets, which is the part that should stay visible.

### 2.4.8 Footage recorded (2026-10-03)

`bash scripts/rsl_rl/record_showcase.sh` on the **O-mat** final checkpoint
(`logs/rsl_rl/hexapod_s2/2026-10-03_02-09-32_s2_headroom_O-mat_seed43/model_13998.pt`, chosen by the user because
it slips less). Defaults: `quality=mid`, `1920x1080`, `noon_grass`, `terrain_mdl=granite`
(`Stone/Granite_Light.mdl` — the pale speckled stone the robot stands on at the start of the earlier
`showcase_rough_L14` take, picked by the user over terrazzo / marble / gravel), 30 s takes, `orbit_turns=0.75`,
terrain level 12, command 0.12 m/s, and **no** `--robot_mdl` (the robot keeps its own material, as in that same
take). The equivalence cells are opt-in now (`EQUIV=1`), since they were run and passed above. Takes, all at
1920×1080 / 50 fps / 1,500 frames, in `logs/videos/showcase`:

| take | view | terrain |
|---|---|---|
| `hero_flat` | hero | flat |
| `orbit_rough_L12` | orbit | rough, level 12 |
| `orbit_stairs_L12` | orbit | stairs up, level 12 (`--backdrop_z -0.7`) |
| `orbit_slope_L12` | orbit | slope up, level 12 (`--backdrop_z -0.7`) |
| `spin_rough_L12` | side | rough, level 12, yaw 0.3 on the spot |

Recorded with `--uniform_friction` (§2.4.9). Tracking, command 0.12 m/s: `hero_flat` 0.119, `orbit_rough_L12`
0.120, `orbit_stairs_L12` 0.120, `orbit_slope_L12` 0.120, `spin_rough_L12` yaw-only (v_x 0.001). Frames checked:
ground texture continuous with no slab intersection, horizon clean, stair treads and the slope reading
correctly, robot legible against the pale stone. An earlier set at level 14 with `terrain_mdl=dirt` and command
0.15 is still in the same folder under the `_L14` tags; those ran under the trained material field.

### 2.4.9 Friction is not the visual material (and `--uniform_friction`, a deliberate physics change)

The granite takes prompted "does the friction change with the surface, this looks slippery". It does not, and the
two systems are worth stating plainly because the confusion is natural:

- **Visual:** one submesh per library entry under `/World/ground/materials/<entry>`, each created by
  `build_terrain_materials` with `CollisionAPI … CreateCollisionEnabledAttr(False)`. They are decoration.
- **Physics:** `MaterialField` (a Voronoi field of library entries over the terrain) × `Palette` (per-episode μ_s,
  μ_d, k draws inside each entry's range) → `FootWriter`, which writes μ and compliance onto the **tibia shapes**
  through the PhysX tensor API. The terrain collider itself is uniform (μ 1.0, multiply).

So `--terrain_mdl` cannot change how slippery the ground is, and every take so far — `multi`, `dirt`, `granite` —
had identical friction. What one visual material removes is the *legibility*: the robot still crosses `ptfe`
(μ 0.2–0.3), `wet_tile` (0.2–0.35) and `wet_clay` (0.25–0.5), but they no longer look different. In the earlier
`showcase_rough_L14` take the robot happened to start on `foam` (μ 0.7–0.9), which is why that opening looked
well planted.

For presentation footage the user chose uniform high friction, so `record_gait.py` gained
**`--uniform_friction`**: `cfg.material_entries = 1`, which restricts the field to entry 0, `concrete`
(μ 0.8–1.0, rigid). Dimensionally safe — the observation's material block is continuous values
(`hexapod_env.material4` → μ_s, μ_d, log10 k, log10(1+c)), not a one-hot over entries, so the 1,269-D S2
checkpoint loads unchanged. **But it is a real change to the physics the policy runs under, and a run made with
it is not the trained condition.** It is on in `record_showcase.sh` for the showcase takes and must never back a
robustness claim; drop the flag for a take that represents training.

**First attempt hit a latent bug and had to be abandoned.** The obvious implementation —
`cfg.material_entries = 1` — hangs: `MaterialField._sites` resamples a site's entry until it differs from its
neighbour's (`while ent and e == ent[-1]`), which cannot terminate when there is only one entry to draw. A run
sat at scene construction for 30 min at 9 % GPU before this was spotted. The loop is now guarded
(`while self.entries > 1 and …`), which is inert for every real configuration (S1 uses 10 entries, S2 uses 12)
and was checked to leave the entries-10 and entries-12 fields bit-identical. `--uniform_friction` instead
rewrites the **palette** after the env is built — every entry gets concrete's μ range and rigid k — so the field,
its seed and the observation layout are untouched and nothing in the shared training path changes.

Effect, measured over the five takes (command 0.12 m/s): tracking is 0.119–0.120 m/s everywhere, including
**slope up at level 12, which managed only 0.043 m/s under the trained material field**. That stall was
friction-limited, not a terrain-geometry limit — a result worth keeping in mind for S1b/S2, where soft and
low-μ ground on slopes is already the known weak point (§2.3.11).

### 2.4.10 Not done (deliberate)

- **Indoor/warehouse backdrop.** `Environments/Simple_Warehouse/warehouse.usd` exists on the asset root and
  `demo_look` is already shaped for it (a `scene=` branch adding one backdrop prim and an indoor HDRI). The open
  question is geometric, not visual: the 7-type × 3-row terrain strip is ~28 m across and will not sit inside a
  prebuilt interior without shrinking the layout or cutting the warehouse floor.
- **Gaussian-splatting backdrop from a phone capture.** Feasible to capture — 3DGS needs only RGB video plus COLMAP
  poses, so any iPhone works and the LiDAR on Pro models only helps metric scale. The blockers are downstream: a
  splat has no collision geometry (physics would still run on the terrain generator or a meshed proxy), and the RTX
  rasteriser does not render 3DGS natively — it needs Isaac Sim's neural-reconstruction path, unverified in this
  6.0.1 install. It would also only make the *picture* real while the physics stays as it is, so under spec §11
  ("claims must not depend on renderer fidelity") it is presentation value only.

### 2.3.12 Headroom pilot: runs, evaluation, verdict (2026-10-02 → 2026-10-03)

**Runs** (seed 43, 4,096 × 16, 6,000 iterations from the S1 seed-43 trunk at 7,999, actor frozen 50 iterations at the
branch, clock dropout 0 → 1 over iterations 8,000–9,800; ≈ 7.2–7.4 h each; logs `logs/s2_headroom_<arm>.log`):

```
python scripts/rsl_rl/train.py --task Hexapod-S2-v0 --headless --num_envs 4096 --seed 43 --max_iterations 6000 \
  --branch_from logs/rsl_rl/hexapod_s1/2026-09-28_13-48-22_s1_w50_seed43/model_7999.pt env.arm=<arm> \
  --run_name s2_headroom_<arm>_seed43
```

| Arm | Run dir (`logs/rsl_rl/hexapod_s2/`) | Final reward (last 500) | Slope per 2,000 at the end |
| --- | --- | --- | --- |
| B0 | `2026-10-02_18-56-36_s2_headroom_B0_seed43/model_13998.pt` | 350 | +0.4 % |
| O-mat | `2026-10-03_02-09-32_s2_headroom_O-mat_seed43/model_13998.pt` | 351 | +1.8 % |
| O-geo | `2026-10-03_09-32-50_s2_headroom_O-geo_seed43/model_13998.pt` | 355 | +1.1 % |

**Gait at the final milestone** (`logs/milestones/<run>/it13998_*`; lift per swing R front / mid / back, L front / mid /
back, cm):

| Arm | Lift R | Lift L | Worst leg gap |
| --- | --- | --- | --- |
| B0 | 3.89 / 5.00 / 4.52 | 3.87 / 5.09 / 4.52 | +1.7 cm |
| O-mat | 2.92 / 3.74 / 3.17 | 2.76 / 3.80 / 2.74 | +1.2 cm |
| O-geo | 2.40 / 5.15 / 4.49 | 2.35 / 4.98 / 4.67 | +1.8 cm |

The three arms converged to **different gaits under the same reward**: O-mat lifts every leg 1–1.8 cm less than B0, O-geo
its front legs 1.5 cm less. The lift drift appeared only after the clock left (iteration ≥ 9,800) and grew to the end
(O-geo front 3.3 → 2.4 cm in the last 4,000 iterations). With one seed per arm this is indistinguishable from seed noise
of the S2 optimum, and it dominates every arm difference below.

**Evaluation** (`scripts/rsl_rl/eval_headroom.sh`, 256 deterministic episodes per cell, clock dropped, layout seed 1,
eval seed 0 for every arm; 50 condition cells + 17 gate cells per arm; JSON `logs/eval/headroom_<arm>_*.json`, summary
`logs/eval/headroom_summary.txt`):

```
scripts/rsl_rl/eval_headroom.sh b0=<B0 model_13998.pt> omat=<O-mat model_13998.pt>   # tmux 0:3
scripts/rsl_rl/eval_headroom.sh ogeo=<O-geo model_13998.pt>                          # tmux 0:2, in parallel
SUMMARY_ONLY=1 scripts/rsl_rl/eval_headroom.sh b0=x omat=y ogeo=z
```

Pre-registered target cells (oracle better than B0 by the relative difference shown; 95 % episode-bootstrap CI):

| Oracle | Cell | Metric | B0 | Oracle | Oracle gain | Rule |
| --- | --- | --- | --- | --- | --- | --- |
| O-mat | 15° up, μ 0.25 strip | window \|v̄_x − cmd\| | 0.0135 | 0.0116 | +13.8 % [+6.0, +20.9] | win |
| O-mat | 15° up, μ 0.25 strip | stance slip | 0.0385 | 0.0311 | +19.2 % [+15.7, +22.6] | win |
| O-mat | 15° down, μ 0.25 strip | window \|v̄_x − cmd\| | 0.0031 | 0.0036 | −15.8 % [−31.0, −1.7] | worse |
| O-mat | 15° down, μ 0.25 strip | stance slip | 0.0279 | 0.0273 | +2.3 % [−2.8, +7.2] | — |
| O-geo | bumps, stairs up / down at L9, L14, L17 (9 cells) | first-contact impact | 2.06–2.23 | same | −1.2 … +2.7 %, every CI spans 0 | — |
| O-geo | same 9 cells | short-swing share | 0.10–0.21 | 0.13–0.26 | **−20 … −33 % (worse)** in all 9 | worse |

Pre-registered controls and gates:

| Check | Result |
| --- | --- |
| O-mat flat negative control (should equal B0) | **Not equal:** flat μ 0.25 tracking +23.3 % [+14.3, +31.1], flat μ 0.9 (no low-friction strip at all) slip +18.6 % [+10.7, +25.5] — O-mat slips less than B0 everywhere, strip or not |
| Matched-control difference-in-differences (oracle gain on the strip minus its gain on the same course at μ 0.9) | O-mat 15° up: tracking +42.7 %, slip +13.5 %. **O-geo, which sees no material at all: tracking +56.2 %, slip +24.0 %** on the same cell, and +48 % / +9.5 % downhill. The placebo oracle gains more than the material oracle |
| Gates (17 S0b conditions; S1 trunk: 0 falls) | B0: falls up to 1.6 % (v 0.15, L9), cycle RMSE ≤ 0.025. **O-mat: cycle RMSE 0.0345 at v 0.15 L9 (gate 0.03) — fails**, 0 falls. O-geo: falls up to 0.8 %, RMSE ≤ 0.024. Stride error vs the command-clock reference −10 … −31 % in all three arms: the reference assumes the clock cadence, which no longer exists once the clock is dropped, so this gate does not apply after S2 as written |

**Ablation: does the oracle use its block?** (`eval_headroom.py --ablate_oracle`: the oracle block reads its normaliser
mean, i.e. z = 0, no information, in distribution; summary `logs/eval/ablation_summary.txt`. A first try held the block at
zero, which for M* is −6 σ on log k: O-mat stood still — not an information ablation, discarded.)

| Oracle | Cell | Metric: live → uninformed (Δ, 95 % CI) |
| --- | --- | --- |
| O-mat | 15° up, μ 0.25 | tracking 0.0116 → 0.0125 (+0.0009 [−0.0004, +0.0020]); slip 0.0311 → 0.0312 (+0.0001 [−0.0010, +0.0012]) |
| O-mat | 15° down, μ 0.25 | tracking +0.0009 [+0.0004, +0.0015]; slip −0.0001 [−0.0015, +0.0012] |
| O-mat | flat μ 0.25 | tracking +0.0001, slip 0.0000 (both CIs span 0) |
| O-geo | bumps L14 | impact −0.009 [−0.073, +0.058]; short swings +0.004 [−0.010, +0.020] |
| O-geo | stairs up L14 | tracking +0.0048 [+0.0007, +0.0089]; short swings +0.025 [−0.000, +0.049] |
| O-geo | stairs down L14 | all Δ within noise |

Neither oracle uses its foresight to any measurable extent: removing the information leaves O-mat's slip unchanged to
the fourth decimal on its own target cell, and O-geo uses H* only marginally on stairs up. The first-layer column norms
agree (per input: M* 0.04, H* 0.14, against q 1.4–1.6, z_p 0.4–0.6). **Everything O-mat gains over B0 comes from its
different gait, not from M*** — which is why O-geo, blind to material, gains as much on the material cells.

**Verdict under the pre-registered rules (§2.3.7):**
- **O-mat: no-go.** The rule's two winning cells exist, but (1) the go rule requires every gate to pass and O-mat fails
  the tracking gate; (2) the flat negative control differs from B0, which the rule sends to investigation, and the
  investigation (placebo DiD, ablation) shows the gain is not foresight. The compliance claim rule (D39 step 6) is not
  met for the same reasons (soft cells: mixed signs, and O-geo "wins" 7 of 9 soft sink cells without seeing material).
- **O-geo: no-go.** No impact gain on any target cell; short-swing share worse on all 9.
- **No-go for both → spec §11 first row: stop before N_v.**

**What the pilot does and does not show.** It shows that under the shared reward, in 6,000 iterations from a blind trunk,
privileged foresight of material or shape is not used by the policy and buys nothing measurable; the arms' differences
are gait-optimum differences of the order of 20–30 % on several metrics, larger than any foresight effect. It does not
show that foresight *cannot* help: (a) §2.3.4 / §2.3.11 already showed the reward does not pay for anticipation and the
blind walker copes reactively almost everywhere — after S2 training B0 even crosses the soft-k 4e2 slopes the trunk
stalled on (|v̄_x − cmd| 0.014 m/s, 0 falls); (b) one seed per arm; (c) the oracles' extra inputs were introduced at the
branch into a converged blind policy. Before any reportable claim, the options are a decision for the user (not
pre-registered): stop the vision arms per spec §11; or redesign the task so that reacting late is expensive (conditions
where the blind walker fails, a reward term that prices impacts / slips at transitions) and repeat the pilot; or train
the oracles longer / with the block live from S1. Bench items stay deferred either way.

### 2.3.13 After the no-go: why the oracles ignored their information, and the redesign under discussion (2026-10-04)

**Status: discussion record, nothing decided or run yet.** User position (2026-10-04): the §2.3.12 ablation shows the
pilot could not test foresight at all (the policies do not read their block), so the result is "the setup gives no
incentive", not "foresight does not help"; the direction is option 2 of §2.3.12 (redesign reward and evaluation, repeat
the pilot), with the first no-go reported as is. Landing impact and actuator efficiency are deployment goals in their own
right (user), which is the justification for pricing them — not the oracle comparison.

**Measured: what the S2 reward pays for.** TensorBoard `Episode_Reward/*` of the three headroom runs, mean of the last
500 iterations (values are reward per second; total ≈ 35 / s):

| Term | B0 | O-mat | O-geo | Share of total (B0) |
| --- | --- | --- | --- | --- |
| track_lin_vel | 22.50 | 22.50 | 22.79 | 64 % |
| alive | 9.95 | 10.00 | 9.98 | 28 % |
| track_yaw | 4.86 | 4.79 | 4.94 | 14 % |
| stance_rigidity_contact | −0.559 | −0.496 | −0.519 | 1.6 % |
| swing_clearance | +0.478 | +0.431 | +0.437 | 1.4 % |
| swing_radial / touchdown_reach | −0.527 / −0.806 | −0.586 / −0.752 | −0.539 / −0.845 | 1.5 % / 2.3 % |
| base_orientation | −0.110 | −0.124 | −0.094 | 0.3 % |
| ang_vel_xy | −0.078 | −0.065 | −0.076 | 0.2 % |
| leg_proximity | −0.077 | −0.056 | −0.052 | 0.2 % |
| joint_torque_excess | −0.0012 | −0.0015 | −0.0013 | 0.003 % |

Reading: 106 % of the net reward is "hold the commanded speed and do not fall", which the blind walker already does
(§2.3.11). Everything foresight could improve — slip, body rocking, tilt, effort — is together under 2.5 % of the reward,
and landing impact is not priced at all. The oracles' lower lift (§2.3.12: 1–1.8 cm below B0) cost them only
≈ 0.045 / s of `swing_clearance`: after the clock terms leave, that floor is too weak to hold the lift.

**Three possible causes of "block not used"** (a reward change addresses only the first): (A) no incentive; (B) the block
arrived late — zero-initialised columns on a converged blind policy, 6,000 iterations; (C) no physical benefit on this
robot at 0.05–0.20 m/s. The repeat pilot must be able to tell them apart: first gate = the ablation of §2.3.12 changes the
oracle's behaviour; only then oracle vs B0.

**Proposals discussed (user) and assessment (agent); numbers above, mechanisms to be checked with
`scripts/tests/reward_economy.py` before any training:**

| # | Proposal | Assessment |
| --- | --- | --- |
| 1 | `swing_clearance` weight looks high | It is 1.4 % of the reward and too weak, not too strong (lift drift above). Needed as the guard against shuffling once an impact term exists |
| 2 | World-frame straight-line tracking reward | The actor cannot observe its world position or heading, so it cannot correct drift; use path deviation (lateral offset, heading drift vs the integrated command) as an evaluation metric first |
| 3 | Raise `stance_rigidity_contact` with a ramp | The term is toe slip + 0.1 × stance joint speed; its mean value (≈ 1.1) is far above what the measured slip (0.01–0.04 m/s per foot) can produce, so the joint-speed part appears to dominate (to verify) and a higher weight would tax the propulsion stroke. Split the term, raise the slip part only |
| 4 | Loosen `leg_proximity` | Not advised: the term is non-zero in all arms, worst gaps are already inside the 2 cm margin (+1.2 … +1.8 cm), self-collision is off in sim, and §2.2.6–2.2.8 showed longer training closes the gaps |
| 5 | Stronger, tighter `base_orientation` | Gravity-referenced tilt punishes following a 15° slope, and mean pitch differs by ≈ 3° between arms as posture. Price the disturbance instead: tilt about the local terrain plane and the rocking rate |
| 6 | Raise `ang_vel_xy` | Agreed: the cleanest "gimbal" signal (mean ≈ 0.16 rad/s now, 0.2 % of the reward) |
| 7 | Raise `joint_torque_excess` | Open: a slipping foot carries less load, so a torque price may reward slipping. Measure first (cost of transport as an evaluation metric), do not reward yet |
| 8 | Landing-impact term | Agreed. Bounded L1 on toe normal speed at touchdown (rule 3: no squared contact forces); peak-force impact stays the evaluation metric |
| 9 | Term specific to surface change | Keep the reward terrain-agnostic; put the transitions in the terrain (more, sharper, on slopes) and in the evaluation window |

**Evaluation gaps found:** `eval_headroom.py` records mean pitch only — no roll, no deviation about the mean, no rocking
rate, no path deviation, no effort. These are the metrics the redesign rests on and must be added before anything is
compared.

**Guard against tuning until the oracle wins:** the new reward is fixed from the deployment goals and the economy check
on existing checkpoints, written down with a new go / no-go rule before any oracle is trained under it; it applies to
every arm from the same stage; §2.3.12 stays in the record as the first outcome.
