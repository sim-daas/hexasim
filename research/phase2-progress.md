# Phase 2: Baseline Paradigm — Development Log

Chronological log for Phase 2 (the comparison baselines of the Shared Training Paradigm spec v2). Format as in
`hexapod-RL-tuning-guide.md`: Observed Behavior / Root Cause / Fix / Result. Every number below was produced by
the command next to it on the code of this phase; raw outputs live in `logs/` (JSON under `logs/eval/`).

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --warmstart logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt agent.run_name=s0a_seed42
```

Log `logs/rsl_rl/hexapod_s0a/2026-09-21_14-09-06_s0a_seed42/` (2,521 s). Mechanics behaved as designed: actor
frozen until iteration 300 (critic EV 0.98 by iteration 250), normaliser fold at 50, V_0 α 0 → 1 over
300–1,500, std held at ≥ 0.05, 6.2 % dropped reads.

*Observed behaviour:* full-length episodes (500 steps, 0 falls) from iteration ~250 on, but the robot does not
walk. Per-10 s reward breakdown at iteration 613: forward_velocity +0.012 (v_x ≈ 0.007 m/s), stand_still −0.97,
cpg_coxa_contact +0.03 (no stroke), cpg_z_target −16.6 (no leg lift). Final evaluation
(`eval_s0.py … model_2999.pt`, 256 episodes): v_x 0.003 m/s, clearance 6 mm, fall 0 %, V_0 RMSE 0.024 m/s.
The frozen actor (= π_base, folded) shows the same standing behaviour, so PPO at LR 1e-4 / std ≈ 0.07 never
escaped a standing optimum it started in. **S0a gates: not met (no gait).**

### 2.0.8 Diagnosis — π_base's gait does not survive explicit actuation

All tests drive `Hexapod-S0a-v0` with **π_base's own actions computed from ground-truth observations**
(`scripts/tests/diag_obs_equivalence.py`, 64 envs, mean v_x over steps 25–250). That separates physics from the
observation pipeline. Control: the same loop in `Hexapod-Flat-v0` → 0.177 m/s.

| Configuration inside Hexapod-S0a-v0 | v_x (m/s) |
| --- | --- |
| AX-12A datasheet model (as trained) | 0.003 |
| AX-12A, stall 3 N·m, compliance slope 4 (stiff), no backlash/delay/punch; no clip, no payload, old reset pose | 0.012 |
| … implicit-PD-equivalent explicit servo (Kp ≈ 20, back-EMF ≈ Kd 0.49), armature 0 / 0.0003 / 0.001 / 0.003 | ≤ 0.028 |
| … same, no-load speed 12.4 or 100 rad/s (torque–speed falloff removed) | 0.028 |
| … ground-truth clock (α = 0) | 0.029 |
| Isaac Lab **explicit** `IdealPDActuator` Kp 20 / 40 / 80 / 160 (Kd = Kp/40), 3 N·m, armature 0.001 | 0.025–0.027 |
| … Kp 20 at 1 kHz physics (control 50 Hz) | 0.005 |
| … Kp 20, joint friction 0 | 0.030 |
| PhysX **implicit** drive Kp 20 / Kd 0.5 / 3 N·m (π_base's actuator), old reset, no clip/payload | **0.178** |
| … new reset pose (tibia +30°) | 0.087 |
| PhysX **implicit** drive at AX-12A-equivalent gains Kp 9.2 / Kd 0.24 / 1.5 N·m, old reset, no clip/payload | **0.190** |
| … with the S0a reset pose, action clip and payload | 0.031 |

Reading:

1. π_base walks under an *implicit* drive even at AX-12A-like softness and 1.5 N·m. Softness and torque limit are
   not what break it.
2. It does not walk under *any* explicit joint-effort actuator tested — Isaac Lab's own IdealPD included — for any
   gain (20–160), physics rate (200 Hz, 1 kHz), armature or joint friction. The result is insensitive to all of
   these, which points to the explicit-effort path itself rather than to servo parameters. **Root cause not yet
   identified.**
3. Unresolved inconsistency: lever-arm torques say the Phase 1 pose needs 2.0 N·m in tripod stance, yet the
   1.5 N·m implicit drive walks there. A candidate is the USD's `physxJoint:jointFriction = 0.5` carrying static
   load, which would itself be a fidelity problem (the real servo's friction is part of its measured behaviour,
   not a free holding torque).
4. The S0a reset pose, action clip and payload each also reduce π_base's gait (0.19 → 0.03 together); individual
   contributions not yet separated.

User decision (2026-09-21): find the cause before training again; switching to the built-in PhysX position
drive is acceptable, with domain randomisation covering the AX-12A detail that would be lost.

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --warmstart logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt agent.run_name=s0a_seed42_r2
```

*Observed:* while the actor was frozen (= π_base), the robot walked (forward reward 0.14 per 10 s, coxa stroke
present) with 57–59 % falls. From the unfreeze at iteration 300 the policy learned to fall: falls 57 % → 93 %
by iteration 340, 99.7 % by 540; episode length 322 → 64 steps. V_0's α was still ≈ 0.03, so the ramp was not
involved. Stopped at iteration 700 (`logs/train_s0a_seed42_r2.txt`).

*Root cause — reward economics.* Per-10 s episode terms (Isaac Lab `Episode_Reward/*`):

| Term | π_base, final Phase 1 training (tfevents) | S0a run 2, iteration 250 (frozen π_base) |
| --- | --- | --- |
| cpg_coxa_contact | **+8.81** (abs q̇) | **+0.68** (signed) |
| cpg_z_target | −4.66 (tibia origin) | −14.55 (toe point, AX-12A) |
| cpg_stance_rigidity | −1.50 | −2.45 |
| cpg_force_target | −2.02 (target 4 N) | −1.64 (target weight/3) |
| forward_velocity | +0.27 | +0.14 |
| alive | +1.00 | +0.62 (episodes end early) |
| **Total** | **+14.1** | ≈ −0.575 per step (−166 per 289-step episode) |

π_base's positive return came almost entirely from the `abs(q̇)` coxa term: under the signed version the same gait
earns 0.68 instead of 8.81, i.e. ~92 % of π_base's coxa reward paid for back-and-forth coxa motion, not net
stroke. With that gone and the toe-point swing error ~3× larger, every surviving behaviour has a negative per-step
reward while a fall costs −5 once. Ending the episode is the cheapest option, and PPO found it within 40
iterations. (Run 1's standing optimum was the same economics under the solver artefact.)

*Status:* reward rebalancing needs a decision (CLAUDE.md rule 2) before run 3.

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --warmstart logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt agent.run_name=s0a_seed42_r3
```
Code: commit `fe78300` + the diff stored in the run's `git/hexasim.diff`. Run 3 used σ = 0.02 m, the value at the time.
On the current code, reproduce it with `'env.rewards.cpg_z_target.params.sigma=0.02'` appended.

Training (`logs/train_s0a_seed42_r3.txt`): frozen-actor falls 57 % → after the unfreeze 23 % (it 400) → 3.4 % (600)
→ 0 % from 800 on. The V_0 ramp to α = 1 (it 1,500) caused no regression. Forward reward 0.15 → 0.40 per 10 s.
Final per-10 s terms: coxa stroke (signed) +7.2 (vs +0.68 for frozen π_base, i.e. a genuine stroke now), toe-swing
kernel +4.2 of 5, stance rigidity −2.0, force −1.8.

Evaluation (`python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0a-v0 --ckpt <run3>/model_2999.pt`,
256 episodes, deterministic, V_0 only, AX-12A, drops, payload):

| Metric | π_base, Phase 1 code | π_base, fixed Flat env | **π_S0a (run 3)** | S0a gate | Verdict |
| --- | --- | --- | --- | --- | --- |
| Fall rate | 0.0 % | 9.0 % | **0.0 %** | ≤ π_base + 2 pp | PASS |
| Contact duty factor | 0.648 | 0.627 | **0.513** | 0.5 ± 10 % | PASS (π_base itself fails) |
| Toe clearance (mean swing apex) | 8.2 cm | 4.9 cm | **2.9 cm** | ± 10 % of π_base | **FAIL** (−41 % vs the fixed-env π_base) |
| Coxa power > 5 Hz (rad²) | 0.183 | 0.158 | **0.143** | not increased | PASS |
| v_x | 0.159 m/s | 0.181 m/s | 0.262 m/s | — | — |
| V_0 RMSE | — | — | 0.037 m/s | — | — |

Without payload: 0 % falls, v_x 0.273, duty 0.504, clearance 2.9 cm (`s0a_r3_it2999_nopayload.json`).

Behaviour check (per-leg contact duty, RF RM RB LF LM LB): π_base 0.53 / **0.99** / 0.82 / 0.52 / 0.49 / 0.53 — its
right-middle leg was permanently planted (a "crutch" the Phase 1 metrics did not flag). π_S0a 0.48 / **0.79** /
0.36 / 0.50 / 0.48 / 0.47, with ~2.4 touchdowns/s on RM against ~1.4 on the other legs. The inherited RM
asymmetry is reduced, not removed.

Open: (1) clearance gate — the kernel (σ 2 cm) pays well for low swings; (2) RM stutter / RB under-loading;
(3) one seed only.

### 2.0.13 Clearance metric corrected; the remaining imbalance is the right-middle leg

*Observed:* the gate table in 2.0.12 reported 2.9 cm clearance, but the reward-economics tool (swings with apex
> 5 mm) put π_S0a's toe apex at 3.7 / 3.9 / 4.1 cm (q25 / median / q75), on the 3.8 cm target. The user's visual
inspection also judged the clearance good. *Root cause:* `eval_s0.py` averaged the apex of every lift-off,
including sub-5 mm stutter micro-lifts. *Fix:* `eval_s0.py` now counts a swing only if its apex is > 5 mm, reports
quartiles, and counts micro-lifts per leg separately. Re-scored (`logs/eval/*_v2.json`):

| | Real swings/s (RF RM RB LF LM LB) | Micro-lifts/s (RF RM RB LF LM LB) | Toe apex q25 / 50 / 75 |
| --- | --- | --- | --- |
| π_base, Phase 1 code | 0.81, **0.10, 0.11**, 0.82, 0.81, 0.82 | 0.09, 0.05, 0.07, 0.01, 0.00, 0.05 | 5.4 / 8.4 / 11.2 cm |
| π_base, fixed Flat env | 1.12, **0.11, 0.13**, 1.11, 1.09, 1.11 | 1.20, 0.79, 0.25, 0.02, 0.07, 0.06 | 4.8 / 6.4 / 9.7 cm |
| π_S0a run 3 | 1.36, **0.52**, 1.35, 1.35, 1.35, 1.35 | 0.05, **1.85**, 0.07, 0.03, 0.06, 0.05 | 3.7 / 3.9 / 4.1 cm |

Reading:
* π_base barely stepped with its right-middle and right-rear legs (0.10 and 0.11 swings/s against ~0.8): two crutch
  legs. This is the leg imbalance seen since Phase 1. S0a run 3 fixed the right-rear leg completely; the
  right-middle leg now mostly stutters (1.85 micro-lifts/s, 0.52 real swings/s).
* Clearance gate, reinterpreted with the user (2026-09-21): π_S0a tracks the 3.8 cm toe target tightly. π_base's
  6–8 cm apex was high-stepping produced by its tibia-origin target, so "± 10 % of π_base" does not measure what
  the gate intends. Clearance is judged adequate; no change to the target height.

*Next (run 4):* σ 2 → 1 cm, continued from run 3's final checkpoint. Economics check (`logs/reward_economy_r3.txt`,
run-3 actor as the walker): σ 2 cm walk +33.3 / stand +2.42; σ 1 cm walk +32.0 / stand +0.65; fall −0.1. The
tighter kernel costs a well-tracking gait ~10 % of its swing credit while stutter micro-lifts earn ≈ nothing,
which puts the pressure on the right-middle leg.

```
python scripts/rsl_rl/train.py --task Hexapod-S0a-v0 --headless --num_envs 4096 --seed 42 --max_iterations 1500 \
  --resume --load_run 2026-09-21_15-39-59_s0a_seed42_r3 --checkpoint model_2999.pt \
  --run_name s0a_seed42_r4_from_r3 'env.v0_ramp_steps=[0,1]'
```
(A first attempt passed `agent.resume=True` as a Hydra override. `cli_args` overwrites it with `--resume`'s
default False, so it silently started from a random actor; it was stopped after a few iterations and deleted.)
(α pinned to 1: the ramp is keyed to the env step counter, which restarts at 0 in a new process.)

### 2.0.14 S0a run 4 (σ 1 cm, continued from run 3) — leg imbalance resolved; π_S0a

Log `logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/` (iterations 3,000 → 4,498, 1,194 s,
`logs/train_s0a_seed42_r4.txt`). No falls after the first iteration; forward reward 0.45 → 0.49 per 10 s; swing
kernel 3.95 → 4.14 of 5 at the tighter σ.

`python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0a-v0 --ckpt <run4>/model_4498.pt` (256 episodes,
deterministic, V_0 only, AX-12A, drops, payload):

| Metric | π_base (Phase 1 code) | Run 3 | **Run 4 = π_S0a** | Gate |
| --- | --- | --- | --- | --- |
| Fall rate | 0 % | 0 % | **0 %** | PASS |
| Contact duty (mean) | 0.648 | 0.513 | **0.487** | PASS (0.5 ± 10 %) |
| Duty per leg RF RM RB LF LM LB | 0.53 0.99 0.82 0.52 0.49 0.53 | 0.48 0.79 0.36 0.50 0.48 0.47 | **0.48 0.54 0.43 0.50 0.49 0.47** | — |
| Real swings/s per leg | 0.81 0.10 0.11 0.82 0.81 0.82 | 1.36 0.52 1.35 1.35 1.35 1.35 | **1.49 1.50 1.48 1.49 1.49 1.49** | — |
| Micro-lifts/s, RM | 0.05 | 1.85 | **0.06** | — |
| Toe apex q25 / 50 / 75 | 5.4 / 8.4 / 11.2 cm | 3.7 / 3.9 / 4.1 cm | **3.7 / 3.8 / 3.9 cm** | Adequate (reinterpreted, 2.0.13) |
| Coxa power > 5 Hz (rad²) | 0.183 | 0.143 | **0.134** | PASS |
| v_x | 0.159 m/s | 0.262 m/s | **0.291 m/s** | — |
| V_0 RMSE | — | 0.037 m/s | **0.031 m/s** | — |

Result: every leg now steps at the same rate. The right-middle and right-rear crutch legs, present since Phase 1,
are gone. **π_S0a = `…/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt`** (actor 114-D + V_0 in the
checkpoint). S0a gates: all met, with the clearance gate reinterpreted as agreed with the user.

Caveats for the paper: one seed; the warm start came through three reward/solver fixes (2.0.9–2.0.13), so π_S0a
is the product of runs 3 + 4 (4,498 iterations × 4,096 × 16 ≈ 295 M env steps after the warm start, run-1/2
iterations not counted); servo, drop and IMU parameters are provisional (datasheet); flat ground at μ 0.75.
Not in S0a by spec: commands, terrain, full dynamics randomisation (S0b onward).

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --warmstart logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt --run_name s0b_seed42_r1
```

1.76 s/iteration (4,096 envs, `replicate_physics=False`). During the frozen-actor burn-in the curriculum already
moved π_S0a to mean terrain level 6.4 at 0.85 % falls (the "up" rule counts overspeed as progress).
Finished at 3,000 iterations (5,242 s): `logs/rsl_rl/hexapod_s0b/2026-09-21_19-59-42_s0b_seed42_r1/`.

Gate evaluation, `scripts/rsl_rl/eval_s0b_gates.sh <ckpt> <tag>` (256 deterministic episodes per condition, JSON in
`logs/eval/s0b_r1_it*_*.json`; L = terrain level):

| Condition | Iter | Falls | v_x | Cycle RMSE | Step RMSE | Stride (ref) | Duty | Apex q50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10, L0 | 1500 | 0 % | 0.134 | 0.035 | 0.058 | 0.167 (0.125) | 0.49 | 4.1 cm |
| 0.10, L0 | 2999 | 0 % | 0.119 | **0.019** | 0.040 | **0.148 (+18 %)** | 0.49 | 4.0 cm |
| 0.10, L5 | 2999 | 0 % | 0.117 | 0.025 | 0.055 | 0.148 | 0.48 | |
| 0.10, L0, no drops | 2999 | 0 % | 0.120 | 0.020 | 0.040 | 0.149 | 0.49 | |
| 0.15, L0 | 2999 | 0 % | 0.145 | **0.008** | 0.028 | **0.180 (−4 %)** | 0.49 | 4.0 cm |
| 0.15, L5 | 2999 | 0 % | 0.145 | 0.015 | 0.039 | 0.181 | 0.49 | |
| 0.15, L0, no drops | 2999 | 0 % | 0.145 | 0.007 | 0.028 | 0.180 | 0.49 | |
| stand, L0 | 2999 | 0 % | 0.005 | — | 0.030 | drift 5.7 cm, Σ\|q̇\| 6.8 rad/s | 0.51 | |

(0.15 m/s is below the 0.16 m/s knee of the clock, so f_ref = 0.8 Hz at both evaluation speeds; all legs
swing 0.85–0.86 /s.) Gates: falls ✓, cycle-RMSE tracking ✓ (per-step RMSE, which includes the within-stride
velocity oscillation, would fail at 0.10), drops vs drop-free ✓ (drops 0.0191 vs 0.0198 — no degradation at the
provisional rate), stride ✓ at 0.15 and ✗ at 0.10 (+18 %, i.e. the residual 0.019 m/s overspeed). Per-leg stride
is asymmetric (right 0.12–0.13 m, left 0.17 m at 0.10) and right-back duty is 0.39: the old right-side imbalance
is partly back.

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-21_19-59-42_s0b_seed42_r1 --checkpoint model_2999.pt --run_name s0b_seed42_r2_from_r1
```

Finished at iteration 4,998 (3,468 s): `logs/rsl_rl/hexapod_s0b/2026-09-21_21-36-27_s0b_seed42_r2_from_r1/model_4998.pt`.
`scripts/rsl_rl/eval_s0b_gates.sh <ckpt> s0b_r2_it4998` (`logs/eval_s0b_r2_it4998.txt`):

| Condition | Falls | v_x | Cycle RMSE | Step RMSE | Stride (ref) | Duty |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10, L0 | 0 % | 0.117 | 0.017 | 0.031 | 0.145 (0.125, **+16 %**) | 0.49 |
| 0.10, L5 | 0 % | 0.117 | 0.021 | 0.042 | 0.147 | 0.49 |
| 0.10, L0, no drops | 0 % | 0.117 | 0.017 | 0.031 | 0.145 | 0.49 |
| 0.15, L0 | 0 % | 0.148 | **0.005** | 0.025 | **0.187 (0.1875)** | 0.49 |
| 0.15, L5 | 0 % | 0.149 | 0.011 | 0.033 | 0.188 | 0.49 |
| 0.15, L0, no drops | 0 % | 0.148 | 0.005 | 0.025 | 0.187 | 0.49 |
| stand, L0 / L5 | 0 % | 0.004 | — | 0.027 | drift 5.2 cm, Σ\|q̇\| 4.8 rad/s | 0.56 |

Gates: falls ✓, tracking ✓ (cycle RMSE ≤ 0.021 everywhere), drops ✓ (no degradation at the provisional rate),
stride ✓ at 0.15, **✗ at 0.10** — v_x plateaued (0.119 → 0.117 over 2,000 more iterations), so the "still
leaving π_S0a's stride" reading of §2.1.5 was wrong for 0.10 m/s: there is an equilibrium at ≈ 0.145 m stride
whose economic cause is not yet identified. Left/right stride asymmetry is gone (0.139–0.154 m); right-back
contact duty stays 0.41.

Standing improved but is not a clean six-leg stand: legs rarely swing (≈ 0.1 /s) but contact flickers (per-leg
duty 0.33–0.73, micro-lifts up to 0.6 /s at the 1 N contact threshold) — the weight sits on 3–4 legs while the
rest barely touch — with 0.27 rad/s per-joint jitter and 5 cm drift per 10 s.

**Status: π_S0 not yet declared.** Open: (1) the 0.10 m/s stride equilibrium (+16 %) — diagnose with a per-term
reward comparison at 0.10 vs 0.117 m/s before changing anything; (2) standing load sharing; (3) the from-scratch
control arm (§4.2), which needs the final S0b budget; (4) the +5 harder terrain levels. V_0 RMSE under DR is
0.047 m/s (S0a flat: 0.031).

**Deviations added in S0b** (spec text unchanged):

| # | Deviation | Where |
| --- | --- | --- |
| D13 | Standing mode at zero command (clock frozen, CPG terms masked, standing terms) — spec silent | 2.1.1, 2.1.5 |
| D14 | Per-env contact physics on the foot material (terrain μ 1, multiply), not a sampled ground μ: one terrain mesh carries one material. μ_eff equals the sampled value (2.1.3) | 2.1.1 |
| D15 | twist_cmd resampled every 5 s, not per episode | Decisions |
| D16 | Tracking gate on gait-cycle-averaged v_x (per-step RMSE includes the within-stride oscillation; reported alongside) | 2.1.3 |
| D17 | Full DR brought into S0b (spec does not fix its stage) | Decisions |

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-21_21-36-27_s0b_seed42_r2_from_r1 --checkpoint model_4998.pt --run_name s0b_seed42_r3_from_r2
```

Finished at iteration 6,997 (3,533 s): `logs/rsl_rl/hexapod_s0b/2026-09-21_23-04-11_s0b_seed42_r3_from_r2/model_6997.pt`
(`logs/eval_s0b_r3_final.txt`). **Result: worse.** 0.10 m/s → v_x 0.128 (stride 0.158, +27 %; run 2: 0.117),
per-step error 0.059 (run 2: 0.031) — the within-stride surge grew once only its average was scored. 0.15 m/s:
0.146, stride −4 %. Standing: contact duty 0.56 → 0.63, but Σ|q̇| 4.8 → 5.6 rad/s, drift 4.9 cm. Falls 0 %.

### 2.1.9 The real problem: the policy barely responds to slow commands

**Observed.** Same sweep on the run 3 policy (`logs/s0b_stride_diag_v3.txt`): shown commands 0.03 / 0.05 / 0.07 /
0.085 / 0.10 → walked 0.106 / 0.110 / 0.117 / 0.123 / 0.128 m/s. Gate eval at commands 0.05 / 0.10 / 0.15 →
0.110 / 0.128 / 0.146 m/s: a slope of ≈ 0.36 instead of 1. The policy has one gait near 0.11–0.15 m/s and only
nudges it; §2.1.7's "reward optimum at 0.11" was partly this — the sweep could not drive it below 0.106.

**Ruled out: V_0.** `eval_s0.py --gt_velocity` (actor sees true velocity) gives the same speeds (0.110 / 0.128 /
0.146), and V_0's mean bias is |≤ 0.003| m/s at all three commands (`logs/eval/s0b_r3_{v0,gt}diag_*.json`).

**Root cause (working hypothesis).** The twist_cmd columns start at zero (warm start) and must be learned against
terms that do not depend on the command and are several times larger per step (alive 10, cpg_z_target ≤ 5,
cpg_coxa_contact ≈ 5, against tracking ≤ 10 at ×10). The economic gain from slowing to a 0.05 m/s command is real
(≈ 0.15/step dt-scaled from tracking against ≈ 0.03 lost on the coxa term), but it is a small share of the
advantage variance PPO sees. Below the 0.16 m/s knee the clock gives no speed cue, so stride length is the only
speed variable and only the command inputs can set it.

**Fix (run 4).** Back to per-step tracking (the averaged form is deleted, §2.1.8 result), weight ×3 (exp σ 0.05
and L1, ×30) so tracking dominates, as is usual for command-tracking locomotion. Branches from run 2 (the better
tracker); standing terms of §2.1.7 kept.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-21_21-36-27_s0b_seed42_r2_from_r1 --checkpoint model_4998.pt --run_name s0b_seed42_r4_from_r2
```

After 1,000 iterations (model_6000, 128 envs, L0): commands 0.05 / 0.10 / 0.15 → 0.066 / 0.1055 / 0.146 m/s
(run 3: 0.110 / 0.128 / 0.146). Finished at 6,997 (3,675 s):
`logs/rsl_rl/hexapod_s0b/2026-09-22_00-18-12_s0b_seed42_r4_from_r2/model_6997.pt` (`logs/eval_s0b_r4_final.txt`):

| Condition | Falls | v_x | Cycle RMSE | Step RMSE | Stride (ref) |
| --- | --- | --- | --- | --- | --- |
| 0.05, L0 (not a gate) | 0 % | 0.062 | — | — | 0.079 (0.0625) |
| 0.10, L0 | 0 % | 0.104 | 0.005 | 0.020 | 0.130 (0.125, **+4 %**) |
| 0.10, L5 | 0 % | 0.105 | 0.009 | 0.027 | 0.132 (+5 %) |
| 0.10, L0, no drops | 0 % | 0.104 | 0.005 | 0.020 | 0.130 |
| 0.15, L0 | 0 % | 0.148 | 0.005 | 0.020 | 0.188 (0.1875, **0 %**) |
| 0.15, L5 | 0 % | 0.148 | 0.008 | 0.026 | 0.187 |
| 0.15, L0, no drops | 0 % | 0.148 | 0.004 | 0.020 | 0.189 |

**All four walking gates pass** at both terrain levels, and the per-step error also fell below 0.03 (so D16's
cycle-averaging no longer decides the gate). Standing is still not clean: per-leg contact duty 0.40–0.78, Σ|q̇|
4.6 rad/s (rest pose, zero action: ≈ 0.5), drift 4.6 cm per 10 s — §2.1.10.

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_00-18-12_s0b_seed42_r4_from_r2 --checkpoint model_6997.pt --run_name s0b_seed42_r5_from_r4
```

At iteration 8,000 (128 envs, L0): standing contact duty 0.40–0.78 → **0.85–0.97 on five legs**, Σ|q̇| 3.2 rad/s,
drift 4.1 cm; 0.10 m/s walking unchanged (0.102, stride +3 %). But the right-back leg stayed up (contact 11 %).
Run 5 was stopped at iteration 8,142 (`…/2026-09-22_01-30-32_s0b_seed42_r5_from_r4/model_8100.pt`) for the next fix.

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_01-30-32_s0b_seed42_r5_from_r4 --checkpoint model_8100.pt --run_name s0b_seed42_r6_from_r5
```

At 9,100 (128 envs, L0): 0.10 m/s → 0.0995 m/s, stride 0.126 (+1 %); standing contact 0.87–0.98 on five legs,
right-back 0.04; Σ|q̇| 3.1, drift 3.2 cm. The clip excess fell only 2.4 → 2.1 (per second), and the requested
right-back coxa target is −69° (q10–q90 −72° to −64°), 14° past the stop, with femur −21.5° and tibia +9°
(`logs/s0b_stand_load3.txt`): the leg is deliberately **tucked** (raised as well as swung back), not merely stuck
past the clip. The coxa sign table was re-checked against the layout probe (every right leg's +coxa moves the
toe forward, `logs/probe_layout.txt`) — not a sign bug. Run 6 stopped at 9,201
(`…/2026-09-22_02-07-00_s0b_seed42_r6_from_r5/model_9200.pt`); the clip penalty is kept.

### 2.1.14 Standing pose term; S0b run 7

The tucked pose is a local optimum: rest-pose standing pays ≈ 0.19/step more, but no small change of the tucked
leg improves the load-share / contact terms until the foot actually reaches the ground. Standard remedy: a
standing-only `standing_pose` = Σ|q − q_rest| (rad), weight −1, a dense per-joint pull home (tucked right-back
≈ 1.5 rad → −1.5/step unscaled). On slopes the few-degree adjustments it penalises are small against the
load-share and contact terms.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_02-07-00_s0b_seed42_r6_from_r5 --checkpoint model_9200.pt --run_name s0b_seed42_r7_from_r6
```

At 10,200 (128 envs, L0): right-back standing contact 0.01 — unchanged; 0.10 m/s 0.102 (stride +4 %). Over run 7 the
standing pose penalty per standing step stayed at ≈ −3.6 to −4.0 and the clip excess at ≈ 1.8–1.9 per second
(mostly walking envs): no trend. The right-back habit is inherited — π_base already had right-side "crutch" legs
(2.0.14) — and survives every warm start. Run 7 stopped at 10,257 (`…/2026-09-22_02-41-40_s0b_seed42_r7_from_r6/
model_10250.pt`).

### 2.1.15 S0b run 8: stronger pulls (clip ×10, standing pose ×3)

`action_beyond_clip` −1 → −10 (still zero for every in-range target), `standing_pose` −1 → −3.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_02-41-40_s0b_seed42_r7_from_r6 --checkpoint model_10250.pt --run_name s0b_seed42_r8_from_r7
```

At 11,250: clip excess 1.8 → ≈ 0.28 per second; 0.10 m/s 0.103 (stride +4 %); standing right-back contact still
0.02. Run 8 stopped at 11,400 (`…/2026-09-22_03-15-45_s0b_seed42_r8_from_r7/model_11400.pt`).

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

Temporary schedule change (D18): zero-command segments 10 % → 50 %, then back to the spec's 10 % for a final
segment so π_S0 finishes on the spec distribution.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 1000 \
  --resume --load_run 2026-09-22_03-15-45_s0b_seed42_r8_from_r7 --checkpoint model_11400.pt \
  --run_name s0b_seed42_r9_from_r8_stand50 env.commands.base_velocity.rel_standing_envs=0.5
```

Run 9 at 11,900 (128 envs, L0): standing unchanged (right-back contact 0.05, same contorted pose), walking fine
(0.0994 m/s, stride 0.125). Five times the standing data did not move it either. Finished at 12,399
(`…/2026-09-22_03-54-55_s0b_seed42_r9_from_r8_stand50/model_12399.pt`).

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

Robot numbers: body 25 cm long, 13–20 cm wide; coxa 5.3 cm, femur 6.2 cm, tibia ≈ 13 cm (URDF). Base origin
≈ 0.17 m above the terrain when standing and the base bottom plate 3.3 cm below it → **belly clearance ≈ 13–14 cm**,
not a binding limit. The binding limit is **swing height ≈ 3.8 cm** (`cpg_z_target` step height; measured apex
3.7–4.0 cm): taller obstacles need more lift than the gait reward pays for. Slopes: servo torque is not binding
(femur ≈ 12 % of stall standing); friction is — at the lowest sampled μ 0.3 anything above ≈ 17° slides whatever
the policy does.

Proposal: gates also at L9 (hardest built level, untested so far; the curriculum's mean level sits near 6); then
levels 10–14 as already laid out in `S0B_TERRAIN_CFG` (rough ±2.25 cm, slopes 15°, bumps 3 cm) plus pyramid
stairs with 1.5 → 3 cm risers, capped there: beyond it needs a higher step-height target, and harder geometry
makes blind arms fail for geometric reasons, blurring the friction/compliance comparison the paper is about
(v7 §5.4: geometry ⟂ physics). Gates required on L0/L5/L9, L14 reported as robustness. Pending the user's decision.

L9 gate check, run 4 policy (`…r4_from_r2/model_6997.pt`, 256 episodes, `logs/eval/s0b_r4_final_v*_L9.json`):
0.10 m/s → 0 % falls, v_x 0.104, cycle RMSE 0.014, stride 0.132 (+6 %); 0.15 m/s → 0 % falls, 0.146, 0.014,
0.185 (−1 %). All walking gates pass on the hardest built level too — there is headroom for levels 10–14.

Run 10 at 13,150 (128 envs, L0) — first standing fix that moves it: right-back contact 0.05 → **0.57**, all-leg
contact duty 0.78 → 0.86 (0.90 0.98 0.57 0.79 0.98 0.97), Σ|q̇| 3.7 → 2.7 rad/s, drift 4.0 → **1.9 cm** per 10 s,
0 % falls. Walking unaffected: 0.10 m/s → 0.1035, stride 0.131 (+5 %).

Run 10 finished at 13,898 (2,772 s): `logs/rsl_rl/hexapod_s0b/2026-09-22_12-11-04_s0b_seed42_r10_from_r9_settle/model_13898.pt`.
`eval_s0b_gates.sh` now covers L0, L5 and L9 (`logs/eval_s0b_r10_final.txt`, 256 episodes each):

| Condition | L0 | L5 | L9 |
| --- | --- | --- | --- |
| 0.10 m/s: v_x / stride (ref 0.125) / cycle RMSE | 0.100 / 0.125 (+0.3 %) / 0.004 | 0.101 / 0.127 (+1.6 %) / 0.006 | 0.101 / 0.128 (+2.2 %) / 0.010 |
| 0.15 m/s: v_x / stride (ref 0.1875) / cycle RMSE | 0.145 / 0.185 (−1.5 %) / 0.005 | 0.146 / 0.185 (−1.6 %) / 0.007 | 0.145 / 0.183 (−2.2 %) / 0.011 |
| falls (walking and standing) | 0 % | 0 % | 0 % |
| standing: contact duty / drift / Σ\|q̇\| | 0.85 / 1.8 cm / 2.3 | 0.82 / 1.8 cm / 2.5 | 0.79 / 2.0 cm / 2.6 |

No drops vs drops: identical (0.10: RMSE 0.0034 vs 0.0036). **All walking gates pass on L0/L5/L9** — the best
walking so far. Standing posture (`logs/s0b_stand_posture_r10.txt`): right-back fixed (load 1.12 × fair share), but
the stance is still off rest — left-back coxa at its +55° stop, LF coxa +35°, RF femur/tibia +29°/−32°; per-leg load
0.45–1.82 × fair share; total |q − q_rest| ≈ 5 rad (run 8: ≈ 11). The last 750 iterations changed it little (drift
1.9 → 1.8 cm). Next, proposed to the user: settle window 1.5 → 2.5 s (two gait periods, time to re-place more than
one leg), then the terrain extension (§2.1.19) and a final segment at the spec's 10 % standing share.

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

`scripts/rsl_rl/eval_terrain_sweep.sh <run 10 model_13898.pt> r10` (`logs/eval/sweep_r10_*.json`, 128 episodes per
cell; a second, stray copy of the loop — Ctrl-C had killed only its Python child — ran alongside and was left to
finish; every cell was then re-run by one loop on the final code, 108/108 present). Cell = falls / cycle RMSE /
stride error; levels 0–9 = training range, 10–14 = harder (level 14: rough ±2.25 cm, slopes 15°, bumps 3 cm, stairs
3 cm risers):

| v_x | type | L0 | L9 | L12 | L13 | L14 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | rough | 0 / .005 / −2 % | 0 / .012 / +2 % | 0 / .019 / +2 % | 0 / .018 / +2 % | 0 / .021 / +2 % |
| 0.10 | slope up | 0 / .004 / 0 % | 0 / .012 / +1 % | 0 / .017 / +1 % | 0 / .019 / +1 % | 0 / .021 / +2 % |
| 0.10 | slope down | 0 / .004 / 0 % | 0 / .011 / +3 % | 0 / .016 / +5 % | 0 / .017 / +5 % | 0 / .020 / +6 % |
| 0.10 | bumps | 0 / .005 / −3 % | 0 / .007 / −1 % | 0 / .008 / −1 % | 0 / .011 / −2 % | 0 / .010 / −1 % |
| 0.10 | stairs up | 0 / .004 / −1 % | 0 / .011 / +1 % | 0 / .018 / −1 % | 0 / .022 / −1 % | 0 / .025 / −2 % |
| 0.10 | stairs down | 0 / .004 / −1 % | 0 / .011 / +6 % | 0 / .015 / +7 % | 0 / .017 / +7 % | 0 / .018 / +8 % |
| 0.15 | rough | 0 / .008 / −3 % | 0 / .013 / −4 % | 0 / .019 / −7 % | 0 / .021 / −8 % | 0 / .026 / −9 % |
| 0.15 | slope up | 0 / .005 / −2 % | 0 / .016 / −4 % | 0 / .029 / −8 % | **0 / .032 / −8 %** | **0 / .038 / −10 %** |
| 0.15 | slope down | 0 / .005 / −1 % | 0 / .009 / −1 % | 0 / .014 / 0 % | 0 / .016 / 0 % | 0 / .018 / +1 % |
| 0.15 | bumps | 0 / .009 / −4 % | 0 / .009 / −4 % | 0 / .010 / −5 % | 0 / .012 / −5 % | 0 / .017 / −5 % |
| 0.15 | stairs up | 0 / .006 / −2 % | 0 / .014 / −5 % | 0 / .020 / −8 % | 0 / .023 / −10 % | **0 / .027 / −11 %** |
| 0.15 | stairs down | 0 / .006 / −2 % | 0 / .010 / −1 % | 0 / .015 / −1 % | 0 / .016 / −1 % | 0 / .016 / −2 % |

**0 % falls in all 108 cells**; the only gate failures (bold) are uphill at 0.15 m/s (slopes L13–14, stairs L14),
where the robot slows. Within the agreed cap (≤ 3 cm obstacles, ≤ 15°) the policy has no break point — trained on
levels 0–9 only. Step 3 of the plan (train to break point + 1–2, hold out the rest) therefore needs re-deciding with
the user.

### 2.1.23 S0b run 11: turning (continuation of run 10)

`eval_s0.py` now reports `wz_mean` and `wz_rmse_cycle` for yaw commands (stride only when a translation is
commanded). Standing share back at the spec's 10 %.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_12-11-04_s0b_seed42_r10_from_r9_settle --checkpoint model_13898.pt \
  --run_name s0b_seed42_r11_from_r10_turning
```

### 2.1.24 Held-out levels 15–20 (decision: train 0–14, hold out 15–20)

Decided with the user: train on levels 0–14 (the agreed cap), keep 15–20 — beyond the cap, eval-only — as the
held-out suite for the secondary result. `single_type_generator` now builds 21 levels with difficulty (ℓ + U)/15
(levels 0–14 unchanged; level 20: rough ±3.0 cm, slopes ≈ 19.7°, bumps and stairs 4.0 cm, at/over the 3.8 cm foot
lift). `eval_terrain_sweep.sh <run 10> r10x 15 16 17 18 19 20` (`logs/eval/sweep_r10x_*.json`, 72/72 cells):

| v_x | type | L15 | L17 | L20 |
| --- | --- | --- | --- | --- |
| 0.10 | rough | 0 / .023 / +1 % | 0 / .027 / −2 % | **0 / .042 / −10 %** |
| 0.10 | slope up | 0 / .025 / +1 % | **0 / .035 / −2 %** | **0 / .044 / −4 %** |
| 0.10 | slope down | 0 / .024 / +8 % | **0 / .030 / +11 %** | **0 / .043 / +16 %** |
| 0.10 | bumps | 0 / .013 / −1 % | 0 / .015 / −1 % | 0 / .019 / −5 % |
| 0.10 | stairs up | 0 / .025 / −1 % | **0 / .034 / −5 %** | **0 / .048 / −19 %** |
| 0.10 | stairs down | 0 / .019 / +8 % | 0 / .021 / +8 % | 0 / .024 / −5 % |
| 0.15 | rough | **0 / .028 / −10 %** | **0 / .035 / −13 %** | **0 / .048 / −20 %** |
| 0.15 | slope up | **0 / .046 / −12 %** | **0 / .060 / −15 %** | **0 / .075 / −19 %** |
| 0.15 | slope down | 0 / .022 / +1 % | **0 / .031 / +2 %** | **0 / .042 / +5 %** |
| 0.15 | bumps | 0 / .015 / −5 % | 1 / .023 / −7 % | **0 / .030 / −10 %** |
| 0.15 | stairs up | **0 / .030 / −13 %** | **0 / .041 / −16 %** | **0 / .068 / −30 %** |
| 0.15 | stairs down | 0 / .019 / −3 % | 0 / .021 / −3 % | **0 / .033 / −18 %** |

Falls stay ≈ 0 (one cell 1 %) even at 4 cm stairs and ≈ 20° slopes: beyond the cap the policy degrades in *speed
and stride accuracy* (uphill and rough first), not in survival. The held-out suite therefore has headroom, but its
metrics must include tracking error and stride, not only falls — noted for the secondary result (TBD).

Run 11 finished at 15,897 (4,354 s): `logs/rsl_rl/hexapod_s0b/2026-09-22_16-11-22_s0b_seed42_r11_from_r10_turning/model_15897.pt`
(`logs/eval_s0b_r11_final.txt`, 256 episodes per condition):

| Condition | L0 | L5 |
| --- | --- | --- |
| turn on the spot −0.3 rad/s: ω̄_z / yaw cycle RMSE | −0.228 / 0.063 | −0.218 / 0.079 |
| turn on the spot +0.3 rad/s | **+0.081 / 0.228** | +0.082 / 0.228 |
| walk 0.10 m/s + yaw 0.2 rad/s: v_x / ω̄_z / yaw RMSE | 0.106 / 0.153 / 0.049 | 0.106 / 0.155 / 0.054 |
| straight 0.10 (L0/L5/L9): v_x, stride | 0.106–0.107, +6 / +6.5 / +8 % | |
| straight 0.15 (L0/L5/L9): v_x, stride | 0.151, +1 / +1 / +0.5 % | |
| standing contact duty (L0/L5/L9) | 0.71 / 0.70 / 0.69 (run 10: 0.85) | |

Falls 0 % everywhere. Turning is partly learned and **lopsided**: right turns track (≈ 75 % of the command), left
turns barely (27 %). The coxa sign is mirror-symmetric by construction, so the asymmetry is the policy's — the
right-side weakness it has carried since π_base — plus only 2,000 iterations. Straight walking still passes every
gate (0.10 stride +6–8 %, inside ±10 %). Standing contact slipped with the standing share back at 10 %.

### 2.1.25 S0b run 12: training terrain = levels 0–14 + stairs; turning continues

`S0B_TERRAIN_CFG`: 15 rows, difficulty (0, 1); mix flat 0.15, rough 0.20, slope down 0.15, slope up 0.15, bumps 0.15,
stairs down 0.10, stairs up 0.10 (stairs: 0 → 3 cm risers, tread 0.3 m). Levels 0–9 keep their per-level
parameters; the mix changed (stairs added), so the training distribution of levels 0–9 is not identical to runs ≤ 11.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 3000 \
  --resume --load_run 2026-09-22_16-11-22_s0b_seed42_r11_from_r10_turning --checkpoint model_15897.pt \
  --run_name s0b_seed42_r12_from_r11_terrain14
```

Run 12 first launch crashed inside Kit start-up (thread in `libomni.kit.telemetry.plugin`, `__throw_regex_error`),
before any project code ran; the identical relaunch worked — a Kit flake, not the import-order bug of §2.1.2.
At 17,400: curriculum mean level 9.3 of 14, 0 falls; turn on the spot +0.3 → +0.108 rad/s (RMSE 0.196), −0.3 →
−0.272 (RMSE 0.029): left turns still barely work.

### 2.1.26 Lopsided turning: the coxa stroke term was farmable by tucked toes

**Observed** (`logs/s0b_turn_probe2.txt`, run 12 at 17,400, 128 envs, turn on the spot). The needed coxa sign
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
toe stroke speed, not coxa spin; at the rest pose the magnitude is unchanged, tucked toes earn ≈ 0. Flat/S0a keep
the fixed sign. Run 12 stopped at 17,630 (`…/2026-09-22_17-41-23_s0b_seed42_r12_from_r11_terrain14/model_17600.pt`).

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2500 \
  --resume --load_run 2026-09-22_17-41-23_s0b_seed42_r12_from_r11_terrain14 --checkpoint model_17600.pt \
  --run_name s0b_seed42_r13_from_r12_coxareach
```

Run 13 at 18,600 (128 envs, L0): turn on the spot −0.3 → −0.291 rad/s (RMSE 0.032); **+0.3 → +0.096** (RMSE 0.211);
walk 0.10 + 0.2 rad/s → 0.098 m/s, +0.152 rad/s (RMSE 0.046); straight 0.10 → 0.097, stride 0.120 (−4 %); 0 falls.
Probe (`logs/s0b_turn_probe3.txt`): the spin exploit is gone (tucked legs now stroke 0.1–0.3 rad/s), but the
**tucked posture remains in both turn directions** — right-front, left-front, left-middle, left-back stand with
|r_xy| at 3–10 % of their rest reach (toes almost under the hips); only right-middle and right-back reach out (60–
70 %). Toes under the hips cannot yaw the body through the coxa; the working right turn relies on the two right legs
that do reach. The fix removed the pay for the pose, not the learned pose (2,000+ iterations old); run 13 continues.

Run 13 finished at 20,099 (4,924 s): `logs/rsl_rl/hexapod_s0b/2026-09-22_18-39-41_s0b_seed42_r13_from_r12_coxareach/model_20099.pt`
(`logs/eval_s0b_r13_final.txt`, 256 episodes per condition, training terrain now levels 0–14):

| Condition | L0 | L5 / L9 |
| --- | --- | --- |
| straight 0.10: v_x / stride / cycle RMSE | 0.098 / 0.123 (−1.4 %) / 0.004 | L9: 0.099 / 0.125 (0 %) / 0.008 |
| straight 0.15 | 0.146 / 0.184 (−2 %) / 0.004 | L9: 0.146 / 0.182 (−3 %) / 0.008 |
| turn on the spot −0.3: ω̄_z / yaw RMSE | −0.294 / 0.026 | L5: −0.287 / 0.033 |
| turn on the spot +0.3 | **+0.114 / 0.192** | L5: **+0.110 / 0.198** |
| walk 0.10 + yaw 0.2: v_x / ω̄_z / yaw RMSE | 0.099 / 0.155 / 0.044 | L5: 0.099 / 0.154 / 0.048 |
| standing: contact duty / drift | 0.68 / 1.8 cm | |

Falls 0 % everywhere. All straight-walking gates pass on L0–L9; walk-and-turn passes the provisional yaw gate
(≤ 0.05); right turns on the spot pass; **left turns on the spot fail** (38 % of the command) — unchanged across
runs 11–13 despite the coxa fix. Status: open, discussed with the user (§2.1.27).

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

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_18-39-41_s0b_seed42_r13_from_r12_coxareach --checkpoint model_20099.pt \
  --run_name s0b_seed42_r14_from_r13_mirror
```

Finished at 22,098 (4,172 s, 2.1 s/iteration with the doubled mini-batches):
`logs/rsl_rl/hexapod_s0b/2026-09-22_20-30-48_s0b_seed42_r14_from_r13_mirror/model_22098.pt`. rsl_rl's symmetry loss
(action mismatch between a sample and its mirror; not optimised, logged only) 0.87 → 0.77 (21,100) → 0.75.
`logs/eval_s0b_r14_final.txt` (256 episodes per condition):

| Condition | run 13 | run 14 |
| --- | --- | --- |
| straight 0.10, L0 / L5 / L9: stride error, cycle RMSE | −1.4 % … 0 %, ≤ 0.008 | −0.1 / −0.4 / −0.2 %, 0.003 / 0.005 / 0.008 |
| straight 0.15, L0 / L5 / L9 | −2 … −3 %, ≤ 0.008 | −2.0 / −2.5 / −3.5 %, 0.005 / 0.006 / 0.008 |
| walk 0.10 + yaw 0.2 (L0 / L5): ω̄_z, yaw RMSE | 0.155 / 0.154, 0.044 / 0.048 | **0.172 / 0.170, 0.030 / 0.035** |
| turn on the spot −0.3 (L0 / L5) | −0.294 / −0.287, 0.026 / 0.033 | −0.278 / −0.273, 0.026 / 0.033 |
| turn on the spot +0.3 (L0 / L5) | +0.114 / +0.110, 0.19 / 0.20 | **+0.155 / +0.149, 0.151 / 0.157** |
| standing (L0): contact duty, drift | 0.68, 1.8 cm | **0.80**, 1.7 cm |

Falls 0 % everywhere; drops vs drop-free identical. Mirror augmentation helped every asymmetric measure (left turn on
the spot 38 % → 52 % of the command; walk-and-turn now passes the provisional yaw gate clearly; standing contact
0.68 → 0.80) without costing straight walking, but did not close the left/right gap within the agreed 1–2 h of
training. Per the user's plan, the next item is the from-scratch control arm (spec §4.2), which also tests whether
the asymmetry is inherited from the π_base lineage.

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

Gate evaluation of `logs/rsl_rl/hexapod_s0b/<…>_s0_fromscratch_seed42/` checkpoints (256 episodes per condition,
`logs/eval_fs_it*.txt`):

| Iteration (of 32,000) | falls | 0.10: v_x / stride err / RMSE | 0.15: v_x / stride err / RMSE | turn ±0.3 | standing duty / drift |
| --- | --- | --- | --- | --- | --- |
| 4,000 | 0 % (L0/L5/L9) | 0.095 / **−36 %** / 0.006 | 0.144 / **−29 %** / 0.007 | **+0.064 / −0.063** | 0.90 / 1.5 cm |
| 8,000 | 0 % (L0/L5/L9) | 0.094 / −4.9 % / 0.006 | 0.140 / −6.8 % / 0.009 | +0.187 / −0.191 | 0.97 / 1.0 cm |

**At 8,000 iterations — 0.30 × the warm-start budget — the from-scratch arm passes all four spec §4.2 S0b gates**
(falls 0 % on L0/L5/L9; v_x cycle RMSE ≤ 0.011; stride −4.6 … −8.3 %; drops identical to drop-free), well inside the
1.2 × rule. It also beats π_S0 on the extras: standing contact duty 0.97 vs 0.80 and drift 1.0 vs 1.7 cm,
walk-and-turn yaw RMSE 0.024 vs 0.030, and turning on the spot is **symmetric** (+0.187 / −0.191 of ±0.3) where the
warm-started lineage sits at +0.155 / −0.278. Per §4.2 this means **the warm start and the KL anchor (§4.4) are
dropped for every arm**; π_S0 becomes this run's policy (training continues to the full budget for quality).

At 15 % of the budget a random-init policy already walks at the commanded speed with no falls and stands better than
π_S0 (contact duty 0.90 vs 0.80), but its gait does not match the 0.8 Hz clock — it takes shorter, faster steps
(stride −29 … −36 %, contact duty 0.58 vs 0.50) — and it barely turns. Notably the turning is **symmetric**
(+0.064 / −0.063), unlike the π_base lineage: evidence that the left/right asymmetry of §2.1.26 is inherited.

| 18,750 | 0 % (L0/L5/L9) | 0.097 / −1.8 % / 0.004 | 0.146 / −1.8 % / 0.003 | +0.257 / −0.254 (RMSE 0.040 / 0.042) | 0.95 / 0.8 cm |

At 18,750 iterations (0.70 × the warm-start budget) the from-scratch arm passes **every** gate including both
provisional turning gates, and beats the warm-started π_S0 on all of them: stride error −1.8 % vs −0.1 … −3.5 %,
v_x cycle RMSE ≤ 0.005 vs ≤ 0.008, walk-and-turn yaw RMSE 0.012 vs 0.030, standing contact duty 0.95 vs 0.80 and
drift 0.8 vs 1.7 cm, and symmetric turning on the spot (+0.257 / −0.254 of ±0.3) against the lineage's +0.155 /
−0.278. Per the user's instruction the run continues to ~22,000 iterations and is then stopped for the final S0
evaluation at 1,024 episodes per condition (`eval_s0b_gates.sh <ckpt> <tag> 1024`; the spec floor is ≥ 200, and
1,024 tightens the 95 % upper bound on a 0 % fall rate from 1.2 % to 0.3 % at no extra wall time).

**Training stopped at 20,749 iterations** (user: the mean return plateaued at ≈ 400 for ~2,000 iterations on
TensorBoard), final checkpoint `logs/rsl_rl/hexapod_s0b/2026-09-22_22-50-43_s0_fromscratch_seed42/model_20700.pt`
= **0.78 × the warm-start budget**, of which the S0b gates were first met at 0.30 ×.

*Why the from-scratch stride error at 0.10 m/s (−1.8 %) is larger than the warm-started one (−0.1 %) while every
other number is better:* the stride gate measures speed accuracy more than gait shape. Stride = v_x / f_actual
against v_cmd / f_ref. At 0.10 the from-scratch policy walks 0.0966 (−3.4 % on the command) and steps at 0.787 Hz
(−1.6 % on the 0.8 Hz reference), leaving −1.8 %; the warm-started one walked 0.0984 at ≈ 0.800 Hz. At 0.15 the
from-scratch policy tracks better (0.1461 vs 0.1447) and its stride error is the smaller of the two. Both sit far
inside the ±10 % gate; the difference is ≈ 2 mm per step.

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

`scripts/rsl_rl/eval_terrain_sweep.sh <π_S0> pis0 0 3 6 9 10 … 20` (128 episodes per cell,
`logs/eval/sweep_pis0_*.json`; one cell, slope_up L17 v0.10, failed on a contended GPU and was re-run:
0 falls, RMSE 0.011, stride +4.5 %). Cell = falls / cycle RMSE / stride error:

| v_x | type | L0 | L9 | L14 (training ceiling) | L17 (held out) | L20 (held out) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | rough | 0 / .003 / +3 % | 0 / .005 / +4 % | 0 / .009 / +2 % | 0 / .015 / −3 % | **0 / .032 / −15 %** |
| 0.10 | slope up | 0 / .004 / +4 % | 0 / .004 / +5 % | 0 / .007 / +6 % | 0 / .011 / +4.5 % | 0 / .020 / +2 % |
| 0.10 | slope down | 0 / .004 / +4 % | 0 / .006 / +2 % | 0 / .009 / +2 % | 0 / .017 / +3 % | **0 / .031 / +3 %** |
| 0.10 | bumps | 0 / .003 / +2 % | 0 / .003 / +3 % | 0 / .004 / +2 % | 0 / .008 / +1 % | 0 / .018 / −4 % |
| 0.10 | stairs up | 0 / .004 / +4 % | 0 / .004 / +5 % | 0 / .007 / +1 % | 0 / .009 / −2 % | **0 / .053 / −26 %** |
| 0.10 | stairs down | 0 / .004 / +4 % | 0 / .006 / +4 % | 0 / .008 / +2 % | 0 / .009 / 0 % | **0 / .022 / −13 %** |
| 0.15 | rough | 0 / .002 / 0 % | 0 / .005 / −1 % | 0 / .014 / −6 % | **0 / .024 / −14 %** | **0 / .048 / −24 %** |
| 0.15 | slope up | 0 / .003 / +1 % | 0 / .004 / +1 % | 0 / .011 / 0 % | 0 / .021 / −2 % | **0 / .037 / −6 %** |
| 0.15 | slope down | 0 / .003 / +1 % | 0 / .006 / +1 % | 0 / .009 / +1 % | 0 / .013 / +1 % | 0 / .028 / 0 % |
| 0.15 | bumps | 0 / .002 / 0 % | 0 / .003 / −1 % | 0 / .007 / −2 % | 0 / .020 / −5 % | 0 / .028 / −8 % |
| 0.15 | stairs up | 0 / .003 / +1 % | 0 / .006 / −1 % | 0 / .011 / −6 % | **0 / .015 / −12 %** | **0 / .085 / −38 %** |
| 0.15 | stairs down | 0 / .003 / +1 % | 0 / .006 / 0 % | 0 / .007 / −4 % | 0 / .008 / −7 % | **0 / .023 / −19 %** |

**Every cell of the training range (levels 0–14) passes, with zero falls**; the warm-started policy already failed
uphill at L13 (§2.1.22). Falls stay at 0 % through the held-out levels too — degradation there is in speed and
stride (first failures at L17 on rough and stairs at 0.15 m/s, large by L20), which is the headroom §10.4 needs and
confirms the suite must be scored on tracking and stride, not falls.

### 2.1.34 Gait geometry: the feet migrated under the hips (open question)

The user saw in the teleop demo and `play.py` that the tibia points inwards rather than outwards.
`scripts/rsl_rl/record_gait.py` (new) walks a checkpoint at 0.15 m/s on flat ground, records a video
(`logs/videos/gait/*.mp4`) and prints per-leg joint angles, toe position in the body frame, **reach** (toe distance
from its own coxa joint projected on that hip's outward direction; rest pose ≈ 0.06 m) and **tibia_lat** (lateral
component of the tibia direction, + = shin outboard). `logs/gait_geometry.txt`, middle-leg values:

| policy | middle reach (m) | middle tibia° | tibia_lat | stance half-width (m) |
| --- | --- | --- | --- | --- |
| π_base | 0.184 | −28 | +0.090 | 0.201 |
| π_S0a | 0.102 | −44 | +0.048 | 0.161 |
| S0b r2 / r4 | 0.097 / 0.103 | −44 | +0.045 | 0.140 / 0.139 |
| S0b r10 / r13 / r14 | 0.126 / 0.135 / 0.136 (right) | +32…+34 (left) | −0.02…−0.03 (left) | 0.149 / 0.145 / 0.154 |
| **π_S0 (from scratch)** | **0.011–0.018** | **+34…+39** | **−0.047…−0.062** | **0.108** |

In π_S0 the four middle and back feet stand almost directly under their hips, knees splayed out and shins slanting
inward; the front legs are near normal (reach 0.08–0.10). Left and right mirror each other to ≈ 2 mm — the mirror
augmentation working.
**Why:** static joint torque scales with the horizontal distance from joint to foot, so pulling the foot under the
hip drives the torque-excess penalty, the servo sag (which the swing-height kernel punishes) and the stance-rigidity
cost toward their minima. Nothing in the reward set asks for stance width or leg extension, so the cheapest posture
wins; a narrow stance also shortens the moment arm for turning on the spot, which fits π_S0 being the best turner.
S0a did not show it because it inherited π_base's wide stance and only trained 4,498 iterations; the from-scratch run
optimised straight to the torque optimum.
**Risk:** all measured gates pass (0 falls to level 20), and the posture is kinder to the AX-12As, but the support
polygon is ~25 % narrower than the rest-pose footprint, and with the shin slanted inward the ground contact sits near
the edge of the foot rather than the rubber tip — a sim2real contact-geometry risk (v7 §5.1's foot model). It will
not resolve itself: S1–S3 keep this reward set minus the CPG terms, and nothing there pushes the feet outward.
**Open with the user:** (1) add a light stance-footprint term (keep stance toes near the rest-pose footprint) and
retrain S0b ~1–2 h; (2) cap the tibia angle; (3) accept and document.

**Swing-phase check** (the user saw "unnecessary tibia motion in the air"). Probe: per leg, per swing (foot off the
ground ≥ 3 steps), joint excursion, path length and direction reversals (deadband 0.0025 rad), 128 envs, flat, 0.15
m/s, steps 100–400 (`logs/swing_probe.txt`). A clean swing folds and unfolds once: path/excursion ≈ 2, ≈ 2 reversals.

| per swing, RF RM RB LF LM LB | π_S0a | S0b r14 | π_S0 |
| --- | --- | --- | --- |
| tibia excursion ° | 17 1 26 16 11 20 | 27 37 26 51 31 42 | 43 18 25 42 17 25 |
| tibia path / excursion | 2.3 1.8 1.7 1.9 **3.1** 1.9 | 1.3 1.7 1.9 1.5 1.8 1.9 | 1.6 2.0 1.9 1.6 2.0 1.9 |
| tibia reversals | 3.1 0.8 **4.0** 1.4 **3.0** 3.2 | 2.0 2.7 2.3 2.1 1.3 2.4 | 1.2 2.4 2.3 1.2 2.3 2.2 |
| femur excursion ° | 28 28 34 13 23 25 | 35 41 16 27 40 42 | 26 38 **49** 26 38 **49** |
| coxa excursion ° | 57 48 47 56 55 57 | 74 57 57 60 94 90 | 65 **95** 72 64 **96** 73 |

π_S0's tibia is not jittery — it is the smoothest of the three (π_S0a wiggles: up to 3.1 × path/excursion and 3–4
reversals). What looks unnecessary is **amplitude**: middle-leg coxa arcs of 95° (π_S0a ≈ 50°), close to the ±55°
limit on each side, and larger femur/tibia arcs. It follows from the tucked stance: toe travel = coxa angle × reach,
and with the middle toes ≈ 1 cm outboard of their hips the coxa must sweep far to make the same ≈ 19 cm stride. The
inward shins and the large arcs are one problem. Proposed to the user: a stance-footprint term (L1 on each stance
foot's reach minus that leg's rest-pose reach, measured at env build), weight sized with `reward_economy.py`,
continue π_S0 ~2,000 iterations, then gates, sweep, geometry table and video.

### 2.1.35 Rest-reach bug (runs 13–14 and π_S0) and the stance-footprint term

**Observed.** Sizing the footprint term on π_S0 (`reward_economy.py --set rewards.stance_footprint.weight=-1`):
motionless rest-pose standing showed a *larger* footprint deviation (0.84 m summed) than the tucked walking policy
(0.47 m) — impossible if the target were the rest pose. `env.coxa_rest_reach` read **0.2079 m for every leg**;
standing at rest the actual reach is 0.054–0.093 m (`logs/reach_debug.txt`).
**Root cause.** `HexapodEnv.load_managers` measured it while the articulation was still in the USD's zero pose
(all joints 0 → legs straight out: 0.0529 + 0.06 + 0.095 = 0.2079 m); the configured rest pose is only applied at
the first reset. **Consequence for trained policies:** `coxa_stroke_weight` (§2.1.26) divided the toe reach by 0.208
instead of the rest reach, so from run 13 onward — including π_S0 — the coxa stroke term ran at ≈ 0.32 × its
intended strength. §2.1.26's "at the rest pose the magnitude is unchanged" was therefore wrong; the sign logic was
unaffected. Plausibly it contributed to the tucking: the coxa term is the one that pays more for extended legs.
**Fix.** At build, write the rest pose to the joints, take one physics step so PhysX refreshes the link transforms,
then measure; assert reach < 0.15 m. Now **0.0669 m for every leg** (`logs/reach_debug2.txt`). The coxa term on π_S0
reads 0.077/step instead of 0.025 (dt-scaled, 0.15 m/s) — ×3.1 as expected; near the command the tracking slope
(≈ 5 per m/s) still dwarfs the coxa term's (≈ 1.2), so no speed bias.

**Stance-footprint term.** `stance_footprint_penalty` = Σ over feet in contact of | |r_xy| − rest reach | (m); the
horizontal distance from the coxa axis is invariant to the coxa sweep, so it restores extension without touching the
stride. Economics (`logs/s0b_economy_footprint2.txt`, π_S0, L0, per step dt-scaled at weight −1): walking 0.15 → 0.0015
(≈ 7.5 cm summed, ≈ 2.5 cm per stance foot); turning on the spot → 0.0030; motionless rest-pose standing → 0.0007
(servo sag, the floor any policy pays). The torque-excess term is ≈ 0 (0.0001), so the tucking was not bought with a
large reward saving — it was simply unopposed. **Weight −20**: the tucked posture costs ≈ 0.03/step (≈ 3 % of the
walking reward), on a par with the stance-rigidity and force terms; walking ≫ standing ≫ falling unchanged.

```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_22-50-43_s0_fromscratch_seed42 --checkpoint model_20700.pt \
  --run_name s0_fromscratch_seed42_footprint agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
```
(total 22,700 iterations, still below the 32,000-iteration 1.2 × budget). Acceptance, agreed with the user: 1,024-episode
gates, geometry table and video first; the terrain sweep only if the gait is visibly better and the gates hold.

Finished at 22,699 (4,262 s; the first launch died in Kit start-up, the telemetry flake of §2.1.25, and was relaunched):
`logs/rsl_rl/hexapod_s0b/2026-09-23_14-50-13_s0_fromscratch_seed42_footprint/model_22699.pt`.

**Gates, 1,024 episodes per condition** (`logs/eval_pi_s0fp_final.txt`): all pass, 0 falls in 17,408 episodes, and
slightly better than π_S0 everywhere — stride error at 0.10 m/s +1.4 / +2.3 / +2.9 % on L0 / L5 / L9 (π_S0 +3.9 /
+4.2 / +4.6 %), at 0.15 m/s −0.5 / −0.1 / −0.2 %; v_x cycle RMSE ≤ 0.005; turn on the spot +0.278 / −0.272 rad/s
(RMSE 0.021 / 0.026; π_S0 0.038 / 0.032); walk-and-turn yaw RMSE 0.009; standing contact 0.97, drift 0.9 cm.

**Posture** (`logs/gait_geometry_fp.txt`, `logs/swing_probe_fp.txt`): middle-leg reach 0.012 → 0.029 m, back-leg
0.018 → 0.034 m (rest 0.067), stance half-width 0.108 → 0.122 m (π_S0a 0.161) — but the middle/back tibias went
from +34…+39° to **+43…+44°** and still slant inward (tibia_lat −0.04…−0.06), and the middle-leg coxa swing arc is
unchanged at 95°. The policy met the footprint target partway and the cheap way — bending the tibia further instead
of straightening the leg. By eye "more or less the same", so per the user's acceptance rule no terrain sweep yet.
Options put to the user: (1) add a light penalty on inward tibia slant in stance, keeping the footprint term;
(2) raise the footprint weight to ≈ −60; (3) accept (this checkpoint or the committed π_S0) and document.

### 2.1.36 Mid-air toe in/out motion: detection (no training yet)

**Observed Behavior.** By eye (user) the four middle/back legs of π_S0 and of the footprint continuation swing the toe
in and out mid-air: the shin "slants more, then less just before landing". The front pair and π_S0a look fine.

**Detection.** `scripts/tests/swing_radial_probe.py` (flat, 0.15 m/s, 128 envs × 300 steps after a 100-step settle):
per swing (≥ 5 airborne steps), the toe's horizontal distance from its coxa axis |r_xy| (radial, in/out), its
tangential sweep, lift and tibia angle, plus the mean profile over 10 normalised swing bins (lift-off → touchdown) of
Δ|r_xy| and Δtibia relative to lift-off. Output `logs/swing_radial_probe.txt`:

| per swing, middle/back legs | π_S0a | π_S0 | π_S0 + footprint |
|---|---|---|---|
| radial excursion (cm) | 6.7–9.7 | 5.9–7.9 | 6.5–8.9 |
| radial path (cm) | 12.6–19.0 | 19.7–20.0 | 18.0–21.6 |
| lift (cm) | 3.6–3.9 | 3.8–4.0 | 3.7–4.0 |
| sweep (cm) | 10.6–14.8 | 4.2–5.6 | 4.1–5.0 |
| Δ\|r\| profile (mm) | 0 → **+69 out** → +6 | 0 → **−53 in** → −19 → **−54 in** | 0 → −57 → −37 → −64 → −53 |
| Δtibia profile (°) | 0 → −6 → +3 | 0 → **+16** → −2 | 0 → **+20** → +2 |

**Findings.**
1. The perceived motion is real and located: π_S0's middle/back legs lift by *folding the tibia* (+16°, toe pulled
   5 cm toward the hip in the first 30 % of the swing), unfold it again by 60–100 % (the "slants less before
   landing"), with a second inward dip, and land 5 cm closer to the hip than they lifted off — the stance then has
   to carry the foot back out. π_S0a lifts with the femur: the toe arcs 7 cm *outward* and lands where it lifted off,
   the tibia moving only ±6°. The footprint term made it slightly worse (+20°).
2. **But the amount of radial motion does not separate the policies**: π_S0a's radial path (12.6–19 cm per swing)
   is close to π_S0's (≈ 20 cm). What differs is the *direction and mechanism* (inward, via the tibia, with a
   reversal; vs outward, via the femur, once) and the stride: π_S0's middle/back sweep is 4–6 cm vs 11–15 cm.
   A plain |Δ|r_xy|| swing penalty (the planned term) would charge π_S0a's liked gait about as much as π_S0's, so
   per the agreed gate it is **not** implemented; re-discussed with the user.

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

**Pilot** (from π_S0, 2,000 iterations):
```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 2000 \
  --resume --load_run 2026-09-22_22-50-43_s0_fromscratch_seed42 --checkpoint model_20700.pt \
  --run_name s0_pilot_coordlift agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
```

**Pilot result** (stopped by the user at a plateau, `…/2026-09-23_19-07-33_s0_pilot_coordlift/model_22050.pt`,
1,350 iterations). Gates at 1,024 episodes (`logs/eval_pilot_coordlift.txt`) all pass, 0 falls: stride at 0.10 m/s
+1.4 / +2.1 / +2.5 % (L0 / L5 / L9), at 0.15 m/s 0.0 / +0.2 / −0.1 %; v_x cycle RMSE ≤ 0.005; turn on the spot
+0.268 / −0.266 rad/s (RMSE 0.032 / 0.033); walk-and-turn yaw RMSE 0.008 / 0.010; standing contact 0.95, drift 1.0 cm.
**The gait did not change** (`logs/swing_radial_probe_pilot.txt`, `logs/gait_geometry_coordlift.txt`): middle/back
radial path per swing 17.7–20.8 cm (π_S0 19.7–20.3), landing still 4.7 cm inside lift-off, tibia fold +19° mid-swing,
middle/back reach 0.018–0.031 m, half-width 0.110–0.117 m. Both new terms stayed flat over the whole run
(Episode_Reward/swing_radial −0.83 → −0.83, touchdown_reach −1.34 → −1.30; ≈ 3–5 % of track_lin_vel), i.e. the
converged policy (std at the 0.05 floor) did not find a cheaper lift, the same as the footprint run of §2.1.35.
The front legs dominate the radial cost (23 cm per swing, toe thrown 15 cm outward via tibia unfold), which the user
considers fine.

**5× check** (user request; radial −5, touchdown −1,500 via Hydra overrides, from π_S0, 1,700 iterations,
`…/2026-09-23_20-32-32_s0_pilot_coordlift_x5/model_22399.pt`):
```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 1700 \
  --resume --load_run 2026-09-22_22-50-43_s0_fromscratch_seed42 --checkpoint model_20700.pt \
  --run_name s0_pilot_coordlift_x5 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0 \
  env.rewards.swing_radial.weight=-5.0 env.rewards.touchdown_reach.weight=-1500.0
```
The terms now move (touchdown −6.0 → −4.2, radial −4.2 → −3.8 per episode-second), but the wrong way: middle-leg
radial path 19.9 → 15.3 cm by *shrinking the middle-leg sweep* (5.3 → 3.9 cm) and *holding the tibia folded
through the swing* (+21° peak, still +9° at touchdown, vs π_S0 +17° → −1°); back legs unchanged (20.3 cm); landing
still 4.5 cm inside lift-off; reach 0.022–0.035 m, half-width 0.112–0.117 m (`logs/swing_radial_probe_pilot2.txt`,
`logs/gait_geometry_pilot2.txt`). Gates (`logs/eval_pilot2.txt`, 1,024 episodes) still pass (spec: RMSE ≤ 0.03,
stride ±10 %), 0 falls, but tracking is a little worse: v_x at 0.10 overshoots +3.0…+3.6 %, stride +5.0 %
(walk-and-turn +8.6 %), v_x cycle RMSE 0.005–0.006; turning improved (+0.272 / −0.278, RMSE 0.026 / 0.018).
Conclusion: fine-tuning a converged π_S0 does not reach a coordinated lift even at 5×; the tucked-tibia basin is
deep and the cheapest local response is to freeze the fold, not to re-coordinate the leg.

**From-scratch run with the coordinated-lift terms** (user decision: 18,000 iterations, where π_S0's return
plateaued). Weights fixed in the cfg at 2× the pilot: radial −2, touchdown −600 (on π_S0's behaviour ≈ 0.04 + 0.05 per
step, ≈ 9 % of the walking reward; ≈ 0 for a coordinated lift; walk 0.88 ≫ stand 0.27 ≫ fall at a 0.15 command). Both
terms apply to all six legs (one rule, no leg-specific tuning). Lift stays defended early by `cpg_z_target` and
`swing_clearance`. Same task, seed, envs, mirror augmentation and everything else as π_S0 (§2.1.31):
```
python scripts/rsl_rl/train.py --task Hexapod-S0b-v0 --headless --num_envs 4096 --seed 42 --max_iterations 18000 \
  --run_name s0_coordlift_fromscratch_seed42 agent.algorithm.actor_freeze_iters=0 agent.algorithm.actor_freeze_ev=0.0
```

**Preliminary check at 1,750 iterations** (user question: mean return ≈ 300 vs ≈ 238 for the π_S0 run at the same
iteration — reward change or better policy?). Curves at iteration ≈ 1,784, old vs new run (per episode-second ×10 s
per episode): track_lin_vel 16.8 → 19.9 and track_yaw 1.6 → 3.2 (+47 of return, real), cpg_coxa_contact 0.21 → 3.06 and
cpg_z_target 2.1 → 3.0 (+37, largely accounting: the rest-reach fix of §2.1.35 makes the coxa term ≈ 3× larger for the
same motion), new penalties −1.8 (−18). Terrain level 9.4 vs 10.0 (similar). So about two thirds of the gap is real
tracking and one third is bookkeeping. Radial probe (`logs/swing_radial_probe_cl1750.txt`, vs the π_S0 run's
model_1750): middle/back radial path per swing 6.0–6.3 cm (old 12–19; π_S0 final ≈ 20), excursion 2.4–2.7 cm,
sweep 8–13 cm (old 4–11; π_S0 final 4–5), lift 3.8–4.6 cm; the tibia folds ≈ 44° mid-swing while |r_xy| changes
only ≈ 1.5 cm, so the femur compensates: a coordinated lift. Gates at 128 episodes (`logs/eval_cl1750.txt`): 0 falls;
v_x cycle RMSE 0.009–0.011; stride +6.1 / +5.0 / +1.8 % / +0.4 %, walk-and-turn +9.4 %; turn +0.280 / −0.272 rad/s;
standing not learned yet (contact 0.68–0.74, joint speed 12.4, drift 4 cm; π_S0 final 0.95 / 0.9 / 0.9 cm).

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

**Milestone 2,000** (`logs/milestones/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/it2000_*`): return
291.6 (last 500 iterations; still rising steeply), terrain level 9.4. Clearance: worst capsule surface gap **+0.5 cm**
(no overlap under any command; was −3.8 cm without the term), under 2 cm on ≤ 3.9 % of steps (0.15 m/s ≤ 3.2 %).
Coordinated lift kept: middle/back radial path 7.3–7.9 cm per swing, lift 3.4–4.1 cm, sweep 7–12 cm. Middle/back
stance reach 0.067–0.071 m (rest 0.067), half-width 0.12 m; the front legs sit forward (coxa ±22…26°) and narrower
(reach 0.04 m).

**Milestone 4,000:** return 337.3 (+15.7 % over 2,000 iterations), track_lin_vel 20.3 → 22.0, leg_proximity
−0.14 → −0.17 per episode-second (small), worst surface gap +0.6 cm; middle/back radial path 7.8–8.1 cm, lift
3.6–4.0 cm, sweep 7–11 cm; half-width 0.12–0.13 m. Still improving.

**Milestone 6,000:** return 365.3 (+8.3 %), track_lin_vel 23.0 (+4.5 %), leg_proximity −0.17 (flat), worst surface gap
+0.2 cm; middle/back radial path 7.8–8.0 cm, lift 3.9–4.0 cm, sweep 8–11 cm; half-width 0.13 m. Still improving.

**Milestone 8,000:** return 382.9 (+4.8 %), track_lin_vel 23.6 (+2.8 %), leg_proximity −0.165, worst surface gap
+0.7 cm; middle/back radial path 6.8–8.4 cm, lift 4.1–4.4 cm, sweep 8–10 cm; half-width 0.14 m. Slowing; plateau
criterion used: return gain < ≈ 1.5 % per 2,000 iterations.

**Milestone 10,000:** return 389.5 (+1.7 %, just above the plateau criterion), track_lin_vel 23.9 (+1.0 %),
leg_proximity −0.165 → −0.107 (still being reduced), worst surface gap +0.9 cm; middle/back radial path 5.9–8.2 cm,
lift 4.1–4.6 cm, sweep 7–10 cm; half-width 0.14 m. Continued to 12,000.

**Milestone 12,000 and stop.** Return 395.3 (+1.5 %: plateau criterion met), track_lin_vel +0.3 %; leg_proximity
still falling (−0.107 → −0.083), worst surface gap +1.3 cm; middle/back radial path 5.7–7.9 cm, lift 4.3–4.4 cm,
half-width 0.14 m. Training stopped at 12,2xx; final checkpoint
`logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt`.

**Gates, 1,024 episodes per condition** (`logs/eval_coordlift_prox_final.txt`,
`scripts/rsl_rl/eval_s0b_gates.sh <ckpt> coordlift_prox_final 1024`): all pass, 0 falls in 17,408 episodes.
Stride error at 0.10 m/s +1.0 / +0.3 / −0.7 % (L0 / L5 / L9; π_S0 +3.9 / +4.2 / +4.6 %), at 0.15 m/s −0.2 / −0.6 /
−1.4 %; v_x cycle RMSE ≤ 0.0046; drops vs no drops equal; turn on the spot +0.321 / −0.311 rad/s of ±0.3 (RMSE
0.039–0.048; π_S0 ±0.26); walk-and-turn v_x 0.100, yaw 0.216 of 0.2, stride +4.6 %; standing contact 0.97, drift
0.8 cm, joint speed 0.92. Candidate π_S0 (pending the user's visual check; no terrain sweep yet).

**Terrain sweep** (user accepted the gait; nearest legs 3–4 cm by eye): `scripts/rsl_rl/eval_terrain_sweep.sh
<model_12200> pis0cl 0 3 6 9 10 … 20` (128 episodes per cell, `logs/eval/sweep_pis0cl_*.json`, `logs/sweep_pis0cl.txt`;
180/180 cells). **All 108 training-range cells (L0–14) pass; 0 falls in all 180 cells.** Held-out L15–20: 50/72 pass
(π_S0 52/72); failures are stride shortfalls, never falls: rough from L16 at both speeds (−11 … −18 %, the new
policy's weak spot), slope-up at 0.15 from L18, stairs-up from L18/19, stairs-down at L19–20.

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
