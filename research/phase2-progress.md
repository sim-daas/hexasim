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
