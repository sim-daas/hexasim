# CLAUDE.md

Guidance for Claude Code (and any other agent) working in this repository.

## Project

`hexasim` is an Isaac Lab (Isaac Sim) reinforcement-learning project training an
Interbotix PhantomX Mark IV hexapod (18-DOF, 6 legs) to walk. It is being developed
as the codebase behind a submission to an international robotics/ML conference, so
code, experiments, and results must hold up to external scrutiny: reviewers and
readers should be able to reproduce every reported number from this repo.

- **Phase 1 (in progress → converging):** Manager-Based RL environment, PPO
  (`rsl_rl`), CPG-guided reward shaping, pure proprioception, flat ground, no
  domain randomization. See `research_progress.md` and
  `research/hexapod-RL-tuning-guide.md`.
- **Phase 2 (next up, not yet implemented): the comparison baselines.** Extends
  the Phase 1 flat-ground CPG policy with terrain curriculum (progressively
  harder generated terrain, material library, transition profiles) and domain
  randomization (friction, compliance, payload, actuator/bus model, dropped
  reads, etc.), plus the shared S0/S1 infrastructure and the baseline arms
  O, B0, B1-K, B1-M, B2 defined in the Shared Training Paradigm spec. This is
  **not** the JEPA work — it is what the JEPA policy will be compared against,
  so it must be trained and evaluated to a reportable, reproducible standard in
  its own right.
- **Phase 3 (design stage, not yet implemented):** Arm F, the Anticipatory
  Visuo-Proprioceptive JEPA (VP-JEPA), and optional arm B3 — evaluated against
  the Phase 2 arms under the same paradigm.

### Design specs (read before non-trivial changes)

- `VP-JEPA_Architecture.md` (v7) — architecture, CPG handling, camera geometry,
  simulation/actuator model, deployment.
- `VP-JEPA — Shared Training Paradigm & Baseline Specification.md` (v2) — the
  training paradigm, arm definitions, invariants, stage schedule, statistics.
  **v2 overrides v7 wherever its §0.5 precedence table says so.** These two
  replace the earlier `algorithm.md` (no longer in the repo).
- `research_progress.md` and `research/` — experimental history and current
  failure modes that aren't visible from the code alone.

The phase split above is this file's reading of the specs plus the user's
statement that Phase 2 is the baseline; if the user redefines it, update this
section first.

## Stack

- Python environment: `conda activate env_isaaclab`
  (`/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python`). Activate this env
  before running training, eval, lint, or any script that imports Isaac Lab —
  the system/base Python does not have it installed.
- Python 3.12, Isaac Lab / Isaac Sim, PyTorch, `rsl_rl` (PPO).
- Custom extension lives in `source/hexasim/hexasim/` (installed editable via
  `pip install -e source/hexasim`), task registered as `Hexapod-Flat-v0` and
  driven by `hexapod_env.py` / `hexapod_env_cfg.py` / `mdp/rewards.py`.
- Entry points: `scripts/rsl_rl/train.py` (supports `--resume_last`),
  `scripts/rsl_rl/play.py` / `play_debug.py`, `scripts/zero_agent.py`,
  `scripts/random_agent.py`, `scripts/list_envs.py`.
- Lint/format: `ruff` (line-length 120, py310 target) via `pre-commit`. Run
  `pre-commit run --all-files` before considering a change done. License-header
  insertion is enforced on `.py`/`.yml` files — don't strip it.

## Current state of the Phase 1 base policy (verified from code and checkpoint)

This is the warm-start candidate for S0 in the paradigm spec; several items
must change before it is deployable (spec §4.1 audit). Re-verify against the
code before relying on any line here.

- Actor obs (**Phase 1 policy only** — Phase 2/3 layouts differ per arm and
  stage, see the paradigm rules below): 75-D = base_lin_vel(3, **ground truth, non-deployable**),
  base_ang_vel(3), projected_gravity(3), joint_pos_rel(18),
  joint_vel_rel(18, **PhysX velocity — S0 mandatory**), last_action(18),
  cpg_clock(12, sin/cos × 6 legs). No noise terms configured
  (`enable_corruption=True` but nothing to corrupt).
- Actuators: `ImplicitActuatorCfg`, Kp 20, Kd 0.5, effort 3 Nm (≈ 2× AX-12A
  stall), vel π rad/s — not the AX-12A model the spec requires.
- `sim.dt` = 1/200 s, decimation 4 → 50 Hz control (20 ms), matching the spec.
- Actions: `JointPositionActionCfg`, scale 0.5, `use_default_offset=True`
  (rest pose coxa 0, femur +30°, tibia −30°). `clip_actions` is null.
- Normalisation: `actor.obs_normalization=True`, statistics live inside the
  checkpoint (`actor_state_dict["obs_normalizer._mean/_var/_std/count"]`, and
  the critic's own). The `empirical_normalization` key in `agent.yaml` is empty
  and unused by this rsl_rl version.
- Learned action std: `actor_state_dict["distribution.std_param"]` (scalar std,
  18 values), ≈ 0.01–0.12 (e.g. femur/tibia ≈ 0.012–0.025, some coxa ≈ 0.09).
  Never reset to rsl_rl's `init_noise_std=1.0` (spec §4.3).
- Latest checkpoint: `logs/rsl_rl/hexapod_flat/2026-09-06_19-45-34/model_14998.pt`.
  Its logged `params/` record **4,096 envs × 16 steps, seed 42**. The paradigm
  spec (§1, §4.1, §8.1) calls the base run "8,192 × 16" — that is wrong for
  this checkpoint. User decision (2026-09-21): S0/S1 run at 4,096 × 16 (compute limit).
  Its "mean v_x 0.209 m/s" came from one 10 s run. The 256-episode reference
  (`scripts/rsl_rl/eval_s0.py`, phase2-progress §2.0.2) is 0.159 m/s, 0 % falls on
  the Phase 1 code, and 0.181 m/s, 9 % falls on the fixed Flat env (random reset phase).
- S0 reconditioning (paradigm spec §4.2), per item:
  - **S0a** (flat): AX-12A actuator model with dropped reads + 18-bit mask;
    per-step frame → 111-D `o_p` = q(18) + q̇(18) + tracking error
    `q_cmd − q_meas`(18) + last action(18) + IMU accel(3) + ang vel(3) +
    gravity(3) + clock(12) + validity mask(18); ground-truth `base_lin_vel`
    replaced by `V_0` on a 0 → 1 substitution ramp over 50 %; joint velocity →
    two-sample difference of ZOH positions; observation noise; actions
    **clipped** to the servo CW/CCW limits; std **floored at 0.05** (S0 only);
    normaliser **re-estimated** (50-iteration burn-in, actor frozen, count
    capped), not carried. dt/decimation and action scale/offsets are kept.
  - **S0b**: `twist_cmd`(3, zero-init) + tracking reward, command distribution
    per spec §1 (resolved); clock `f_ref = clamp(v_cmd,x/0.20, 0.8, 2.5)` from
    the command, per-env phase jitter; easy-to-medium terrain curriculum;
    phase-free floor terms introduced here, not in S2.
  - Gates for both are in spec §4.2; a from-scratch blind control arm decides
    whether the warm start is kept at all.
- TCN window is **W = 50** (v2 overrides v7's 25); pin the residual block in
  code and assert RF ≥ W at construction; sweep W ∈ {25, 50, 75} on S1.
- Phase 2.0 (2026-09-21, `research/phase2-progress.md`) fixed the v7 §2 / v2 §0.4
  code defects: toe point everywhere (`mdp.rewards.toe_pos_w`), clock floor 0.8 Hz
  + per-env phase jitter, one signed/clipped coxa term, `duty_factor` single source
  (`env.cfg.duty_factor`), leg/joint order asserted at env build, force target =
  weight/3, friction combine multiply (ground μ 0.75 = π_base's effective training μ).
  Coxa is deliberately **excluded** from `cpg_stance_rigidity` (it is the stance
  stroke `cpg_coxa_contact` pays for). Still open: `net_forces_w[..., 2]` is world-Z,
  not surface normal — must be projected before any sloped terrain (S0b).
- `Hexapod-Flat-v0` keeps `PXMARK4_CFG` (implicit PD, Phase 1 rest pose) so π_base
  stays reproducible. `Hexapod-S0a-v0` uses `PXMARK4_AX12A_CFG`: AX-12A servo model
  (`hexasim/actuators/ax12a.py`, datasheet values, provisional) and rest pose femur
  +30° / tibia +30°. The Phase 1 pose fails the §4.2 torque gate (femur 134 % of
  stall in tripod stance). Isaac Lab's `applied_torque` for **implicit** drives
  under-reads ~2× — use lever-arm torques (`scripts/tests/static_torque_envelope.py`).
- **Explicit actuators need `PhysxCfg(enable_external_forces_every_iteration=True)`**
  (phase2-progress §2.0.9): with Isaac Lab's default TGS setting, explicit joint
  efforts barely act and a policy under them learns to stand. Explicit PD on these
  light links also needs armature (AX-12A model: 0.003 kg·m²) to be numerically
  stable at 5 ms. The USD `jointFriction` is not applied (sim friction reads 0).
- **π_S0a (2026-09-21):** `logs/rsl_rl/hexapod_s0a/2026-09-21_16-43-30_s0a_seed42_r4_from_r3/model_4498.pt`
  on `Hexapod-S0a-v0` — 0 % falls, v_x 0.29 m/s, all six legs 1.49 swings/s, toe apex
  3.8 cm, V_0 only. S0a reward economics differ from Flat (bounded toe-swing kernel
  σ 1 cm ×5, alive ×10, stand_still −75): check any reward change with
  `scripts/tests/reward_economy.py` (walk > stand > fall) before training.
- **π_S0 (final, 2026-09-24, trained from scratch):**
  `logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt` — 12,214
  iterations (7.3 h), no warm start, mirror augmentation, with the gait-quality terms of phase2-progress §2.1.37–38:
  `swing_radial` (−2), `touchdown_reach` (−600), `leg_proximity` (−400, 2 cm capsule margin; self-collision is off
  in sim). All §4.2 S0b gates pass at 1,024 episodes (0 falls in 17,408; stride −1.4 … +1.0 %; turning +0.32 /
  −0.31 of ±0.3; standing 97 % contact), all 108 training-range sweep cells pass, 0 falls in all 180. The previous
  π_S0 (`…/2026-09-22_22-50-43_s0_fromscratch_seed42/model_20700.pt`, tucked tibia-fold gait, legs overlapping in sim)
  is superseded (§2.1.39). **The blind from-scratch control arm passed the gates at 0.30 × the warm-start budget, so
  the warm start and the KL anchor (spec §4.4) are dropped for every arm**; the π_base → S0a → S0b lineage
  (research/phase2-progress §2.0–2.1) is history, kept for the record and for the paper's honesty section.
  Check gait changes with `scripts/tests/swing_radial_probe.py` and `scripts/tests/leg_gap_probe.py`, not only gates.
- **S1 (research/phase2-progress §2.2, final config accepted 2026-09-27):** `Hexapod-S1-v0` = S0b + spatial terrain
  materials (static field, per-entry visual submeshes, per-foot μ/k tensor writes; damping per env, D25), per-env drop
  slope (D24), the env-owned aux learner (`hexasim/rl/s1/`: E_p TCN W 50, V = v_kin + Δv̂, probes, β-NLL with fixed π_S0
  statistics in `probe_stats_pi_s0.pt`), 1,125-D actor layout (`mdp/layout.py`), contract P (683), `S1PPO` (ê gate after
  the v7 §4.3 tests, V_0 retired by per-episode dropout, D23). **S1 reward = S0b except `leg_proximity` −1,600,
  `swing_clearance` 4 cm (D28), `cpg_z_target` step 4 cm and a per-leg two-sided `touchdown_reach` (front legs 10 cm,
  rest reach for the others; D29)**: longer training of the S0b reward pulls the front feet inward and closes the leg
  gaps (§2.2.6, §2.2.8). **S1 budget = 8,000 iterations.** Pilot 3 (seed 42, W 50, resumed at 4,500) passes every S0b
  gate with 0 falls, worst leg gap +0.5 cm, lift 3.8–4.5 cm, V within v7 §4.3; known imperfection D30: back-coxa
  requests past the ±55° clip (0.026–0.029 rad/step vs 0.015 in π_S0), to revisit in S2. Checkpoint
  `logs/rsl_rl/hexapod_s1/2026-09-26_19-41-49_s1_pilot3_seed42_resume4500/model_7999.pt`. W sweep done: **W = 50** (D32).
  **5 S1 trunks** (seeds 42–46, D31; run dirs `…/hexapod_s1/*_s1_w50_seed4{3..6}`): **all five** pass every gate with 0
  falls in 21,760 episodes and a small seed spread (§2.2.11) — the pairing units for the spec §10.1 statistics. Next: the blind headroom pilot (bullet below), then N_v benchmark → S1b → S2; S1-K (B1-K trunk)
  deferred. Eval seeds must be fixed across arms: terrain geometry
  depends on the run seed (Isaac Lab rough/bump terrains use numpy's global RNG, §2.2.9).
  Gotchas: `OPENBLAS_NUM_THREADS=1` for Kit scripts; evals next to S1 training need `load_policy.eval_physx_buffers`;
  load S1 checkpoints with `load_policy.load_env_state` before `env.reset()`; a resume (`--resume --load_run … --checkpoint
  …`) writes a new run dir and `max_iterations` counts from the loaded iteration.
- **S1 review response (research/phase2-progress §2.3, 2026-10-02):** W 50 is deviation D32 (rule selects no W); trunks
  hold every gate at 15 ms latency + 2× drop slope; held-out material layout (seed 1) ≡ training layout — frozen evals
  use layout ≠ 0; flat-ground friction is harmless for this robot, so material transitions matter only on slopes/steps;
  **primary behaviour metrics = transition-window tracking/slip/falls + first-contact impact, stride descriptive (D38)**.
  **Next: blind headroom pilot B0 vs O-mat (M*) vs O-geo (H* height strips, D37)** from seed 43 at 4,096 × 16, spec
  in §2.3.7 (pre-registered go/no-go). Partly implemented (`Hexapod-S2-v0`, `train.py --branch_from`, `env.clock_live`,
  `cfg.arm`, `S1PPO` clock anneal/re-burn-in); H* block, trunk widening, smoke test and the headroom eval still to do.
- **S0b (research/phase2-progress §2.1):** `Hexapod-S0b-v0` — twist_cmd (117-D actor), command
  clock, standing mode at zero command (`env.cpg_active`), terrain curriculum (`mdp/terrain.py`, height map gives
  `env.terrain_z` / `terrain_normal`; never use absolute z), full DR incl. per-env foot material (bound on the tibia
  rigid body — the colliders are instance proxies), training terrain levels 0–14 incl. stairs (levels 15–20 are the
  eval-only held-out suite), left–right mirror data augmentation (`mdp/symmetry.py`, on for every arm from here).
  Never import `pxr` or Isaac Lab command/marker classes at module level in `mdp/` (train.py loads the cfg before Kit
  starts → crash). Gates: `scripts/rsl_rl/eval_s0b_gates.sh`; terrain stress sweep:
  `scripts/rsl_rl/eval_terrain_sweep.sh`; mirror check: `scripts/tests/mirror_test.py`.
- Resuming: use `--resume --load_run … --checkpoint …`; a Hydra `agent.resume=True`
  is silently overwritten by `cli_args`. Pin `env.v0_ramp_steps=[0,1]` when resuming S0a
  (the ramp is keyed to the per-process step counter).
- Isaac Lab here returns `ProxyArray`s from `robot.data.*` / sensor data: use `.torch`.
- The pre-commit `insert-license` hook is broken repo-wide (missing
  `.github/LICENSE_HEADER.txt`). Never run `pre-commit run --all-files` blindly:
  it reformats untracked-by-you files, including the specs. Use `--files`.

## Running things: tmux session 0 (user watches it live)

Run every sim/training/eval/probe command through tmux session `0`, not the Bash tool (the user attaches with
`tmux attach -t 0`). Pure-Python checks (ruff, self-tests, TensorBoard reads) may run in Bash.

- Windows: `0:1` is the Claude session itself — **never send keys to it**. `0:2` = training, `0:3` = evals/probes/
  videos (both zsh, repo cwd, `env_isaaclab` active). `0:4` nvtop, `0:5` scratch python.
- Launch: `tmux send-keys -t 0:3 C-u` (clear the line), then `tmux send-keys -t 0:3 "clear; <cmd>; touch logs/.done_<tag>" Enter`.
  `rm -f logs/.done_<tag>` first. Wait on the marker file (`until [ -f … ]; do sleep 30; done`, in a background Bash or
  Monitor) — never grep scrollback for "done" (stale matches).
- One training at a time on `0:2`; evals on `0:3` run alongside it (GPU shared, slower). Stop training with
  `tmux send-keys -t 0:2 C-c`, then wait for its `.done_` marker; the last `model_*.pt` (saved every 50 iterations) is usable.
- Gotchas: zsh does not word-split `$VAR` — use `${=CMD}`; Kit dying at start-up (exit 139, `__throw_regex_error` in the
  log) is numpy's OpenBLAS at-fork handler racing Kit's start-up fork — run with `OPENBLAS_NUM_THREADS=1` (set in
  `train.py` / `eval_s0.py`; export it for other Kit scripts); a stuck pane ("not in a mode") needs `q` / `C-u`; Kit can exit 0 after a Python
  error, so judge eval cells by their output JSON, not the exit code; sort checkpoints with
  `ls $D | grep '^model_' | sort -t_ -k2 -n | tail -1` (paths contain underscores).
- Long runs: `scripts/rsl_rl/milestone_review.sh <run_dir> <iter>` (videos + probes + reward slope every 2,000 iterations);
  stop when the return gains < ≈ 1.5 % per 2,000 iterations.

## Experimental paradigm rules (from the specs)

Stages: **S0a/S0b** recondition → **S1** blind trunk (union layout, 5+ seeds)
→ **S1b** blind batch transition at N_v × T_v → **S2** depth → **S3** RGB-D
(the main stage; budget B_S3 ≥ 1.5 × B_S2, never shortened to save time).
Arms: O (oracle, sim only), B0 (blind), B1-K (KiVi-faithful), B1-M
(KiVi-matched), B2 (foothold-attended, no predictor — the falsification arm),
F (VP-JEPA), B3 (optional generative-target control).

**Arms and comparisons**
- One shared paradigm; arms differ only by the deltas in spec §2. B2 and F
  differ by exactly {horizon-indexed queries, L_jepa, L_inv, EMA target} — do
  not let any other difference creep in. Never call B1-K a "replication" —
  write "KiVi re-implemented on our platform under our training paradigm".
- Everything in paradigm spec §1 (Invariants) is identical for every arm: robot
  USD incl. camera mast mass (blind arms too), actuator + dropped-read model,
  rewards, terrain generator/material library/transitions, dynamics
  randomisation, critic, probes, warm start, envs × batch, camera model for
  vision arms, staleness features, seeds. An arm-specific change to any of
  these is a confound, not a tweak.
- S2/S3 run at the measured vision ceiling **N_v × T_v for every arm, blind
  arms included** (N_v·T_v ≥ 24,576). N_v is **unmeasured and blocked**
  (spec §6.2); N_v = 64 is explicitly rejected.
- Actor observation layouts are **per arm and per stage** — authoritative table
  is paradigm spec §2.9 (arm F: v7 §4.9 with v2 overrides). The actor is widened
  once, at S1, to the union of all layouts except B1-K's; unused blocks are held
  at **exactly zero** so their zero-init columns stay zero. Keep layouts in one
  config-driven place; an arm must never read a block the table doesn't give it.

**Invariants that are easy to break silently (paradigm spec §1)**
- Actor inputs must be deployable: no ground-truth base velocity, no PhysX
  joint velocity (use q̇_t = (q_t − q_{t−2}) / 2Δt on ZOH-filled quantised
  positions, hold last valid q̇ if an endpoint is held, no recursive filter), no
  PhysX joint torque, no contact switches (instrumentation only — never an
  actor, critic or estimator input).
- Dropped reads: per-servo, probability rising with commanded torque/tracking
  error; ZOH of q and last valid q̇; 18-bit mask in every per-step input; one
  fill implementation shared by sim and robot.
- Toe point: `toe_pos_w = tibia_pos_w + quat_apply(tibia_quat_w, (0.095, 0, −0.09))`
  for every reward, label, anchor and `v_kin`. No hard-coded height constants.
- Friction: foot μ = 1.0 with combine mode **multiply** (or both min), never
  PhysX's default average; verify with a sim sliding test.
- Commands: `twist_cmd` training distribution v_x ~ U[0.05, 0.20] m/s, 10 %
  zero-command episodes, 30 % with small v_y/yaw; evaluation v_x ∈ {0.10, 0.15}.
- Clock: `f_ref` from the command with 0.8 Hz floor; per-env phase jitter at
  reset. The clock and every clock-phase reward term leave **together** over
  the first 30 % of S2; clock dropout is per episode (or ≥ one gait period),
  **never per step**. Surviving shaping uses phase-free, contact-conditioned
  terms (`swing_clearance`, `stance_rigidity_contact`), introduced in S0b
  (spec §5).

**Critic, probes, losses**
- Critic: **separate network** (MLP [512, 256, 128], own normaliser, nothing
  shared with the actor); input = sg[arm's actor obs] ⊕ contract **P** (683-D,
  identical for every arm, computed at step t, action-independent — material
  strips at the *nominal* foothold, never the actual touchdown). v2 §3.5
  overrides v7 §4.10. Judge arms on task metrics, never value/return curves.
- Universal probe set (186-D, spec §3.2) with ground-truth touchdown location,
  **β-NLL (β = 0.5)**, **constant weight from S1 to end of S3**, touchdown
  material terms 2×; identical for all arms (B1-K excepted by design). v2 §3.4
  overrides v7 §4.13 — no λ_priv = 0 in S1, no terminal anneal.
- Velocity estimator: residual form `v̂_b = v_kin + Δv̂` plus contact-force
  estimate `f̂`; `v_kin` stance weights from tracking error + toe-plane fit,
  **not** contact switches. V must pass v7 §4.3's slip-stratified and
  IMU-dropout tests before FAH trains.
- Auxiliary losses train from a per-env sequence buffer (L = 128) with
  episode-boundary masking, decoupled from the PPO rollout (spec §8.3).
- KL anchor to π_S0: **void since 2026-09-23** (the from-scratch control arm
  dropped the warm start; spec §4.4 kept for the record). No L_KL anywhere.
- Warm-start widening (spec §4.3): `W' = [W | 0]`; LayerNorm per new block only,
  never over the concatenated input; normaliser counts reset for blocks that go
  live at S2 and re-estimated during the actor-frozen critic re-burn-in.

**Tests that gate merges (spec §4.3, §8.2)**
- Value-loss backward leaves every actor-side grad `None`; PPO backward leaves
  every encoder grad `None`; L_probe / L_jepa leave FAH grads `None`
  (stop-gradient on p̂ into sampling); widened actor matches π_S0 to < 1e-6 over
  10,000 random observations; TCN asserts RF ≥ W. Do not merge changes that
  break them.

**Rendering and simulator**
- S0/S1 headless with cameras disabled; blind arms never render. S2:
  TiledCamera `distance_to_image_plane` (not `distance_to_camera`), colour
  intrinsics 69° × 42° at 256 × 144, rendered on camera ticks only, one shared
  depth-corruption post-process. S3: real-time RTX mode **pinned** and recorded
  in every run config; never the Interactive path-tracing mode; run the fast-yaw
  ghosting and post-reset history-bleed tests before any S3 run.
- One image operator (anisotropic resize to 128 × 96, no crop, median depth
  over valid pixels) shared by sim and robot, unit-tested on real frames.
- Isaac Sim 6.0.1 / Isaac Lab 3.0 pinned unless the user decides to upgrade to
  6.1.0 **before S0**; no version change after S0.

**Seeds, statistics, reporting**
- Seeds are paired: every arm except B1-K branches at S2 from the same S1 seed.
  B2 and F target 8 seeds, hard floor 5. Cut order if compute runs short:
  B3 → B1-K to 3 seeds → B1-M to 3 seeds.
- Budgets are fixed in env steps from a one-seed pilot of **both B2 and F**
  (the larger requirement), identical for all arms.
- Primary statistic: **paired t-test on seed-level F − B2 differences, 95 % CI**.
  Secondary: sign test, IQM with stratified bootstrap CIs (rliable),
  probability of improvement, every per-seed difference. ≥ 200 episodes per
  condition per seed, deterministic actions, common random numbers across arms.
- Report every falsification outcome in spec §11 honestly, including the
  staleness-injection test. Claims must not depend on renderer fidelity.
- **Gate order:** correctness fixes → S0 → S1 → blind **O vs B0 headroom test**;
  no vision run starts before that gate passes (spec §12).
- Anything the specs mark "measured" or "still needed from you" must be
  measured, not assumed (e.g. static torque envelope, per-surface μ, N_v, T_ro,
  Z_min, drop rate vs torque, latent_age p99, L1/R1 toe lateral offset).

## Writing reliable, robust code in this repo

This is a physics-simulation RL codebase where the dominant failure mode is not a
crash — it's a policy that *trains successfully and converges to the wrong
behavior*. `research/hexapod-RL-tuning-guide.md` documents ~20 phases of exactly
this (sliding, hovering legs, bent knees, frozen clocks, reward exploits). Treat
that history as a checklist of mistakes not to repeat:

1. **Never trust a metric without inspecting the underlying behavior.** A rising
   reward curve or "no crash" is not evidence a change worked. Before claiming a
   fix works, run a headless eval (as `research_progress.md`'s metrics section
   does) and/or `play_debug.py`, and check per-leg contact forces, joint
   trajectories, and swing/stance timing — not just scalar return.
2. **Do the economic analysis for any new reward term.** RL policies exploit
   whatever is cheapest, not what you intended. Before adding or reweighting a
   reward, work out the per-step cost/benefit of the "lazy" exploit (standing
   still, dragging, one leg permanently airborne, etc.) versus the intended
   behavior, and confirm the intended behavior is strictly cheaper. Write that
   analysis down (see Documentation below) — it's what makes the reward
   defensible later.
3. **Don't square raw simulator signals with unbounded transients** (contact
   forces, torques, impulses) — PhysX/Newton produce real single-step spikes of
   tens of N from micro-collisions, and MSE on those spikes creates catastrophic,
   policy-destroying penalties (`cpg_force_target_penalty`, Phase 10 §7). Use L1
   for physics measurements; L2 is fine for bounded, smooth state errors (e.g.
   joint position tracking).
4. **Normalize error terms to a consistent scale** before weighting them. A raw
   error measured in meters (millimeters of swing height) can be numerically
   invisible next to a `-60` height penalty even when the reward's *intent* is to
   dominate — see Phase 17's "minimal error" bug. Sanity-check the actual
   magnitude a term contributes per step, not just its configured weight.
5. **Verify overridden framework methods actually get called.** Phase 15's
   "frozen CPG clock" bug was a silently-ignored `_step_impl` override (Isaac
   Lab's `ManagerBasedRLEnv` calls `step`, not `_step_impl`) that ran for
   multiple training phases before being caught. When overriding a base-class
   hook, confirm the method name against the installed library version, and add
   a cheap runtime assertion or log line during development to prove it fires.
6. **Check tensor shapes/devices against the installed Isaac Lab API**, not
   against memory or older examples — `ArticulationData`/`ContactSensor` fields
   (e.g. `body_contact_forces` vs `net_forces_w`, history-indexed vs current
   tensors) have changed across versions and silently produce wrong indexing
   instead of errors. Grep the installed package instead of assuming a field
   exists.
7. **Prefer a curriculum/transition over a hard cutover** when swapping which
   reward terms drive the policy, and prefer physically-grounded, self-calibrating
   targets (e.g. the adaptive EMA foot baseline in `hexapod_env.py`) over
   constants derived from one rest pose — constants silently go stale under any
   posture change.
8. **When a policy gets stuck in a local optimum, don't just retrain with a new
   seed** as the first move — diagnose the exact physical/economic state the
   exploit relies on (per-leg forces, joint angles, timing) and make that state
   costly, then retrain. Retraining without a diagnosis reproduces the same
   optimum most of the time.
9. **Keep configs and reward math consistent with each other explicitly** — e.g.
   `duty_factor` must match the tripod phase offsets used elsewhere in the same
   env (Phase 11's duty-factor contradiction silently created a 66%
   double-support deadzone). When two places encode the same physical assumption,
   either derive one from the other or assert they match.

General engineering hygiene that applies on top of the above:

- No dead/speculative code paths, no silent `except: pass`, no fallback logic for
  conditions that can't occur in this environment.
- Scripts under `scripts/` and one-off analysis (`analyze_trajectory.py`,
  `scripts/rsl_rl/analyze_*.py`) are allowed to be rougher than the `source/`
  package, but must still produce numbers that make it into a progress doc only
  if they were actually run against the current checkpoint — don't report stale
  or hypothetical numbers as results.
- Random seeds, checkpoint paths, and iteration counts referenced in any results
  writeup must be reproducible from the command that was actually run; include
  that command in the progress doc entry.

## Documentation convention: phase-separated progress files

Every phase of this project keeps its history in Markdown, not just in commit
messages or chat — future readers (reviewers, collaborators, future-you) need to
know *why* a change was made and *what it fixed*, not just what the diff was.

- **`research_progress.md`** (repo root) is the running, high-level summary
  across all phases — one section per phase, each with final state reached and
  headline metrics (displacement, velocities, stability stats, gait balance).
  Append a new `## Phase N Summary` section when a phase concludes; don't rewrite
  or delete earlier phase sections.
- **`research/<topic>-progress.md`** files hold the detailed, chronological
  development log for a phase: bugs hit, root cause, fix, and — for reward work —
  the economic analysis from the checklist above. `research/hexapod-RL-tuning-guide.md`
  is the Phase 1 example; follow its format (`### Phase N: <name>` →
  `Observed Behavior` / `Root Cause` / `Fix` / result) for new phases. When a new
  phase of work starts (e.g. Phase 2 / terrain-curriculum + domain-randomization
  baseline), create a new file such as `research/phase2-progress.md` rather than
  appending unrelated work into the Phase 1 log.
- **`VP-JEPA_Architecture.md` and the Shared Training Paradigm spec** are the
  living design specs for not-yet-fully-implemented work. Update them when the
  design changes; don't let them drift out of sync with what's implemented — if
  implementation diverges from a spec, either update the spec or note the
  deviation and why in the relevant `research/phaseN-progress.md`. Do not edit
  their content otherwise without the user's say-so.

When you (the agent) fix a bug, change a reward/algorithm, or run an evaluation
with reportable results, write it into the appropriate file above in the same
session — don't leave it for the user to reconstruct from `git log` later.
