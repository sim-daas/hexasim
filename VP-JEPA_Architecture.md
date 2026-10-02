# Anticipatory Visuo-Proprioceptive JEPA (VP-JEPA)
## Consolidated Architecture Specification — v7

Supersedes `VP-JEPA_Architecture_v6_FINAL.md` and everything before it. Those remain valid as derivation history; this document is the single source of truth. Where Training Paradigm v2 conflicts, v2 governs (v2 §0.5).

**Platform:** PhantomX MK-IV, 18-DOF Dynamixel **AX-12A**, Intel RealSense D435, Jetson Orin Nano Dev Kit 8 GB.
**Simulator:** Isaac Sim / Isaac Lab, **PhysX**.

Baselines are out of scope. Two ablations are stubbed in §9 because they constrain architecture choices that must be frozen early.

Subsections are deliberately uneven in length. Settled mechanics are stated once and left alone; the parts where a wrong choice is silent and expensive get the words.

---

## 0. New Findings in v7

Eight items. The first four are training-correctness defects — each produces plausible-looking curves while quietly failing to train the thing it was supposed to train.

| # | Finding | Severity |
|---|---|---|
| 1 | **`λ_priv` is switched off before the modality it exists to supervise appears.** The future-physics probe anneals to 0 during Stage 2, but RGB — the only friction-carrying modality — only arrives in Stage 3. The mechanism that forces vision to carry physics is gone before it can act. | **Critical** |
| 2 | **Most `L_jepa` targets are invalid.** A 400 ms horizon is 20 steps; inside a 24-step PPO rollout only 4 of 24 samples have a valid target. The headline loss trains on ~17% of collected data and the longest horizon is the worst affected. | **Critical** |
| 3 | **The critic's gradient path is unconstrained.** `λ_rl_enc = 0` detaches the *actor's* PPO gradient from the encoders. It says nothing about the critic's value loss, which will flow into any shared trunk and imprint privileged-correlated structure into representations the actor consumes. | **High** |
| 4 | **`V` has a kinematic shortcut that averaged loss cannot reveal.** Leg kinematics plus gait phase predicts body velocity well under no-slip, which is ~90% of steps. A network that learns that mapping and ignores the IMU scores excellent Huber loss and fails precisely during slip — the regime the paper is about. | **High** |
| 5 | **Depth-validity can correlate with friction on hardware and never does in simulation.** The D435 loses depth on **specular and transparent** surfaces — water film, polished tile, ice — not on matte low-friction ones. Modelled unconditionally (§5.4a); the confound is then removed by surface choice rather than analysis (§5.3). Creates a new requirement on the material library (§5.4). | **High** |
| 6 | **Dropped servo reads are modelled as an actuator phenomenon, not an observation artifact.** Whatever fill policy runs on hardware never appears in the TCN input buffer during training. Drops also correlate with electrical load, therefore with rough terrain. | Medium |
| 7 | **`latent_age` as a single scalar is not commensurate with what it modifies.** Each horizon token loses lead time at the same rate but from a different starting budget; one global scalar makes the policy learn three different discount functions from one input. | Medium |
| 8 | **Stage 2 cannot learn visual physics anticipation at all**, because depth is decoupled from physics by design (§5.4). Stage 3 is therefore the main training stage, not a fine-tune, and the stage budget and renderer choice both need to reflect that. Path tracing is deleted. | Medium |

**Carried forward unchanged from v6:** the paired clock/reward anneal (§2.1), stall lock (§2.2), phase reset diversity (§2.3), the resolved leg-index and duty-factor items (§2.4, §2.6), D435 depth at 848 × 480 (§3.1), and rolling-shutter modelling (§3.2b).

---

## 1. The Claim

**Single defended contribution:** anticipatory future-proprioceptive-latent prediction, action-conditioned, placed causally inside the control loop of a legged system.

**Supporting narrative (strengthened by the hardware constraints, not weakened):** vision anticipates for the *leading* legs; the proprioceptive latent produced by those legs anticipates for the *trailing* legs, which cannot see their own footholds. A two-stage anticipation cascade that matches tripod gait physics.

**Not claimed — must be cited to prior art:**

| Component | Prior art |
|---|---|
| Multimodal vision + proprioception fusion | Fu et al. 2022, Miki et al. 2022 |
| Modality dropout / graceful fallback | Miki et al. 2022, KiVi 2025 |
| Egomotion-aligned visual memory | Neural Volumetric Memory, Yang et al. 2023 |
| Action-conditioned JEPA | V-JEPA 2-AC |
| Privileged critic, privileged probing | RMA, DreamWaQ (Nahrendra et al. 2023) |
| Proprioceptive velocity estimation | DreamWaQ, HIM |

Nearest competitor: **KiVi (2025)** — same sensors, same problem, reactive fusion, hardware results published.

---

## 2. CPG Handling — Rewritten

### 2.1 The Clock: Scaffolding, Removed on a Schedule

```python
f = torch.clamp(v_cmd[:, 0] / 0.20, min=0.8, max=2.5)   # command-referenced; 0.8 Hz stall floor
```

The clock is now referenced to the **commanded** forward velocity rather than measured `root_lin_vel_b`. Two things follow: the frequency no longer depends on a privileged quantity at all, and the reference no longer collapses when the robot stalls (§2.2). The design intent is that the clock is Stage-1 scaffolding: it bootstraps tripod structure and is then removed. That intent is correct. What matters is that removal happens in **two places, on one schedule**.

**The reward anneal alone is insufficient.** v5 §7.1 and §7.3 place the 12-dim clock in the proprioceptive encoder input and the actor observation. Zeroing `cpg_*` reward weights does not touch the observation vector, so the clock dims would survive into Stage 3 regardless.

Note that the command-referenced formula above removes the *deployability* objection — a clock driven by `v_cmd` is reconstructible onboard, unlike the previous `root_lin_vel_b` version. The reason to remove it anyway is different and still holds: the clock is a phase reference for the tripod gait, and leaving it in the observation biases the policy toward tracking that phase at exactly the point where anticipation requires departing from it. It is scaffolding, and scaffolding that stays becomes structure.

**Resolution — anneal both together.**

| Stage | CPG reward weights | Clock in observation |
|---|---|---|
| 1 | Full | **Present** |
| 2 | Annealed to §2.5 floors over first ~30% | **Input-dropout annealed 0 → 100% over the same window** |
| 3 | At floors | **Absent (dims removed or permanently zeroed)** |

Progressive input dropout rather than a hard cut: the policy learns to operate without the clock while it still has the gait in its weights, instead of facing a sudden observation-space change that invalidates the warm start.

**Why the clock must remain observable in Stage 1.** The CPG reward asks the policy to match an externally generated phase. A policy that cannot observe that phase can learn stepping *frequency* and relative tripod structure, but not absolute phase alignment — so a large part of the reward becomes unlearnable noise. This is compounded by the per-environment phase jitter in §2.3. Keep the clock visible while the reward that depends on it is active; remove both together.

**The velocity estimator `V` survives, with different justification.** It is no longer needed to reconstruct the clock. It is retained because:
- `v̂_b` is a strong policy observation in its own right (DreamWaQ, HIM).
- FAH predicts touchdown *position* in the body frame, which depends on body displacement over the remaining swing. Commanded twist is an adequate proxy on good terrain and a poor one under slip — which is precisely the regime this paper is about.

`V: z_p^t → v̂_b ∈ R³`, Huber loss against `root_lin_vel_b`, deployed on proprioception alone. If an ablation shows FAH and the policy are insensitive to it, drop it.

**Consequence for FAH (§4.5):** with the clock gone from the observation space, FAH must infer gait phase implicitly from `z_p`. A 0.5 s window of joint positions and velocities over a quasi-periodic gait does contain phase, so this is workable, but it is a real added burden on that head. Track FAH touchdown-time error across the clock anneal; if it degrades sharply, the phase information in `z_p` is weaker than assumed and FAH needs either a longer window or an explicit learned phase state.

### 2.2 Stall Lock (new)

With `min=0.2`, a stalled or backward-pushed robot sees the gait reference nearly freeze — the reward stops asking it to step at precisely the moment stepping is the recovery. Raise the floor to a gait-plausible **0.8 Hz**, or floor on `max(vx, 0.5·v_cmd)`. **Superseded by the §2.1 formula; the floor is 0.8 Hz on the command.**

This matters only while the CPG reward is active, i.e. Stage 1 and early Stage 2. That is not a reason to defer it: Stage 1 is where the gait is formed, and it is the policy you warm-start everything else from.

### 2.3 Phase Reset Diversity (new)

`_reset_idx` restores the identical `_cpg_base_offsets` for every environment, so across the whole batch the clock is phase-locked to episode start. Add a per-environment global phase offset that preserves relative tripod structure:

```python
n = len(env_ids)
jitter = torch.rand((n, 1), device=self.device) * 2.0 * math.pi
self.cpg_phase[env_ids] = (self._cpg_base_offsets.unsqueeze(0) + jitter) % (2.0 * math.pi)
```

Prevents the policy from overfitting a single phase-to-terrain-onset correlation, which would directly contaminate the lead-time measurement.

### 2.4 Leg Index Ordering — Resolved

Verified directly: contact forces were checked against actual per-leg behaviour in simulation. No action required.

One note for the record, since the two orderings are resolved by different mechanisms. `sensor.data.net_forces_w` follows the ContactSensor's `prim_paths_expr` resolution; `_tibia_body_ids` follows `art.data.body_names` scan order. The direct check covers the former. A one-line construction-time assertion is worth keeping purely as a regression guard against USD or Isaac Lab version drift — it costs nothing and would catch a silent reordering after an upgrade:

```python
expected = ["right_front", "right_middle", "right_back", "left_front", "left_middle", "left_back"]
tibia_names = [art.data.body_names[i] for i in self._tibia_body_ids]
assert all(e in t for e, t in zip(expected, tibia_names)), f"tibia order: {tibia_names}"
```

The same class of assumption applies to `phases.repeat(1, 2)` in `cpg_stance_rigidity_penalty` (v5 §1.1), which remains unverified.

### 2.5 The CPG Reward Conflict (retained from v5 §1, with Addendum 1 §A2 floors)

`cpg_z_target_reward` and `cpg_z_target_l1_penalty` penalise exactly the deviations anticipation is supposed to produce. If they carry meaningful weight in Stage 2, the headline metric goes to zero by construction.

**Anneal over the first ~30% of Stage 2, to floors rather than zeros:**

| Term | Stage 2 target | Reason |
|---|---|---|
| `cpg_z_target_*` | **10–15% of Stage 1 weight** | At zero, warm-started gaits drift into low-clearance shuffling, destroying the foot-clearance signal the lead-time metric partly reads |
| `cpg_force_target_penalty` | **retain at moderate weight — do not anneal** | Impact-force shaping is *aligned* with anticipation; peak GRF reduction is your second headline metric |
| `cpg_coxa_*` | **zero** | Gait-bootstrapping only; also resolve the contradiction below |
| `cpg_stance_rigidity_penalty` | **floor** | At zero, CoM modulation becomes indistinguishable from stance jitter in measurement |
| **clock in observation** (§2.1) | **input-dropout 0 → 100%** | Same schedule as the reward anneal. Annealing the reward alone leaves a deployment dependency in the observation vector |

Log clearance, stride length, and duty factor across the anneal. If they degrade, the floors are too low.

### 2.6 Reward Term Defects (retained from v5 §1.1)

- **`cpg_coxa_velocity_reward` and `cpg_coxa_contact_reward` contradict each other** — same joint, same phase window, opposite signs. Net behaviour is an artifact of relative weights. Pick one.
- **Both use `abs(joint_vel)`** — direction unrewarded, so a wrong-way sweep scores identically. Unbounded and summed over six legs, it is a textbook hacking vector (high-frequency coxa buzzing farms it). Use signed velocity projected on commanded heading, and clip.
- **`cpg_stance_rigidity_penalty` excludes coxa** — the very joint the contradictory terms fight over. Include it or document the exclusion.
- **`foot_z` — resolved as a defect.** The tibia-link origin plus `+0.0892` (the rest-pose origin height at first contact, measured 0.08921–0.08923 m) is exact only at rest. The toe sits at `(0.095, 0, −0.09)` m in the tibia frame (URDF `foot_link`, merged at import), so the toe-to-origin height varies by ~16 mm at 10° of tibia pitch. Use `toe_pos_w = tibia_pos_w + quat_apply(tibia_quat_w, (0.095, 0, −0.09))` in every term; delete the `0.0892` fallbacks (`rewards.py` lines 429, 559).
- **Duty factor — resolved, no defect.** Every CPG term derives the boundary as `swing_end = (1.0 - duty_factor) * 2.0 * math.pi` with `duty_factor = 0.5` default, giving `swing_end = π`. Consistent with the `{0, π}` offsets and with alternating tripod. No hardcoded `π/3` exists anywhere; the docstring's `[0, π/3)` is stale text describing a `beta = 5/6` wave gait the code does not implement. **Fix the docstring** so the next reader does not re-derive this from scratch.
- **Residual risk: `duty_factor` is a per-term parameter.** Six reward terms each take their own copy, and nothing structurally prevents one being configured differently. The failure would be silent — two terms disagreeing about whether a leg is in swing at the same phase, producing a reward landscape with no coherent gait target and no obvious signature in the learning curves. Pin it once in the env config, have every term read that single value, or assert at startup that all six agree.
- **`net_forces_w[:, :, 2]` is world-Z, not surface normal.** On the inclined terrain this paper is about, normal force ≠ vertical force. Project onto local surface normal or state the bias.
- **Stale comments ("63-dim obs", "duty factor 5/6"): fix alongside the docstring.**

---

## 3. Camera Geometry — Corrected

### 3.1 The Minimum-Depth Cutoff (correction to v5 §2)

v5's table reports the geometric FOV ground intersection but omits the sensor's minimum depth range. A ground point at horizontal distance `d` from a camera at height `h` sits at slant range `sqrt(d² + h²)`; if that is below `Z_min`, the depth return is invalid regardless of FOV.

Usable near edge: `d_usable = sqrt(Z_min² − h²)`.

**The cutoff is active only when `Z_min > h`.** The nearest possible ground point is the one directly beneath the camera, at slant range exactly `h`. If `Z_min ≤ h`, every ground point in the FOV is already beyond minimum range and the cutoff never binds — the FOV limit alone determines the near edge.

**Correction to the previous draft of this section.** It listed `Z_min ≈ 0.175 m` at `h = 0.20 m` as "invalid — no usable near ground." That is backwards. `Z_min < h` is the *good* case: the constraint disappears entirely. The error mattered because it pointed toward a D405 purchase that the geometry does not actually require.

**`Z_min` on the D435 is resolution-dependent, and lower resolution helps.** Minimum range is set by the disparity search range, so it scales down with capture resolution: roughly 0.28 m at 1280 × 720, roughly 0.175 m at 848 × 480, and lower again at 424 × 240. Since `E_v` consumes 128 × 96, you are downsampling heavily regardless, so capturing depth at 848 × 480 or 424 × 240 costs nothing you need and buys back the entire near field.

| Depth resolution | approx `Z_min` | Cutoff at h = 0.20 m | Cutoff at h = 0.25 m |
|---|---|---|---|
| 1280 × 720 | ~0.28 m | active — near edge 0.196 m | active — near edge 0.150 m |
| 848 × 480 | ~0.175 m | **inactive** | **inactive** |
| 424 × 240 | lower still | **inactive** | **inactive** |

**Conclusion: the D435 is fine.** Run the depth stream at 848 × 480 or below and mount at `h ≥ 0.20 m`, and the minimum-depth problem disappears without a D405 and without an extreme mast. Confirm on the bench — point the camera at a flat surface, walk it in, and record the closest distance returning valid depth at each resolution. Ten minutes, and it fixes the number this whole section depends on.

Front-leg footholds land at roughly **0.10–0.25 m ahead of the camera**, which the FOV covers comfortably at these heights.

### 3.2 Corrected Coverage Table

**Superseded by v2 §7.5.** This table uses the depth half-angle (29°). The RGB-aligned stack is bounded by the colour FOV (69° × 42°, half-angle 21°). Mount pitch 45–50° at h ≈ 0.22 m, not 35–40°. The table below is retained for the `Z_min` reasoning only; its pitch recommendation no longer applies.

Vertical half-FOV `α = 29°`. With `Z_min` inactive at 848 × 480 and `h ≥ 0.20 m`, the FOV intersection alone sets the near edge.

| h | pitch | near edge | far edge | front-foothold band covered? |
|---|---|---|---|---|
| 0.20 m | 40° | 0.077 m | 1.03 m | yes |
| 0.22 m | 40° | 0.085 m | 1.13 m | yes |
| 0.22 m | 35° | 0.098 m | 2.06 m | yes, more lookahead |
| 0.25 m | 40° | 0.096 m | 1.29 m | yes |
| 0.25 m | 45° | 0.075 m | 0.87 m | yes, less lookahead |

**~~Recommended: D435 at h ≈ 0.22 m, pitch 35–40°~~ — superseded, see the note at the head of this section.** Depth at 848 × 480 stands. The shallower-pitch argument does not survive the colour-FOV correction: with a 21° half-angle the near edge binds before lookahead does, so pitch goes steeper, not shallower. Going higher than 0.25 m remains unnecessary and costs CoM height on a platform with marginal actuators (§5.2).

### 3.2a Self-Occlusion by the Front Legs

Raising the mount is confirmed feasible. The front legs will enter the lower image region during swing and will sometimes occlude the foothold patch being sampled. This is real, and it has an exact solution.

**Raising the mount helps here too.** A steeper view angle looks *over* the legs rather than through them, so the same change that shrinks the min-depth dead zone also reduces self-occlusion. The two constraints point the same way, which is unusual and worth exploiting. The cost is a higher CoM on a ~2.5 kg platform with marginal actuators (§5.2) — keep the mast light, short, and rigid, and re-check the static torque envelope after mounting.

**On the compute concern: none of this touches the control loop.** Self-occlusion handling lives entirely on the 30 Hz vision path (§4.1), which has a 33 ms budget, not the 50 Hz control path with its 20 ms. `E_p` and `π` are unaffected either way. The question is only whether it fits in the vision budget, and there are two tiers at very different costs.

**Tier 1 — per-leg ray test. Effectively free; implement this regardless.**

You do not need a rendered mask to decide `m_ℓ`. You need to know whether *two specific points* — the projected footholds for L1 and R1 — are occluded. That is a ray-versus-capsule intersection test against the robot's own link geometry, approximated as capsules from FK. Two rays, ~18 capsules, a few dozen floating-point operations per frame. Microseconds, on CPU, no GPU involvement, no measurable overhead at any control rate.

This gives you the entire occlusion-aware masking behaviour with none of the cost.

**Tier 2 — full rasterized mask channel. Optional; drop if the vision path is tight.**

Rasterizing the link capsules into a 128 × 96 mask and feeding it as a sixth input channel to `E_v` lets the network *discount* occluded pixels across the whole image rather than only dropping a token. Cost is a small GPU raster, realistically 0.3–0.5 ms at 30 Hz, but it adds a channel to `E_v` and therefore a re-benchmark.

Benchmark the vision path first (action item 3). If `E_v` + FAH + sampling + `P` lands comfortably under the 15 ms budget, add the channel. If it is tight, ship Tier 1 alone — the depth-consistency test in `m_ℓ` (§4.6) already catches occlusion as a nearer surface, and Tier 1 makes that decision explicit rather than inferred. **Freeze the channel count before `E_v` training begins**; adding a channel later invalidates the encoder weights.

**In simulation this is automatic in the wrong direction if ignored:** TiledCamera renders the robot's own legs, so sim and real agree by default. The failure mode is *forgetting* to render self-collision geometry, which would produce clean sim images and occluded real ones — a silent, one-directional sim2real gap. Confirm the robot's own links are visible in the rendered depth and RGB.

**Timing note.** Occlusion is worst during swing, which is when the anticipatory sample is taken. Log the `m_ℓ = 0` rate attributable to self-occlusion separately from out-of-frame and out-of-range causes. If self-occlusion alone drives the mask to zero on a large fraction of swings, the mount geometry needs revisiting before anything else in the visual pathway is tuned.

### 3.2b RGB-Aligned Depth on the D435 — Two Consequences

Using the color-aligned depth frame is the right call: `E_v` stacks RGB and depth as channels, so they must share a pixel grid. Two things follow that are not obvious.

**1. Alignment costs host cycles — use the CUDA path.** `rs2::align` on the CPU is a per-pixel reprojection and is not cheap on an Orin Nano that is also running TensorRT and a 50 Hz control loop. librealsense built with CUDA support on Jetson accelerates alignment on the GPU; confirm your build has it and that the aligned stream is actually taking that path. Align **depth to color**, and align at capture resolution before downsampling, not after.

Alignment also creates holes — regions visible to the color sensor but not the depth sensor, plus occlusion shadows at depth discontinuities. These land disproportionately at object boundaries, which is exactly where terrain transitions are. **This raises the importance of the depth-validity mask channel**, which is already in the input (§4.4) and should be populated from the post-alignment validity, not the pre-alignment depth frame.

*If alignment proves too expensive:* you never actually need full-frame alignment for retrieval, because §5.2's sampling projects a 3D point into the image analytically. You could project `p̂_ℓ` separately into the color and depth frames using each sensor's own intrinsics and extrinsics, and sample each at its own pixel coordinate. That removes alignment entirely but breaks the stacked-channel encoder, requiring separate RGB and depth stems fused at token level. More architecture change than it is worth unless the benchmark forces it — note it as the escape hatch, not the plan.

**2. The D435's color sensor is rolling shutter; the depth imagers are global shutter.** On an oscillating hexapod this is a real, unmodelled effect. Under body angular velocity the RGB frame skews internally while the depth frame does not, so RGB and depth are misaligned *within* a frame by an amount proportional to how fast the body is rotating — worst during the pitch and roll transients that accompany foot impacts.

This is not a nuisance to randomize away, because it is systematic and correlated with motion. **Model rolling-shutter skew explicitly in simulation** (§5.6): apply a row-dependent time offset to the rendered RGB frame proportional to sampled body angular velocity, while leaving depth un-skewed. Without this, the network learns an RGB-depth correspondence in sim that is tighter than anything the hardware can deliver, and the failure will appear as unexplained degradation in the RGB pathway on the real robot — precisely the pathway whose contribution you are trying to measure.

Also log per-frame angular velocity alongside `m_ℓ`. If retrieval quality degrades with body rotation rate, rolling shutter is the first suspect.

### 3.3 The Scoping Consequence (retained from v5 §2.1–2.3)

- **Front-leg footholds:** 0.10–0.25 m ahead of camera → **current frame.**
- **Middle-leg footholds:** ~0 to −0.10 m (under body) → needs ~1–1.5 s of history.
- **Rear-leg footholds:** ~−0.20 to −0.30 m → the camera last saw that ground ~0.35 m back, which at 0.10–0.15 m/s is **2.5–3.5 s** of drift-free relative pose. Not achievable. Do not attempt.

**Therefore: retrieval is scoped to the front legs, and the entire odometry / ring-buffer / SE(3) retrieval stack is deleted.** Sampling happens in the current frame with fixed extrinsics.

Retain only a minimal **8-frame (~0.25 s) fallback buffer** for footholds pushed out of frame by sharp yaw or large pitch — rotation from IMU alone (accurate at that horizon), translation from commanded velocity, every sample flagged low-confidence. Be willing to delete it; the validity mask already provides graceful degradation, and every inference-path component removed is failure surface you do not have to defend.

**Do not add VIO.** T265 is discontinued; D435i-based VINS/OpenVINS on an oscillating hexapod costs 1–2 CPU cores and degrades under motion blur, to recover a capability the cascade already supplies.

---

## 4. Architecture

### 4.1 Rates

| Loop | Rate | Contents |
|---|---|---|
| Control | **50 Hz** (20 ms) | Bus read (~4–5 ms) → `E_p` → `V` → `π` → SYNC_WRITE (~0.6 ms) |
| Anticipation | **30 Hz** (33 ms) | `E_v` → FAH → sampling → `P` |

The anticipatory path is off the critical control path. `π` consumes the most recent available `ẑ` plus `latent_age`.

### 4.2 Proprioceptive Encoder `E_p`

Dilated 1D TCN, 4 blocks, dilations {1,2,4,8}, hidden 128. **Window 25 steps at 50 Hz = 0.5 s** (per Addendum 2 §B4 — decided deliberately, not inherited).

| Signal | Dim |
|---|---|
| Joint positions `q_meas` | 18 |
| Joint velocities (finite-differenced + filtered) | 18 |
| **Position tracking error `q_cmd − q_meas`** | 18 |
| Previous action | 18 |
| IMU linear acceleration | 3 |
| IMU angular velocity | 3 |
| Body-frame gravity | 3 |
| CPG clock, sin/cos — **Stage 1 only, annealed out (§2.1)** | 12 → 0 |
| **Per-servo read-validity mask (new, §4.2a)** | 18 |
| **Total** | **111 (Stage 1) → 99 (Stage 3)** |

Output `z_p^t ∈ R^{64}`.

The clock dims are retained in the tensor layout and driven to zero by input dropout rather than physically removed, so the warm-started weights stay valid throughout. At deployment they are permanently zero and carry no information, which is what makes the policy reproducible on hardware.

**No `Present Load`.** AX-12A load is a PWM-duty-derived 10-bit magnitude plus direction bit, heavily quantised, lagging, and contaminated by the servo's own P-controller. Tracking error is a cleaner load proxy, does not pass through the internal current estimator, is the same quantity the controller acts on, and is **free** — `q_cmd` is already in software.

**Promote the IMU in your mental model.** Slip and compliance appear in body-level acceleration and attitude response long before a hobby servo's registers notice. The IMU is the primary slip channel; the joints corroborate.

### 4.2a Dropped Reads: Fill Policy and Validity Mask

AX-12A drops are **per-servo, not per-bus** — a timeout or checksum failure tells you exactly which of the 18 failed. The architecture had masked-input machinery throughout the vision path (`m_ℓ`, depth validity) and none at all on the proprioceptive path. That asymmetry is unjustified, and the fix is the one already used elsewhere.

**Adopted: zero-order hold plus an explicit 18-dim validity mask.**

- Hold the last valid `q_meas`, **and hold the last valid velocity** rather than finite-differencing across the gap. Differencing a held position produces a spurious zero velocity, which is a lie the network has no way to detect without the mask.
- `q_cmd − q_meas` is computed from the held position and is therefore stale by one or more steps. Flagged by the same mask bit.
- Mask bit = 1 for a fresh read, 0 for a held one. One bit per servo, appended to the input vector.

**Linear extrapolation is rejected**, and your reasoning for rejecting it is correct: `q` is a 10-bit quantised encoder reading, velocity is already its first difference, and extrapolating that is a second difference of a quantised signal. That is the same operation rejected for joint acceleration in earlier drafts, and it should be rejected here for the same reason. Extrapolation also grows unbounded across consecutive drops, where ZOH degrades gracefully and self-heals on the next successful read.

**The simulation consequence is the part that was actually missing.** §5.1 modelled dropped reads as an actuator-side phenomenon and §5.6 randomised bus latency, but neither injected the resulting fill artifact into the TCN input buffer. If hardware ZOHs and simulation never produces a ZOH-flat step with its mask bit cleared, the TCN sees off-distribution input exactly when the bus is struggling.

That timing is not incidental. Drop rate on a Protocol 1.0 daisy chain rises with electrical noise and supply sag, both of which rise with torque, which rises on rough terrain. So drops concentrate in the regime the paper is about. Model drop probability as a function of commanded torque or tracking-error magnitude rather than as a uniform random rate. Same failure class as the rolling-shutter effect in §3.2b: systematic, motion-correlated, and invisible if you randomise it symmetrically.

### 4.3 Velocity Estimator `V`

`z_p^t → v̂_b ∈ R³`. Two-layer MLP. Feeds FAH (§4.5) and the actor observation. **Retained at inference.**

Justified by FAH's dependence on body displacement over the remaining swing, where commanded twist is a poor proxy under slip.

**On hardware coupling.** `V` is hardware-coupled, but not in a way that threatens the result. You are not deploying on a different robot, and RMA, DreamWaQ and HIM all deploy exactly this component. The dependence is on the sim actuator model and randomisation envelope covering the real AX-12A, which is why §5.1 is non-negotiable. If backlash, torque falloff and bus latency are faithful, `V` transfers; if they are not, `V` fails first and loudest because it reads servo dynamics most directly. That is a known, bounded risk.

**The real risk is different, and averaged validation loss will not show it.** On a hexapod running a phase-structured gait there is a strong shortcut available: stance-leg kinematics plus gait phase predicts forward velocity, *assuming no slip*. A network can achieve excellent mean Huber loss by learning that mapping and never using the IMU at all. It will then be catastrophically wrong during slip, because under slip the legs execute the same kinematics and the body does not move. That is the one regime the entire paper is about, and it is perhaps 5–10% of steps, so it contributes almost nothing to the mean.

**Structural fix — predict the residual, not the velocity.** Compute `v_kin` analytically from stance-leg kinematics (deployable onboard, no learning), and have `V` predict the correction:

```
v̂_b = v_kin + Δv̂,    Δv̂ = MLP(z_p^t)
```

The shortcut is now unavailable by construction, because the shortcut *is* `v_kin` and it has already been subtracted out. Whatever `Δv̂` learns must come from information kinematics does not contain — which is the IMU, and which is exactly the slip signal. This also makes the component interpretable: `Δv̂` is an explicit slip estimate and can be reported as such.

**Tests before `V` is trusted downstream, whichever formulation is used:**

1. **Stratify `v̂_b` error by true foot-slip velocity.** If error under slip exceeds roughly 3× the rigid-ground error, the kinematic shortcut has been learned.
2. **IMU-dropout ablation on `V` alone** at evaluation. Zero the 9 IMU dims. If `v̂_b` barely degrades, the IMU pathway is dead and `V` is a gait-phase lookup table.
3. **Slip-conditioned sample weighting on `L_vel`**, so slip episodes are not drowned out by nominal walking.

Do not ship `V` to FAH until it passes these. FAH's touchdown-position prediction under slip sits directly downstream, and that feeds the headline lead-time metric.

### 4.4 Vision Encoder `E_v`

CNN stem, not ViT-tiny — TensorRT optimises convolutions far better at this scale, Orin Nano has no DLA so everything contends for one 1024-core GPU, and locality is the right inductive bias for terrain texture.

- Input 128 × 96, **5 or 6 channels**: RGB (3), inverse-depth (1), depth-validity mask (1), and optionally the FK self-occlusion mask (1, §3.2a Tier 2) if the vision-path benchmark permits. Depth from the color-aligned frame (§3.2b), clipped to `[Z_min, 3.0]` m, with validity computed post-alignment.

**Freeze the channel count before training `E_v`.** Adding the Tier 2 mask later invalidates the encoder's first-layer weights and costs a retrain. Decide from the benchmark in action item 3, not later.
- Stem: 4 stride-2 conv blocks → 8 × 6, widths 32/64/128/192.
- Head: 3 transformer layers, `d = 192`, 3 heads, 48 tokens.
- Output: `F_t ∈ R^{8×6×64}` patch map + pooled `g_t ∈ R^{64}`.

8 × 6 over a ~0.08–1.0 m span is coarse, but foothold uncertainty is ±3–5 cm anyway, so the grid is not the binding constraint. If it proves to be, drop one stride-2 block (→ 16 × 12) and re-benchmark.

### 4.5 Foothold Anticipation Head (FAH)

**Inputs:** `z_p^t`, commanded twist, `v̂_b`, current foot positions in body frame. **No CPG clock** — gait phase must be inferred implicitly from `z_p` (§2.1). During Stage 1 the clock is still present in `z_p`'s input; FAH learns to operate without it as the anneal proceeds.

**Outputs, per front leg `ℓ ∈ {L1, R1}`:** `p̂_ℓ^{td,B} ∈ R³`, `Δt̂_ℓ`, heteroscedastic `σ̂_ℓ`.

**Supervision:** Isaac contact sensor gives ground-truth touchdown position and time. Gaussian NLL. Free labels, no privileged information at inference. **Retained at inference.**

**Why learned here is correct, unlike v4's `μ₀`:** with a reward-only CPG the foothold is genuinely emergent from the policy. There is no closed form to compute. The earlier criticism was that `μ₀` *was* analytically computable and was being learned regardless; that does not apply.

### 4.6 Sampling

Project `p̂_ℓ^{td,B}` into the current frame with fixed `T_{C←B}`; sample `F_t` bilinearly, Gaussian kernel width proportional to `σ̂_ℓ`. Large `σ̂_ℓ` widens the kernel and past a threshold zeroes `m_ℓ`.

`m_ℓ = 1` iff in-frame **and** within valid depth range **and** depth-consistent (not occluded by a nearer surface).

Bounded learned pixel residual (±2 patches, `tanh`, zero-init) absorbs extrinsics calibration error. Geometry does the work; learning only corrects.

### 4.7 Predictor `P`

- Cross-attention transformer, 3 layers, `d = 128`, 4 heads.
- Queries: learned horizon tokens, **`K = {60, 200, 400} ms` = {3, 10, 20} steps at 50 Hz.**
- Keys/values: `[v_L1 ⊙ m_L1, v_R1 ⊙ m_R1, g_t, z_p^t, a^t_ZOH, twist_cmd, v̂_b, σ̂, Δt̂]`, plus the contact-memory tokens of §4.8. No clock.
- Output: `ẑ_p^{t+k} ∈ R^{64}` per horizon.

Zero-order-hold action conditioning — adequate because a quasi-periodic gait makes near-future joint targets largely predictable from the recent action history already inside `z_p`, and it avoids rolling the policy out inside the predictor. If ZOH proves too crude at the 400 ms horizon, the cheapest upgrade is a learned short action-sequence prior rather than a policy rollout.

### 4.8 Phase-Indexed Contact Memory (Addendum 1 §A3)

The "proprioception-propagated" story for trailing legs needs something that actually implements it; a 0.5 s TCN window has no explicit representation of *"the front leg on this side hit a low-friction patch."*

Spatial indexing is impossible (§3.3). Index by **contact event** instead:

- On each front-leg touchdown, push `z_p` sampled at contact, plus elapsed `Δt`, into a short FIFO (last 4 contacts per side).
- Feed as extra predictor tokens, tagged same-side / opposite-side and with elapsed time.
- No pose estimate, no spatial index, no drift. Tripod ordering does the spatial association implicitly: the middle leg on a side lands roughly where that side's front leg landed a half-cycle earlier.

~8 extra tokens of dim 64. Present it honestly as weaker than spatial correspondence. Not load-bearing for the main claim — if it shows nothing, drop it.

### 4.9 Policy `π` (PPO, asymmetric)

**Actor observation (fully deployable):**
```
[ o_p^t , z_p^t , v̂_b , ẑ_p^{t+60} , ẑ_p^{t+200} , ẑ_p^{t+400} ,
  v_L1⊙m_L1 , v_R1⊙m_R1 , m_L1 , m_R1 , σ̂_L1 , σ̂_R1 ,
  rem_60 , rem_200 , rem_400 , twist_cmd ]
```

No clock. No ground-truth velocity. No privileged quantity of any kind. Every element is computable onboard from the AX-12A position stream, the IMU, the camera, and the commanded twist. Note `o_p^t` is the raw current proprioceptive frame — the actor already has unfiltered access to the present, and the TCN summary supplements it rather than replacing it.

**Action:** 18 joint position targets. No residual RL — correct, since hard terrain requires large departures from tripod.

**Gradients:** latents enter the actor **detached** (`λ_rl_enc = 0`) so PPO cannot collapse the latent space into a reward shortcut. This covers the actor only; see §4.10 for the critic, which is a separate and easily-missed path.

#### 4.9a Staleness: per-horizon remaining lead time

Control runs at 50 Hz, anticipation at 30 Hz, and they are asynchronous, so on most control steps `ẑ` was computed by a vision pass that finished some time ago. Age accumulates from four sources: the structural 0–33 ms rate mismatch, vision-path compute (≤ 15 ms budget, worse at p99), camera-to-host transfer and alignment, and whole dropped frames at +33 ms each. Realistic range is 0 to 80–100 ms with a long tail and a sawtooth structure, not a smooth distribution.

The reason this is not a minor scalar: `ẑ^{+60}` predicts the latent 60 ms after the *vision* timestamp, not after the current control time. With `t = t_v + age`, the effective horizon is `k − age`. At age 40 ms a 60 ms prediction is worth 20 ms of lookahead; past 60 ms it describes a state the robot has already occupied and is not anticipatory at all.

**Feed one normalised remaining-lead-time per horizon** rather than a single global age:

```python
for k in (60, 200, 400):            # ms
    rem_k = (k - latent_age) / k    # 1.0 fresh, 0.0 expired, <0 stale
    feat_k = clip(rem_k, -0.5, 1.0)
```

Three dims, each commensurate with the token it discounts, each naturally in `[0,1]` while valid. The negative region is clipped rather than zeroed deliberately — "just expired" and "badly expired" are different situations and collapsing them discards a real distinction. The three values are affine functions of one another and therefore redundant in the information-theoretic sense; that is fine and intentional, since it hands the policy the discount it needs per token instead of asking it to learn three different functions of one scalar.

**Hard cutoff on vision loss.** If the camera stalls, age grows without bound and a saturated scalar is worse than nothing. Past a threshold (≈ 150 ms, or `rem_400 < −0.5`), force `m_L1 = m_R1 = 0` and zero the `ẑ` tokens outright. The masked-input machinery for this already exists; use it rather than letting a two-second-old latent into the actor behind a clipped scalar.

**Cover the real age distribution in training.** §5.6 randomises camera-to-control latency over 0–50 ms, which is not obviously enough. Measure the hardware age distribution including Orin thermal throttling, and make the training distribution cover its p99. If sim ages cluster at 0–40 ms and hardware p99 is 90 ms, the policy has never seen its worst operating condition.

### 4.10 Critic (privileged, simulation only)

This section is longer than its size in the codebase warrants, because three of the failure modes here are silent.

**Inputs:** actor observation + true body velocity + ground-truth friction and stiffness at the current and at the **actual** touchdown location + true contact forces + local height field.

**Query the true touchdown location, not FAH's prediction.** The v6 spec queried privileged physics at the *predicted* touchdown point, which makes the critic's input distribution a function of a concurrently-training head. As FAH's predictions shift over training, the set of ground-truth samples the critic sees shifts with them — a non-stationary critic input on top of PPO's already non-stationary value target. The coupling is second-order and might wash out, but if it does not, it presents as generic PPO variance and is very hard to attribute.

The fix is to remove the coupling rather than monitor it. The critic is privileged and sim-only, so it has no reason to be restricted to what FAH knows: query the physics at the location where the foot **actually** lands. This is available retrospectively once the rollout is collected, and the trajectory buffer needed for it is the same buffer §4.12 requires for valid JEPA targets, so it costs nothing extra. If backfilling proves awkward, the nominal kinematic touchdown point (CPG phase plus commanded twist) is a stationary second choice — less accurate, still fully decoupled.

**Keep the diagnostic anyway:** log critic value-loss variance segmented by `σ̂_ℓ`. Under the true-touchdown formulation it should show no relationship. If it does, something still couples them and the assumption needs re-examining.

**No shared trunk with the actor.** `λ_rl_enc = 0` detaches the actor's PPO gradient from `E_p` and `E_v`. It does nothing about the critic's value loss, which will flow into any shared encoder and imprint privileged-correlated structure into representations the actor consumes — quietly reintroducing the privileged information the whole asymmetric design exists to keep out. This is among the most common asymmetric actor-critic bugs in legged RL codebases and it must be checked explicitly rather than assumed.

Use a **fully separate critic network** with its own encoder over the raw privileged and proprioceptive state. The critic never deploys, so parameter count and latency are free, and separation removes the question entirely instead of relying on a detach being placed correctly.

**Privileged observation set is a fixed contract.** Whatever privileged inputs the critic receives must be identical across every arm that gets compared. A richer privileged critic for VP-JEPA than for a baseline confounds representation quality with value-estimation quality, and it is the kind of thing a careful reader finds in the appendix in under a minute.

**A very informative critic can hide the predictor's contribution.** If the critic already knows what terrain is coming, its advantage estimates are clean whether or not the actor anticipated, so training curves across arms will look similar even when behaviour differs substantially. Judge arms on task metrics — lead time, peak GRF, success rate — never on value loss or return curves relative to a privileged critic. A converged critic loss curve is not evidence that two arms are equivalent.

### 4.11 Losses

```
L = λ_jepa·L_jepa + λ_inv·L_inv + λ_priv(t)·L_priv + λ_fah·L_fah + λ_vel·L_vel + λ_vic·L_vic + L_PPO
```

- **`L_jepa`** — Σ over `k` of smooth-L1 between `ẑ_p^{t+k}` and `sg[z̄_p^{t+k}]` from the EMA target (momentum 0.996 → 0.9999, training only). **Target validity is not automatic — see §4.12.**
- **`L_inv`** — inverse dynamics, predict `a^t` from `(z_p^t, z̄_p^{t+k})`. Anti-collapse anchor.
- **`L_priv`** — two terms. Current-state probe from `z_p^t`; and, **weighted 2×**, a **future** probe predicting physics at the predicted touchdown location from `ẑ_p^{t+k}`. The future probe is what forces the visual pathway to carry physics — without it, proprioceptive extrapolation alone satisfies the objective and vision becomes ornamental. **Schedule corrected — see §4.13.**
- **`L_fah`** — Gaussian NLL on touchdown position and time.
- **`L_vel`** — Huber on `v̂_b` (or on `Δv̂`, §4.3) against ground truth, **slip-weighted** so that slip episodes are not drowned out by nominal walking.
- **`L_vic`** — VICReg variance/covariance on `z_p` batch statistics.

**Discarded at inference:** `E_p'`, inverse-dynamics head, both probers, the critic.
**Retained at inference:** `E_v`, `E_p`, `V`, FAH, sampling, `P`, `π`.

### 4.12 JEPA Target Validity — Decouple the SSL Losses from the PPO Rollout

A 400 ms horizon is 20 steps at 50 Hz. Inside a 24-step PPO rollout, only 4 of 24 samples have a target 20 steps ahead. The headline loss would train on roughly 17% of collected data, and the longest horizon — the one carrying the strongest anticipation claim — would be the worst served. Shorter horizons fare better but are still truncated.

Extending the rollout to ≥ 64 steps fixes the arithmetic but changes PPO's bias-variance behaviour for reasons unrelated to representation learning, which is a poor trade.

**Adopted: a separate FIFO transition buffer for the self-supervised losses.** `L_jepa`, `L_inv`, `L_priv` and `L_vel` do not need on-policy data — they are supervised or self-supervised objectives over observed transitions, and slight off-policy staleness is harmless for them. Keep PPO's rollout length wherever it performs best, and sample SSL minibatches from a rolling buffer that spans rollout boundaries. Every sample then has a valid target at every horizon.

**Mask targets across episode termination regardless.** A target that crosses a fall, a reset, or a terrain-instance change is not a prediction of anything. Store an episode id per transition and drop any `(t, t+k)` pair spanning a boundary. Without this the longest horizon quietly trains on nonsense at exactly the reset events that are most common early in training.

### 4.13 `λ_priv` Schedule — Corrected

v6 cosine-annealed `λ_priv` to zero across Stage 2. Stage 2 is depth-dominant, and depth is decoupled from physics by design (§5.4). RGB — the only friction-carrying modality — arrives in Stage 3. So the probe whose entire purpose is to force the visual pathway to carry physics was being switched off before the modality that can carry it existed. The mechanism never fires.

**Corrected schedule:**

| Stage | `λ_priv` | Reason |
|---|---|---|
| 1 | 0 | No vision yet; nothing to ground |
| 2 | moderate, constant | Grounds the *proprioceptive* path and shapes `z_p` geometry. Cannot ground vision here, and should not be expected to |
| 3, first ~60% | **maximum, held** | RGB is being introduced; this is the only window where the future probe can force the visual pathway to carry friction |
| 3, final ~40% | cosine → 0 | Release the scalar constraint so the latent generalises out of distribution rather than overfitting to regression targets |

The principle: `λ_priv` must be high while the modality it is supervising is present and still plastic, and annealed only after that. Tying its schedule to a stage index rather than to modality introduction was the error.

---

## 5. Simulation

### 5.1 Actuator Model — Non-Negotiable

The torque proxy depends entirely on this being faithful. An idealised sim servo will teach adaptations that do not exist on hardware, and sim2real will fail for reasons unrelated to vision.

- P-only position control at the **measured** AX-12A gain, with Compliance Margin/Slope behaviour
- Torque saturation at ~1.5 N·m stall, derated for continuous operation and thermals
- Velocity-dependent torque falloff
- Position quantisation to 10 bits (0.29°/tick)
- Backlash ±0.3–1.0° (plastic gear train — note this is the same angular scale as the joint deflections that signal surface compliance)
- Deadband
- **Bus round-trip latency 4–12 ms with jitter**
- **Dropped reads, per-servo and load-correlated (§4.2a).** Protocol 1.0 has no free retry semantics. Drop probability should scale with commanded torque or tracking-error magnitude rather than being a uniform random rate, because supply sag and electrical noise both rise with load, which rises on rough terrain. The ZOH fill and its cleared mask bit must be written into the observation buffer, not merely modelled as an actuator effect.

Expose `q_cmd − q_meas`. Do **not** expose PhysX joint torque.

### 5.2 Actuator Saturation vs. the Claim (Addendum 2 §B5)

The anticipatory behaviours you intend to show — lowering CoM, shortening stride, extending stance — are the most torque-expensive gait modifications available. Lowering the body increases femur/tibia moment arms and raises static holding torque across all stance legs at once. On a ~2.5 kg PhantomX with three legs sharing load in tripod stance, the femur joint is already near its limit at nominal posture.

**Risk:** the policy may be physically unable to execute the clearest anticipatory response and will instead pick whatever adaptation is cheap — possibly too subtle to measure.

**Actions:** compute the static torque envelope across the intended CoM range *before* training; make **stride shortening and stance-duration extension** the primary reported adaptations (cheaper in torque, equally diagnostic), with CoM lowering secondary; keep `cpg_force_target_penalty` at moderate weight, which is both aligned with the objective and kinder to the actuators.

### 5.3 Terrain Conditions (Addendum 1 §A1)

PhysX friction randomisation is exact and cheap. **Compliance is not** — yielding terrain is approximated by compliant-contact stiffness/damping, a lumped linear model. Reasonable for foam or soft rubber. Poor for vegetation, which yields anisotropically, buckles, and exerts sustained lateral drag on the tibia rather than a normal-direction spring force.

- **Drop tall grass and vegetation from the framing entirely.** Inherited from the KiVi/Miki literature; not credibly simulable in PhysX, therefore not claimable.
- **Defensible claim: variable friction and variable contact compliance.** Use characterised foam of known stiffness.
- **Validate before claiming compliance transfer:** measure force-displacement on the real foam, fit PhysX contact stiffness/damping to it, report the fit. Unvalidated compliance transfer is the easiest thing for a reviewer to attack, because they can see you never measured it.
- Friction is unaffected by the PhysX compliance limitation and is your strongest condition. If foam characterisation proves expensive, make friction primary and compliance secondary.

**Low-friction surfaces — decided.**

| Role | Surface | Optical character | Why |
|---|---|---|---|
| **Primary** | Matte low-friction (virgin white PTFE sheet; dry talc or graphite over a rigid substrate) | Diffuse — stereo matching works normally | Depth validity stays uncorrelated with friction, so the depth-only arm remains a clean geometric lower bound and the RGB ablation is interpretable |
| **Secondary** | Specular low-friction (water film, polished tile) | Specular — depth drops out | Ecologically the more common real case, and the only condition where the validity cue is available. Reported separately, never pooled with the primary |

This ordering is what makes the depth-validity question tractable rather than confounding: the headline result is measured where validity carries no friction information, and the specular case becomes a separate, honest observation about a cue the sensor supplies for free.

**Grading the gradual condition — trade-offs, since this is the discriminating case.**

| Approach | Durable across trials | Continuously gradable | Visual subtlety |
|---|---|---|---|
| PTFE sheet | yes | no — it is an edge | none; a white sheet on grey concrete is a sharp albedo boundary. **This is a sharp-condition surface, not a gradual one** |
| Graded talc or graphite | **poor** — displaced by feet, friction drifts across trials | yes, by coverage density | high — subtle albedo/texture change, large friction change |
| Graded grit (bonded abrasive) | yes | yes | **low** — roughness is directly visible, so RGB↔friction is strong and the task gets easier |

Powder grades best and photographs best but will not survive 10–20 trials without re-application; if you use it, re-apply and re-measure friction between trial blocks and report the measured spread. Bonded grit is the repeatable option but makes the gradual condition visually easier than intended, which weakens exactly the claim it is meant to support. Decide with eyes open; there is no option that is simultaneously durable, gradable, and visually subtle.

**Friction must be measured per surface.** Hardware `μ_s` and `μ_d` come from a tilt test and a drag test on the robot itself (servos powered, rest pose), 5 repeats each, reported as mean ± SD. Generic handbook values are not acceptable for defining transition conditions. The simulator's friction range must contain every measured hardware value. PhysX combine mode must be **multiply** (foot `μ = 1.0`) or **min**, never the default **average**.

**Secondary result (TBD): held-out harder-terrain suite.** Placeholder agreed 2026-09-22, finalised manually later.
Terrain levels beyond the training ceiling (rough to ±3 cm, slopes to ≈ 20°, bumps and stairs to 4 cm, at and over the
3.8 cm swing height) are evaluated but never trained on, identically for every arm; see paradigm spec §10.4. F vs B2
is the fair comparison there. Geometry stays sampled independently of friction and compliance (§5.4), so this is a
separate, secondary question from the friction claim.

**Two transition conditions, both reported.** The observation that real environments span this range is correct and should be made explicit in the paper rather than left implicit:

| Condition | Description | What it tests |
|---|---|---|
| **Sharp boundary** | Clear visual edge — the man-made case (tile to spilled liquid, concrete to painted metal) | Ecological validity. A vision baseline and a naive edge-detect heuristic will also do well here, so it is not where the contribution is proven |
| **Gradual transition** | Slow, visually subtle change with no detectable edge — the natural/rural case | **This is the discriminating condition.** Texture statistics rather than boundaries carry the signal, which is what defeats the edge-detect heuristic |

Report both. The sharp case shows the system works where the field already claims results; the gradual case is what separates VP-JEPA from an expensive edge detector, and it is the one a reviewer will look for. Randomise transition sharpness continuously in simulation so the policy is never trained on a single boundary profile.

A third case worth including cheaply: a **visually distinct boundary with no physical change** (painted line on uniform terrain), and its inverse, a **physical change with no visual cue**. The first catches a policy that has learned to react to edges rather than physics; the second bounds how much of the effect is vision at all.

### 5.4 Appearance Factorisation (modality-laziness fix)

**Randomise hard (nuisance, physics-uncorrelated):** illumination intensity and direction, colour temperature, exposure, white balance, gamma, per-channel gain, sensor noise, motion blur, vignetting, lens distortion residual.

**Do not randomise (physically bound):** albedo, roughness, normal micro-structure, specularity — sampled *jointly* with friction and compliance from a material library. Wet looks wet and is slippery.

**The material library must break the "shiny implies slippery" heuristic, and this is now load-bearing.** The intuitive library — wet, ice and polished metal as the low-friction entries; concrete, rubber and grit as the high-friction ones — makes specularity an almost perfect friction predictor. A network trained on it will learn the shortcut, and it will then fail on your **primary** hardware test surface, because matte white PTFE is diffuse and slippery. That is not a subtle degradation; it is the headline condition returning the wrong answer with confidence.

The library therefore has to be built to break the correlation deliberately:

- **Matte and slippery** — PTFE, dry talc, graphite, fine dry dust, smooth diffuse plastic. Without these the shortcut is unavoidable.
- **Shiny and grippy** — wet rubber, polished-but-high-friction coatings, glossy paint over grit.
- Plus the intuitive combinations, so the correlation is weakened rather than inverted.

The discriminating cue must end up being fine-scale texture statistics and chromatic character, not gloss. Verify it: sample held-out materials with specularity and friction independently assigned, and check that the friction probe's accuracy does not collapse on the off-diagonal pairs. If it does, the library is still teaching gloss.

**No grayscale conversion.** Chromatic cues (wet vs. dry, ice vs. concrete) are real friction signal.

**Critical:** terrain **height field sampled independently of friction/compliance**, so `depth_geometry ⟂ physics` and `RGB ⊥̸ physics`. Without this the network shortcuts the visual problem through geometry and the RGB pathway never becomes load-bearing.

#### 5.4a Depth *Validity* Is Not Depth Geometry — and It Leaks Friction

A real D435 loses depth returns on **specular and transparent** surfaces — a water film, polished tile, polished metal, ice. It does **not** meaningfully drop out on matte low-friction surfaces: a virgin white PTFE sheet is diffuse, and dry talc or graphite powder is diffuse, so stereo matching works normally on both.

This distinction matters more than it first appears, because it means the problem is created or avoided by your choice of test surface (§11, question 2), not forced on you. Where dropout does occur, it lands on low-friction conditions, so on hardware the depth-validity mask becomes correlated with friction — while in a simulator returning clean depth on every surface it never is.

This is a one-directional sim2real gap landing exactly on the headline condition, and it cuts both ways: the network never learns that depth dropout implies specularity implies slippery, and it simultaneously learns to trust depth everywhere, so it is most confidently wrong at the transition.

**Model specular dropout regardless of which surfaces you test on.** It exists on hardware, and a policy that has never encountered a depth hole will behave unpredictably the first time it meets one — a puddle, a polished floor, a wet patch outside the experimental protocol. Make per-pixel depth-validity probability a function of the material's specularity and wetness, which §5.4 already samples jointly with friction. This is cheap, it removes a real sim2real gap, and it is not contingent on the experimental design.

**What the surface decision changes is not whether to model it, but how the results read.** With a **matte primary** surface (§5.3), depth validity carries no friction information in the headline condition, so:

- **Depth geometry** — the range values — stays decoupled from physics, as §5.4 requires.
- **Depth validity** — whether a return exists — is the depth sensor responding to *surface optics*, the same property RGB responds to. It is an appearance channel arriving on the depth stream, not a geometric shortcut. In the primary condition it is uninformative about friction; in the secondary specular condition it is informative.

Two consequences, both to be stated in the paper rather than found by a reviewer:

1. **The depth-only arm is a clean geometric lower bound in the primary condition and is not in the secondary one.** Report the two separately and never pool them. This is the whole reason the matte surface is primary.
2. **In the secondary specular condition, some of the effect may ride on validity dropout rather than RGB texture.** That is a legitimate and interesting observation — the sensor supplies a free friction cue — but it is a different claim from the one the paper makes, and it must be labelled as such. The painted-line and invisible-transition controls in §5.3, plus a matched matte/shiny pair at equal friction, separate them.

### 5.5 Modality Dropout

Robustness engineering — cite KiVi and Miki et al., claim nothing. RGB zeroed 10% of episodes; depth zeroed/corrupted 10%; `z_p` zeroed **in the predictor input only** 10% (never in the policy input — the control loop cannot run blind to proprioception). Naturally-occurring `m_ℓ = 0` supplies additional structured masking for free.

### 5.6 Dynamics Randomisation

Friction (static/dynamic), contact stiffness, damping, restitution, payload mass and CoM offset, servo `k_p` and compliance slope, **bus latency 4–12 ms with jitter and dropped reads**, joint friction and backlash, IMU bias/noise, camera extrinsics ±2° / ±5 mm, camera-to-control latency jitter 0–50 ms with random frame drops.

**Plus rolling-shutter skew on RGB only (§3.2b).** Row-dependent time offset proportional to body angular velocity, applied to the rendered color frame while depth stays global-shutter. This is systematic rather than random — it is a fixed property of the D435's color sensor — so it must be *modelled*, not randomized around. Randomize the readout time within the sensor's plausible range; do not randomize whether the effect exists.

### 5.7 Training Stages

| Stage | Content | Renderer |
|---|---|---|
| **1** | Blind asymmetric actor-critic. Actor warm-started from the existing CPG-reward policy; critic from scratch with actor frozen during burn-in so early advantage noise cannot destroy the converged gait. CPG terms at full weight, **clock observable**. Velocity estimator trained here. `λ_priv = 0`. | none |
| **2** | Depth-dominant. **CPG reward terms annealed to the §2.5 floors and the clock annealed out of the observation, on the same schedule.** JEPA, inverse-dynamics and FAH losses come online. `λ_priv` moderate and constant. | TiledCamera depth, RTX Real-Time |
| **3** | **The main stage.** RGB introduced progressively. `λ_priv` at maximum and held through the first ~60%, then annealed (§4.13). **Clock dims permanently zero** — observation space is now exactly the deployment observation space. | RTX Real-Time, MDL materials |

**Stage 2 is a bootstrap, not a co-equal stage, and the budget should reflect that.** Depth geometry is decoupled from physics by design (§5.4), so no amount of Stage 2 training can teach visual *physics* anticipation — there is no signal to learn. What Stage 2 does deliver is real but narrower: the JEPA predictor learns to use geometric context, FAH learns touchdown prediction, the retrieval and masking paths get exercised, and `z_p` acquires structure. Treat it as scaffolding for Stage 3 and allocate compute accordingly. If Stage 3 turns out to be short relative to Stage 2, the friction claim is undertrained no matter how good the Stage 2 curves look.

**Path tracing is deleted.** RTX Interactive Path Tracing cannot produce frames at the rate on-policy RL data collection requires, and the visual fidelity gap to RTX Real-Time is not what limits transfer here — photometric randomisation and the rolling-shutter and validity models (§3.2b, §5.4a) are. Spend the compute on environment count and Stage 3 length instead.

Stage 3's observation space must be byte-identical to what runs on the robot. Any quantity still present at the end of Stage 3 is a quantity you have committed to producing onboard.

---

## 6. Deployment and Compute

**Onboard only.** All reported results run on the robot; offboard streaming is development logging. The headline metric is a tens-of-milliseconds lead time, and wireless jitter is variable enough to confound it irrecoverably.

### 6.1 Execution Split

| Path | Components | Device | Rate | Budget |
|---|---|---|---|---|
| Control (critical) | `E_p` + `V` + `π` | **CPU** (ONNX Runtime / oneDNN), isolated core | 50 Hz | ≤ 1.5 ms |
| Anticipation | `E_v` + FAH + sampling + `P` | GPU (TensorRT) | 30 Hz | ≤ 15 ms of 33 ms |

Putting the control loop on **CPU** is the key move: a 4-block TCN plus two MLPs is sub-millisecond on the A78AE cores, and it avoids GPU kernel-launch overhead and sync jitter (~0.2–0.5 ms per launch, long tail under contention). Vision can overrun a frame without the control loop missing a deadline.

### 6.2 Configuration

- `E_v`: INT8 (PTQ, a few hundred calibration frames). FAH and `P`: FP16. Control path: FP32 on CPU.
- **MAXN / 15 W + `jetson_clocks` + active cooling.** The 7 W default will not meet these numbers.
- RealSense **D435, depth at 848 × 480 @ 30 Hz** (keeps `Z_min` inactive, §3.1), color-aligned, then downsampled to 128 × 96. Stereo matching runs on the D4 ASIC, so host cost is USB transfer, **alignment**, and librealsense — pin all of it off the control core.
- **Verify librealsense is built with CUDA** and that `rs2::align` is using it (§3.2b). CPU alignment at 848 × 480 is the single most likely source of unexpected host load in this pipeline.
- Memory is a non-issue (< 30 MB of models). The constraint is latency and GPU contention, not VRAM.
- Control loop in C++ with pinned buffers. Python/ROS glue will cost more than the network.

### 6.3 Bus Configuration — Mandatory

Return Delay Time = 0 on all 18 servos. FTDI latency timer = **1 ms** (the 16 ms default destroys the loop on its own). 1 Mbps baud. **Read `Present Position` only** (2 bytes) — the tracking-error substitution means you no longer need load off the bus, which shortens each status packet from 12 to 8 bytes and brings 18 reads to ≈ 4–5 ms. Plus SYNC_WRITE ≈ 0.6 ms.

**100 Hz is marginal with a bad p99 tail. 50 Hz is the design point.** Measure before committing.

---

## 7. Lead-Time Instrumentation

Servo-current onset is unavailable on AX-12A and is eliminated. **Contact switches are confirmed as a hardware addition**, which changes lead time from inferred to measured and makes this section far stronger than it was.

1. **Foot contact switches — primary, confirmed.** Microswitches or FSRs on the tibia tips, read on a small MCU alongside the IMU, hardware-timestamped on the same monotonic clock as everything else. **Instrument all six legs, not just the front two.** The marginal cost over two is trivial, and the trailing legs are what validate the proprioception-propagated half of the cascade (§3.3) — without them that claim is asserted rather than shown. They also give you real stance detection, which improves the velocity estimator's validation, FAH's hardware validation, and every gait statistic you report.
2. **High-speed video — validation and backup.** 240 fps phone camera = 4.2 ms resolution against a 60–400 ms effect. Use it once to characterise switch latency and debounce behaviour, then rely on the switches for volume. Sync via a trigger LED in frame.
3. **Instrumented terrain — independent cross-check.** Force plate or piezo mat under the transition. With switches in place this is no longer load-bearing, but an independent contact timestamp that does not share the robot's clock or failure modes is worth having for the headline result.
4. **Tracking-error onset — free corroboration.** `q_cmd − q_meas` on the front tibia steps at contact. Now that switches provide ground truth, characterising this becomes cheap and gives a second onboard channel. Unlike `Present Load` it does not route through the servo's current estimator.

**Switch integration notes:** debounce in hardware or with a fixed-latency digital filter and characterise that latency once — a variable-latency debounce will smear the metric exactly like software timestamping would. Sample the switches faster than the control loop (≥ 500 Hz on the MCU) so contact timing is not quantised to 20 ms, which would be a third of your shortest horizon.

All logs on a single monotonic clock. Software timestamps alone will smear a 60 ms effect.

---

## 8. Standing Diagnostics

Each of these detects a failure that produces normal-looking training curves. Log them from the first run, not after something goes wrong.

| Diagnostic | Detects | Failure signature |
|---|---|---|
| **`latent_age` × lead-time joint distribution** | Whether the predictor causally reaches the control loop | A *negative* correlation is expected and reportable. **No correlation at all is the alarming case** — it means the policy is ignoring `ẑ`, which would invalidate §1 |
| `v̂_b` error stratified by true foot-slip velocity | `V` kinematic shortcut (§4.3) | Slip error > ~3× rigid-ground error |
| IMU-dropout eval on `V` | Dead IMU pathway | `v̂_b` barely degrades |
| Critic value-loss variance segmented by `σ̂_ℓ` | FAH/critic coupling (§4.10) | Any systematic relationship under the true-touchdown formulation |
| `m_ℓ = 0` rate split by cause | Mount geometry vs. occlusion vs. range | Self-occlusion dominating during swing |
| Retrieval quality vs. body angular velocity | Rolling shutter (§3.2b) | Degradation rising with rotation rate |
| Per-servo drop rate vs. commanded torque | Whether sim drop model matches hardware | Hardware correlation steeper than sim |
| FAH touchdown-time error across the clock anneal | Whether phase survives in `z_p` (§2.1) | Sharp degradation as clock dims fade |
| Clearance, stride length, duty factor across the CPG anneal | §2.5 floors set too low | Drift toward low-clearance shuffling |

---

## 9. Deferred Ablations That Constrain Architecture

Baselines and the full ablation grid are out of scope here. These two are listed because they change decisions that must be frozen before training, so they cannot be deferred to write-up time.

**A. Proprioceptive receptive field.** Window is currently 25 steps (0.5 s) with dilations {1,2,4,8}. If 0.5 s proves insufficient — most likely signal is FAH touchdown-time error degrading once the clock is annealed out — escalate in cost order:

1. Extend dilations to {1,2,4,8,16} → RF 63 steps, window 50 (1.0 s). One line; everything else unchanged.
2. Add a slow branch: a second TCN over a decimated stream (every 4th step), concatenated at the output. Gives multi-second context at roughly a quarter of the cost of widening the fast branch.
3. Widen the fast branch. Last resort — it scales control-path latency, which is the one budget with no headroom.

**B. Raw history stack in the actor observation.** The actor already receives `o_p^t`, the raw current frame. The ablation is whether adding `t−1, t−2` helps: the TCN summary is lossy and biased toward whatever `L_jepa`/`L_inv`/`L_priv` care about, which is not necessarily what the actor needs for precise instantaneous control, and a short raw stack hedges against that at ~100 extra dims. Standard practice in legged RL. Run it as an ablation with ≥ 5 seeds rather than adopting it by default — this is a platform where PPO seed variance is already expected to be high, and adding observation dimensions on a hunch is how that gets worse.

---

## 10. Action Order

0. **Static torque envelope at the rest pose (§5.2).** It gates S0a.

**Correctness before anything else — these four are silent failures and all are cheap.**

1. **Verify no shared trunk between actor and critic (§4.10).** Separate critic encoder, or an explicit second detach. One afternoon; otherwise every subsequent result is suspect.
2. **Move the SSL losses to a decoupled transition buffer with episode-boundary masking (§4.12).** Until this is done `L_jepa` is training on a fraction of the data and the 400 ms horizon barely trains at all.
3. **Fix the `λ_priv` schedule to track modality introduction, not stage index (§4.13).**
4. **Switch the critic's privileged physics query to the actual touchdown location (§4.10).**

**Bench measurements — three, one day, and they set parameters everything else depends on.**

5. **Measure D435 `Z_min` at 848 × 480 and 424 × 240 (§3.1).** Flat surface, walk it in, record the closest valid return. Ten minutes.
6. **Measure the Dynamixel bus round-trip** (position-only, RDT = 0, FTDI = 1 ms) against §6.3. Sets control rate and therefore the horizons. **Log per-servo drop rate against commanded torque** while you are there — it calibrates the §5.1 drop model.
7. **Benchmark the vision path on the Orin** — `E_v` + FAH + sampling + `P` as a dummy TensorRT graph at exact shapes, p50/p99 under thermal load, with CUDA alignment running and a synthetic 50 Hz CPU loop alongside. **This decides the §3.2a Tier 2 channel**, which must be frozen before `E_v` training. **Record the resulting `latent_age` distribution** — §4.9a needs its p99.

**Then, in the environment:**

8. **Implement the ZOH fill and 18-dim validity mask, in sim and on hardware, with the same policy (§4.2a).**
9. **Fix the contradictory coxa terms; replace the `foot_z` origin offset with the toe-frame transform (§2.6).**
10. **Pin `duty_factor` to a single source and assert all six terms agree (§2.6).** Fix the stale docstring.
11. **Implement the paired anneal (§2.1, §2.5)**, stall lock (§2.2), phase reset diversity (§2.3).
12. **Add rolling-shutter skew and specularity-driven depth dropout to the render path (§3.2b, §5.4a, §5.6).**
13. **Confirm self-collision geometry renders in TiledCamera (§3.2a).**

**In parallel, hardware:**

14. **Bench-characterise AX-12A tracking error against known external load**; compute the static torque envelope (§5.2) including mast mass.
15. **Add contact switches on all six legs (§7)**, MCU-sampled at ≥ 500 Hz on the shared monotonic clock.
16. **Mount the camera** at h ≈ 0.22 m, **pitch 45–50°** (v2 §7.5, §3.2), rigid and light, extrinsics properly calibrated. Mount the D435 laterally offset so its **colour optical centre lies on the sagittal plane** (not the housing centre): mirror augmentation (v2 §1, D35) is then physically exact for RGB, and the sim depth corruption runs after mirroring.
17. **Build the gradual-transition surface (§5.3).** Procurement lead time; start now.
18. **Per-surface `μ` measurement (§5.3).** Tilt and drag tests on the robot, 5 repeats each, mean ± SD.

**Then train:** `V` first, and do not proceed past it until it clears the three tests in §4.3. FAH next, then the predictor.

---

## 11. Open Questions

Three. The first is a construction question; the other two are decisions with deadlines.

1. **How much of the gradual-transition surface can you build, and by which grading method?** (§5.3) Still the only item that can change the shape of the result rather than a parameter inside it. Friction remains the right choice over compliance — PhysX simulates it exactly, so it carries the least unvalidated transfer risk. The live sub-question is the trade-off table in §5.3: powder grades continuously and stays visually subtle but will not survive a trial block; bonded grit is repeatable but makes the gradual case visually easier than the claim wants; a PTFE sheet is a sharp boundary, not a gradual one. No option is durable, gradable and subtle at once, so this is a judgement about which weakness you would rather defend in review.

2. **Training compute for S2/S3.** `N_v` on the A4000 is unmeasured, and 64 envs is not viable (v2 §6.2).

3. **Isaac Sim 6.0.1 vs 6.1.0.** Decide before S0 or never (v2 §7.4).

---

## 12. Resolved Decisions (record)

**Low-friction test surfaces — matte primary, specular secondary.** Recorded in §5.3. The reasoning, kept because it will need restating in the paper: the D435 fails to return depth on shiny and wet surfaces, so on a specular test surface "there is a hole in my depth image" is itself a reliable slipperiness cue. A network can learn that shortcut instead of reading appearance, the lead-time numbers still look good, and the RGB ablation goes flat — removing RGB barely hurts, because the depth hole still carries the signal. On a matte surface depth returns normally, the shortcut does not exist, and the only pre-contact cue is appearance, which is the claim. Hence matte PTFE (or talc/graphite over a rigid substrate) as the primary condition, with wet and polished surfaces reported separately and with the extra cue stated plainly.

**Worth one trial pair regardless:** the same slipperiness on a matte and on a shiny surface. If anticipation holds on matte, texture is doing the work. If it only holds on shiny, depth holes were doing the work all along — better to find that yourself than to have a reviewer find it.

**Specular dropout is modelled in simulation unconditionally** (§5.4a), independent of which surfaces are tested, because it exists on hardware and an unprepared policy will behave unpredictably at its first depth hole — a puddle, a wet floor, anything outside the protocol.

**Consequence carried into §5.4:** the material library must contain matte-slippery and shiny-grippy entries. The intuitive library makes gloss an almost perfect friction predictor, and a network that learns that shortcut fails on the matte primary surface — the headline condition, answered confidently and wrongly.

Everything else is a bench number, a code fix, or an implementation task.