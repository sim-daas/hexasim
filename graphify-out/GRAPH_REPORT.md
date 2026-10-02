# Graph Report - hexasim  (2026-09-30)

## Corpus Check
- 100 files · ~180,069 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 4, .urdf 1, .toml 1)

## Summary
- 1329 nodes · 2550 edges · 106 communities (92 shown, 14 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 105 edges (avg confidence: 0.83)
- Token cost: 817,749 input · 0 output

## Community Hubs (Navigation)
- Dashboard Architecture & Arms
- Phase 2 Run Log Findings
- Reward Functions (rewards.py)
- Scene Tutorial Scripts
- PPO Runner Configs
- Template Env Config
- HexapodEnv Core
- S1 Run Summaries & Kinematics
- S0a/S0b Env Subclasses
- Observation Terms & Layouts
- S0PPO & Normaliser Folding
- S0 Gait Evaluation
- Isaac Lab Utilities
- Robot Asset Config
- Reward Annealing & Swing Terms
- Phase 2.0 Correctness Fixes
- Phase 2.1 S0b Development
- Phase 2.2 S1 Infrastructure
- Env Config Variants
- AX-12A Actuator Code
- CPG Reward Terms
- Sim Check Scripts
- Gait Probe Scripts
- Phase 1 Reward Tuning
- S1 Aux Learner
- Proprio TCN Encoder
- S1 Observation Config
- Actuator Sanity Tests
- Project Guidance & Phases
- RSL-RL Training Entry
- Teleop Demo & Terrain Cfg
- Aux Losses (beta-NLL)
- Obs Equivalence Diagnosis
- Sequence Buffer
- Gait Recording & Layout Probes
- Rendering & Falsification Spec
- Random/Zero Agents
- RSL-RL CLI Args
- Legacy Trajectory Analysis
- Phase 1 Exploits & S0b Gates
- Final pi_S0 & Lessons
- Terrain Material Tests
- S0 Reconditioning Spec
- FAH & Proprio Encoder (v7)
- S1 Blind Trunk Summary
- Blind Trunk Spec (S1)
- Critic Contract & Citations
- CPG Reward Defects & Camera
- Train Script
- Toe & Stance Geometry Rewards
- Terrain Height & Curriculum
- Vision Arms B2/B3/F
- Omniverse UI Extension
- S0a Reconditioning Summary
- Headroom Gate & Compute
- KiVi Arms & Proprio Frame
- VP-JEPA Claim & Related Work
- Terrain Materials Spec
- S0 Lineage (CLAUDE.md)
- Play Env & Tmux Rules
- List Envs Script
- S1 PPO & Merge Tests
- JEPA Losses & Critic (v7)
- 30 Hz Anticipation Path
- Clock Anneal & Priv Probes
- Pre-commit Hooks
- S1 Budget & Seeds
- Policy Checkpoint Lineage
- Play Script A
- Play Script B
- Foot Friction Writer
- Flat Env Config
- Scene Configs
- Hardware & Actuator Model
- Paradigm Invariants (CLAUDE.md)
- Teleop & Env State Loading
- Research Progress Summary
- Coxa Contact & pi_S0a
- Isaac Sim Screenshot
- Probe Loss Spec
- Warm Start & KL Anchor
- Diagnostics & Contact Switches
- Arms & Stage Schedule
- S1 Task Components
- Project Stack
- Crutch Legs & Mirror
- Leg Proximity & Clip
- Aux Config
- Rewards Config
- Velocity Command
- Evaluation Protocol
- RSL-RL Arg Parser
- S1 Sweep Script
- Texture (Source Asset)
- Texture (Model)
- Record-All Script
- S0a Actions
- non_foot_body Term
- isaaclab-dev
- Forward Distance Metric
- Leg Order Assertion
- World-Z Contact Force Gap

## God Nodes (most connected - your core abstractions)
1. `Phase 2 Run Logs (archive)` - 64 edges
2. `Hexapod Project Dashboard (source)` - 36 edges
3. `Phase 2.0: Correctness fixes + S0a reconditioning` - 23 edges
4. `Phase 2.1: S0b commands, clock, standing, terrain, DR` - 23 edges
5. `Phase 2.2: S1 blind trunk` - 22 edges
6. `toe_pos_w()` - 20 edges
7. `CLAUDE.md agent guidance` - 18 edges
8. `Hexapod RL Development & Reward Tuning Log (Phase 1)` - 17 edges
9. `S1Aux` - 16 edges
10. `VP-JEPA Shared Training Paradigm & Baseline Specification v2` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Floor-is-lava exploit: MSE on contact forces` --semantically_similar_to--> `Reward economy check (walk > stand > fall, reward_economy.py)`  [INFERRED] [semantically similar]
  research/hexapod-RL-tuning-guide.md → research_progress.md
- `Principle: make the local optimum economically unviable instead of reseeding` --semantically_similar_to--> `Reward economy check (walk > stand > fall, reward_economy.py)`  [INFERRED] [semantically similar]
  research/hexapod-RL-tuning-guide.md → research_progress.md
- `Frozen CPG clock bug (_step_impl override never called)` --semantically_similar_to--> `coxa_rest_reach measured in USD zero pose bug`  [INFERRED] [semantically similar]
  research/hexapod-RL-tuning-guide.md → research_progress.md
- `CLAUDE.md agent guidance` --references--> `ruff linter and formatter hooks`  [INFERRED]
  CLAUDE.md → .pre-commit-config.yaml
- `main()` --calls--> `widen_actor_state()`  [INFERRED]
  scripts/rsl_rl/train.py → source/hexasim/hexasim/rl/s0_ppo.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **30 Hz anticipation path (E_v, FAH, sampling, P)** — vp_jepa_architecture_e_v_encoder, vp_jepa_architecture_fah, vp_jepa_architecture_foothold_sampling, vp_jepa_architecture_predictor_p, vp_jepa_architecture_anticipation_loop_30hz [EXTRACTED 1.00]
- **Total loss L = JEPA + inv + priv + fah + vel + vic + PPO** — vp_jepa_architecture_loss_l_jepa, vp_jepa_architecture_loss_l_inv, vp_jepa_architecture_loss_l_priv, vp_jepa_architecture_loss_l_fah, vp_jepa_architecture_loss_l_vel, vp_jepa_architecture_loss_l_vic [EXTRACTED 1.00]
- **Systematic, motion-correlated sim2real effects that must be modelled not randomized** — vp_jepa_architecture_rolling_shutter_model, vp_jepa_architecture_dropped_reads_zoh_mask, vp_jepa_architecture_specular_depth_dropout [INFERRED 0.85]
- **B2 vs F falsification contrast (differ by horizon queries, L_jepa, L_inv, EMA target)** — vp_jepa___shared_training_paradigm___baseline_specification_arm_b2, vp_jepa___shared_training_paradigm___baseline_specification_arm_f, vp_jepa___shared_training_paradigm___baseline_specification_fusion_phi, vp_jepa___shared_training_paradigm___baseline_specification_predictor_p, vp_jepa___shared_training_paradigm___baseline_specification_paired_t_test [EXTRACTED 1.00]
- **Three privileged-information channels shared by all arms** — vp_jepa___shared_training_paradigm___baseline_specification_critic_contract_p, vp_jepa___shared_training_paradigm___baseline_specification_velocity_estimator_v, vp_jepa___shared_training_paradigm___baseline_specification_probe_set [EXTRACTED 1.00]
- **Stage pipeline S0a → S0b → S1 → S2 → S3** — vp_jepa___shared_training_paradigm___baseline_specification_stage_s0a, vp_jepa___shared_training_paradigm___baseline_specification_stage_s0b, vp_jepa___shared_training_paradigm___baseline_specification_stage_s1, vp_jepa___shared_training_paradigm___baseline_specification_stage_s2, vp_jepa___shared_training_paradigm___baseline_specification_stage_s3 [EXTRACTED 1.00]
- **Gait-quality reward corrections behind final pi_S0** — research_phase2_progress_swing_radial_motion, research_phase2_progress_touchdown_reach, research_phase2_progress_leg_proximity, research_phase2_progress_rest_reach_bug, research_phase2_progress_pi_s0_final [EXTRACTED 1.00]
- **S1 env-owned aux learner (E_p, V, probes, buffer)** — research_phase2_progress_tcn_ep, research_phase2_progress_velocity_estimator_v, research_phase2_progress_seq_buffer, research_phase2_progress_probe_stats, research_phase2_progress_s1ppo [EXTRACTED 1.00]
- **Standing-mode failure and fixes** — research_phase2_progress_standing_mode, research_phase2_progress_zero_clock_standing, research_phase2_progress_settle_window, research_phase2_progress_action_beyond_clip [INFERRED 0.85]
- **S0b standing tucked-leg fix attempts** — research_phase2_run_logs_tucked_leg_optimum, research_phase2_run_logs_standing_pose_term, research_phase2_run_logs_action_beyond_clip, research_phase2_run_logs_settle_window, research_phase2_run_logs_standing_mode [EXTRACTED 1.00]
- **Gait-quality terms of pi_S0 (coordinated lift, clearance)** — research_phase2_run_logs_swing_radial_motion, research_phase2_run_logs_touchdown_reach, research_phase2_run_logs_leg_proximity, research_phase2_run_logs_pi_s0 [EXTRACTED 1.00]
- **S1 pilot lineage to budget and seed trunks** — research_phase2_run_logs_s1_pilot, research_phase2_run_logs_s1_pilot2, research_phase2_run_logs_s1_pilot3, research_phase2_run_logs_plateau_criterion, research_phase2_run_logs_w_sweep, research_phase2_run_logs_s1_seed_trunks [EXTRACTED 1.00]
- **S1 blind trunk components (E_p, V, probes, contract P, layout, S1PPO)** — research_progress_e_p_tcn, research_progress_velocity_estimator_v, research_progress_universal_probes, research_progress_contract_p, research_progress_actor_layout_1125, research_progress_s1ppo [EXTRACTED 1.00]
- **Gait-quality corrections that forced pi_S0 retrain** — research_progress_swing_radial_motion, research_progress_touchdown_reach, research_progress_leg_proximity_penalty, research_progress_rest_reach_bug, research_progress_pi_s0_final [EXTRACTED 1.00]
- **Phase 1 CPG reward stack** — research_hexapod_rl_tuning_guide_cpg_clock, research_hexapod_rl_tuning_guide_cpg_z_target_reward, research_hexapod_rl_tuning_guide_cpg_force_target_penalty, research_hexapod_rl_tuning_guide_cpg_coxa_contact_reward, research_hexapod_rl_tuning_guide_cpg_stance_rigidity_penalty, research_hexapod_rl_tuning_guide_cpg_curriculum [INFERRED 0.85]
- **Shared paradigm invariants identical across arms** — claude_paradigm_invariants, claude_deployable_actor_inputs, claude_dropped_reads_model, claude_toe_point_definition, claude_friction_multiply, claude_cpg_clock [EXTRACTED 1.00]
- **Env task lineage Flat -> S0a -> S0b -> S1** — claude_hexapod_flat_v0, claude_hexapod_s0a_v0, claude_hexapod_s0b_v0, claude_hexapod_s1_v0 [INFERRED 0.85]
- **S1 Play demo/recording flow** — research_playenv_teleop_updates_demo_teleop, research_playenv_teleop_updates_record_gait, research_playenv_teleop_updates_hexapod_s1_play_v0, research_playenv_teleop_updates_hexapods1envcfg_play, claude_load_env_state [EXTRACTED 1.00]
- **Planned VP-JEPA anticipation pipeline** — docs_dashboard_dashboard_source_realsense_d435, docs_dashboard_dashboard_source_vision_encoder, docs_dashboard_dashboard_source_foothold_anticipation_head, docs_dashboard_dashboard_source_feature_lookup, docs_dashboard_dashboard_source_contact_memory, docs_dashboard_dashboard_source_future_predictor, docs_dashboard_dashboard_source_ema_target_encoder, docs_dashboard_dashboard_source_action_recovery_head [EXTRACTED 1.00]
- **S1 blind trunk components** — docs_dashboard_dashboard_source_per_step_body_reading, docs_dashboard_dashboard_source_body_sensing_encoder, docs_dashboard_dashboard_source_velocity_force_estimator, docs_dashboard_dashboard_source_physics_quiz_probes, docs_dashboard_dashboard_source_privileged_contract_p, docs_dashboard_dashboard_source_critic_network, docs_dashboard_dashboard_source_actor_union_layout [EXTRACTED 1.00]
- **Comparison arms under one shared paradigm** — docs_dashboard_dashboard_source_arm_o, docs_dashboard_dashboard_source_arm_b0, docs_dashboard_dashboard_source_arm_b1_k, docs_dashboard_dashboard_source_arm_b1_m, docs_dashboard_dashboard_source_arm_b2, docs_dashboard_dashboard_source_arm_f, docs_dashboard_dashboard_source_arm_b3 [EXTRACTED 1.00]

## Communities (106 total, 14 thin omitted)

### Community 0 - "Dashboard Architecture & Arms"
Cohesion: 0.05
Nodes (72): Hexapod Project Dashboard (source), Action-recovery (inverse) head L_inv, Actor union observation layout (unused blocks held at zero), Anticipation loop 30 Hz (planned), Arm B0 (Blind), Arm B1-K (KiVi faithful), Arm B1-M (KiVi matched), Arm B2 (foothold-attended, no predictor) (+64 more)

### Community 1 - "Phase 2 Run Log Findings"
Cohesion: 0.06
Nodes (60): Phase 2 Run Logs (archive), action_beyond_clip penalty, AX-12A explicit servo model, D30 back-coxa requests past +-55 deg clip, eval_s0 clearance metric correction (swings > 5 mm), coxa_stroke_weight scaled by rest reach (lopsided turning fix), cpg_z_target toe swing kernel, Right-side crutch leg imbalance (+52 more)

### Community 2 - "Reward Functions (rewards.py)"
Cohesion: 0.07
Nodes (45): SceneEntityCfg, action_beyond_clip_penalty(), adjacent_leg_gaps(), ang_vel_xy_penalty(), base_height_penalty(), base_orientation_penalty(), cpg_clock(), forward_velocity() (+37 more)

### Community 3 - "Scene Tutorial Scripts"
Cohesion: 0.06
Nodes (31): dataclasses, isaaclab_physx_sim_schemas, isaaclab_physx_sim_spawners_materials, isaaclab_sim, ManagerBasedEnv, design_scene(), main(), Designs the scene by spawning ground plane, light, objects and meshes from usd… (+23 more)

### Community 4 - "PPO Runner Configs"
Cohesion: 0.11
Nodes (34): RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, PPORunnerCfg, PPORunnerCfg_S0a, PPORunnerCfg_S0b, PPORunnerCfg_S1, configclass, Extra knobs of :class:`hexasim.rl.s1_ppo.S1PPO` (phase2-progress §2.2). (+26 more)

### Community 5 - "Template Env Config"
Cohesion: 0.07
Nodes (29): isaaclab_assets_robots_cartpole, isaaclab_envs_mdp_commands_commands_cfg, isaaclab_scene, isaaclab_utils_configclass, ActionsCfg, EventCfg, HexasimEnvCfg, HexasimSceneCfg (+21 more)

### Community 6 - "HexapodEnv Core"
Cohesion: 0.11
Nodes (12): HexapodEnv, HexapodS1Env, ManagerBasedRLEnv, Tensor, Hexapod env with a CPG clock, an adaptive toe-contact baseline and the velocity…, S1 blind trunk env (paradigm spec §6, phase2-progress §2.2). On top of S0b: *…, Body-frame horizontal offsets (..., K, 2) rotated by base yaw → world xy…, (μ_s, μ_d, log10 k, log10(1 + c)) at world xy (N, ..., 2); damping c is per env… (+4 more)

### Community 7 - "S1 Run Summaries & Kinematics"
Cohesion: 0.09
Nodes (26): json, pathlib, re, Summarise final S1 evaluations of several runs side by side (phase2-progress…, row(), tb(), _chain(), _joint() (+18 more)

### Community 8 - "S0a/S0b Env Subclasses"
Cohesion: 0.10
Nodes (14): isaaclab_managers, rsl_rl_modules_normalization, HexapodS0aEnv, HexapodS0bEnv, S0a reconditioning env (paradigm spec §4.2): deployable 111-D o_p frame + the…, S0b env (paradigm spec §4.2): twist_cmd, command-referenced clock, standing…, FootWriter, Writes per-foot (μ_s, μ_d, −k) into the tibia collision shapes via the PhysX… (+6 more)

### Community 9 - "Observation Terms & Layouts"
Cohesion: 0.13
Nodes (23): isaaclab_utils_module, Custom MDP functions for the hexapod environment., contract_p(), Tensor, Actor input, 114-D: o_p (111) ⊕ v_in (3). o_p = [q − q_default (18), q̇ (18),…, Critic-only extras, 22-D (S0a subset of contract P, spec §3.5). True base…, Actor input, 117-D: o_p (111) ⊕ v_in (3, V_0) ⊕ twist_cmd (3)., Critic-only dynamics-randomisation parameters (part of contract P's physics… (+15 more)

### Community 10 - "S0PPO & Normaliser Folding"
Cohesion: 0.10
Nodes (10): Linear, PPO, fold_normalizer(), Rewrite ``first_layer`` so that layer(norm_new(x)) == layer(norm_old(x)) for…, S0PPO, slice, Tensor, Overwrite a normaliser's statistics on ``cols``. The buffers are refreshed… (+2 more)

### Community 11 - "S0 Gait Evaluation"
Cohesion: 0.12
Nodes (15): copy, itertools, rsl_rl_algorithms, rsl_rl_models, rsl_rl_storage, Headless gait evaluation for the S0a gates (paradigm spec §4.2): one 10 s…, Toe point (spec §1). Duplicated from mdp.rewards so the script also runs on the…, toe_pos_w() (+7 more)

### Community 12 - "Isaac Lab Utilities"
Cohesion: 0.22
Nodes (17): isaaclab_envs, isaaclab_rl_rsl_rl, isaaclab_rl_utils_pretrained_checkpoint, isaaclab_tasks_utils_hydra, isaaclab_utils_assets, isaaclab_utils_dict, isaaclab_utils_io, isaaclab_utils_seed (+9 more)

### Community 13 - "Robot Asset Config"
Cohesion: 0.10
Nodes (19): isaaclab_assets, isaaclab_envs_mdp, isaaclab_physx_physics, isaaclab_sensors, source_hexasim_hexasim_actuators_init, # NOTE: -60° (-1.047 rad) was below the lower limit of -45° (-0.785 rad), Configuration for the PhantomX Mark IV hexapod robot by Interbotix. USDA…, ActionsCfg (+11 more)

### Community 14 - "Reward Annealing & Swing Terms"
Cohesion: 0.14
Nodes (10): ManagerTermBase, LinearAnnealRewardWeight, ManagerBasedRLEnv, Phase-free swing floor: per foot, peak toe height above the terrain over an…, Σ over airborne legs of |Δ|r_xy|| / dt (m/s): in/out toe motion relative to the…, Σ over legs touching down this step of | |r_xy| − rest |r_xy| | (m): the…, Curriculum term: linearly interpolate a reward weight over an env-step window., swing_clearance (+2 more)

### Community 15 - "Phase 2.0 Correctness Fixes"
Cohesion: 0.15
Nodes (20): Phase 2 Baseline Paradigm Development Log, AX-12A armature 0.003 kg m^2, AX-12A servo model, Camera mast mass (0.15 kg provisional), Clearance metric: swings >5 mm apex only, coxa_sweep_sign twist-projected coxa sign, cpg_coxa_contact signed stroke reward, cpg_z_target bounded toe-swing kernel (sigma 1 cm, x5) (+12 more)

### Community 16 - "Phase 2.1 S0b Development"
Cohesion: 0.15
Nodes (19): Coordinated swing lift (femur + tibia together), coxa_stroke_weight reach weighting, CPG clock (0.8 Hz floor, phase jitter, command-referenced f_ref), gait_by_material.py per-term ledger probe, Feet migrated under hips (tucking), Per-step tracking x3 (exp + L1), Phase 2.1: S0b commands, clock, standing, terrain, DR, No module-level pxr imports in mdp/ (+11 more)

### Community 17 - "Phase 2.2 S1 Infrastructure"
Cohesion: 0.16
Nodes (19): Foot damping per env (D25), Foot material bound on tibia rigid body, fp16 rollout inference + mixed-precision aux, Blind O vs B0 headroom test, non_foot_body penalty never annealed (D33), N_v benchmark (blocked), Phase 2.2: S1 blind trunk, probe_stats_pi_s0.pt fixed probe statistics (+11 more)

### Community 18 - "Env Config Variants"
Cohesion: 0.14
Nodes (12): HexapodFlatResumeEnvCfg, HexapodS0aEnvCfg, HexapodS0aEnvCfg_PLAY, HexapodS0bEnvCfg, HexapodS0bEnvCfg_PLAY, HexapodS1EnvCfg, HexapodS1EnvCfg_PLAY, S1: S0b + spatial terrain materials, per-env drop slope, the aux learner and… (+4 more)

### Community 19 - "AX-12A Actuator Code"
Cohesion: 0.14
Nodes (11): ActuatorBase, ActuatorBaseCfg, AX12AActuator, AX12AActuatorCfg, configclass, slice, Tensor, quantise() (+3 more)

### Community 20 - "CPG Reward Terms"
Cohesion: 0.12
Nodes (17): Articulation, cpg_coxa_contact_reward(), cpg_force_target_penalty(), cpg_stance_rigidity_penalty(), cpg_z_target_kernel_reward(), cpg_z_target_l1_penalty(), joint_leg_index(), Leg index (0..5, LEG_NAMES order) of every selected joint, from joint names. (+9 more)

### Community 21 - "Sim Check Scripts"
Cohesion: 0.20
Nodes (8): collections, gymnasium, isaaclab_app, isaaclab_tasks_utils_parse_cfg, Friction combine-mode check (paradigm spec §1): effective foot μ must equal the…, Check the left–right mirror of ``mdp.symmetry`` against the simulator…, Per-component timing of one Hexapod-S1-v0 control step (collection cost;…, timed()

### Community 22 - "Gait Probe Scripts"
Cohesion: 0.15
Nodes (12): isaaclab_terrains, Gait metrics split by ground friction (phase2-progress §2.2.6: swing-lift drift…, run(), Leg-to-leg clearance per command (phase2-progress §2.1.38): capsule surface gap…, Decompose each swing's toe motion relative to its hip (phase2-progress…, configure(), eval_physx_buffers(), load_env_state() (+4 more)

### Community 23 - "Phase 1 Reward Tuning"
Cohesion: 0.18
Nodes (16): Hexapod RL Development & Reward Tuning Log (Phase 1), Adaptive EMA contact foot baseline (cpg_foot_baseline), Permanently airborne (hanging toe) leg exploit, base_height_penalty (one-sided, -60), base_orientation_penalty (projected gravity, two-zone), CPG reward curriculum (scaffolding fade-out, CPG ramp-in), cpg_force_target_penalty (L1), cpg_z_target_reward (swing-only, normalized) (+8 more)

### Community 24 - "S1 Aux Learner"
Cohesion: 0.21
Nodes (8): no_grad, Tensor, hist (B, W, 111) → z_p (B, 64), ê (B, 15) = [v̂_b, f̂], v_kin (B, 3)., Rollout inference (the actor's z_p, ê). fp16 autocast: 1.6× faster at 8,192 ×…, Set the statistics from the buffer's current contents (used by…, One aux step block per PPO iteration; returns logs., Slip-stratified v̂_b error and the IMU-zero ablation on buffer samples (current…, S1Aux

### Community 25 - "Proprio TCN Encoder"
Cohesion: 0.20
Nodes (9): CausalConv, dilations_for(), ProprioTCN, Tensor, Power-of-two dilation stack, at least {1, 2, 4, 8} (v7 §4.2), extended until…, (B, W, in_dim) → z_p (B, out_dim). Frames ordered oldest → newest; the newest…, E_p — proprioceptive encoder: causal dilated TCN over the last W per-step…, receptive_field() (+1 more)

### Community 26 - "S1 Observation Config"
Cohesion: 0.22
Nodes (15): ContractCfg, CriticCfg, MirrorCfg, ObservationsCfg, PolicyCfg, configclass, ObsGroup, Actor: the 1,125-D union layout (``mdp.layout``). Critic: sg[actor obs] ⊕… (+7 more)

### Community 27 - "Actuator Sanity Tests"
Cohesion: 0.16
Nodes (8): argparse, math, Implicit PhysX drive vs explicit IdealPD at identical gains, robot floating in…, AX-12A servo-model sanity check on Hexapod-S0a-v0: hold a fixed pose for 3 s…, Per-env foot material check for Hexapod-S0b-v0 (paradigm spec §1 friction…, lever_torques(), Static torque envelope at the rest pose (paradigm spec §4.2 prerequisite, v7…, Per-leg femur/tibia holding torque from contact force and geometry: ((toe -…

### Community 28 - "Project Guidance & Phases"
Cohesion: 0.20
Nodes (15): CLAUDE.md agent guidance, Hexapod-Flat-v0 task, research/hexapod-RL-tuning-guide.md, Use L1 not L2 on raw physics signals, Phase 1: flat-ground CPG PPO base policy, Phase 2: comparison baselines (terrain curriculum + DR), research/phase2-progress.md, Phase 3: Arm F VP-JEPA (+7 more)

### Community 29 - "RSL-RL Training Entry"
Cohesion: 0.15
Nodes (14): common, datetime, importlib_metadata, logging, packaging, platform, _check_rsl_rl_version(), _parse_args() (+6 more)

### Community 30 - "Teleop Demo & Terrain Cfg"
Cohesion: 0.14
Nodes (12): isaaclab_devices_keyboard, isaaclab_terrains_height_field, isaaclab_utils_warp, Keyboard-driven demonstration of an S0b policy on every terrain type at one…, HfScaledRoughTerrainCfg, configclass, Isaac Lab's random-uniform height field ignores difficulty; scale its amplitude…, S0b terrain (paradigm spec §4.2): generator config, command-referenced… (+4 more)

### Community 31 - "Aux Losses (beta-NLL)"
Cohesion: 0.20
Nodes (11): S1 auxiliary learner: E_p, V, universal probe (O/B0 heads), VICReg, trained…, beta_nll(), GaussianHead, Tensor, Element-wise β-NLL; same shape as ``y``., 2-layer MLP (256 hidden) → per-target mean and log-variance (§3.2: same head…, Variance hinge on per-dim std (target γ) and off-diagonal covariance, both…, Auxiliary losses and heads for S1 (paradigm spec §3.2–3.4, v7 §4.11). * β-NLL… (+3 more)

### Community 32 - "Obs Equivalence Diagnosis"
Cohesion: 0.16
Nodes (10): collections_abc, isaaclab_actuators, isaaclab_envs_mdp_commands_velocity_command, isaaclab_utils_buffers, gt_obs(), Flat-v0 policy obs from simulator ground truth, in π_base's frame (old rest…, Diagnosis: does the warm-started S0a actor, on its deployable o_p, command what…, Dynamixel AX-12A servo model (v7 §5.1): P-only compliance curve on a DC motor,… (+2 more)

### Community 33 - "Sequence Buffer"
Cohesion: 0.20
Nodes (8): device, Tensor, Random valid rows. Returns hist (B, W, F), cur (B, C), td_pos_b (B, legs, 3),…, Row for the current control step., Legs (N, legs) that touched down at the step just added; ``pos_w`` (N, legs,…, Envs whose episode ended at the step just added; the next row starts a new…, (N, L) rows usable as sample position t: written, with a full W-step history in…, SequenceBuffer

### Community 34 - "Gait Recording & Layout Probes"
Cohesion: 0.16
Nodes (9): isaaclab_utils_math, isaacsim_core_rendering_manager, pxr, Record one policy's gait on flat ground and print its leg geometry…, Print the robot's runtime layout: joint/body/sensor order, limits, masses, toe…, Hexapod-S1-v0 wiring check (phase2-progress §2.2), random-ish actions around…, # NOTE: The UI extension (``ui_extension_example.py``) imports ``omni.ext``,…, Python module serving as a project/extension template. (+1 more)

### Community 35 - "Rendering & Falsification Spec"
Cohesion: 0.15
Nodes (14): Clock Dropout per Episode, Colour FOV Binds Image Geometry (69°x42°), Shared Depth Corruption Post-process, Pre-registered Visual-Detectability Gate, Falsification Logic (§11), Shared Image Operator (128x96, no crop), Intel RealSense D435 Specifications, NVIDIA Omniverse RTX Renderer (+6 more)

### Community 36 - "Random/Zero Agents"
Cohesion: 0.21
Nodes (11): contextlib, isaaclab_tasks, isaaclab_tasks_experimental, isaaclab_tasks_utils, main(), Random actions agent with Isaac Lab environment., Script to an environment with random action agent., main() (+3 more)

### Community 37 - "RSL-RL CLI Args"
Cohesion: 0.18
Nodes (13): parse_rsl_rl_cfg(), Namespace, RslRlBaseRunnerCfg, Parse configuration for RSL-RL agent based on inputs. Args: task_name: The name…, Update configuration for RSL-RL agent based on inputs. Args: agent_cfg: The…, update_rsl_rl_cfg(), main(), DirectMARLEnvCfg (+5 more)

### Community 38 - "Legacy Trajectory Analysis"
Cohesion: 0.18
Nodes (8): glob, numpy, os, pandas, setuptools, # NOTE: Add dependencies, Installation script for the 'hexasim' python package., toml

### Community 39 - "Phase 1 Exploits & S0b Gates"
Cohesion: 0.20
Nodes (12): Adaptive CPG frequency f = v_x/0.2 clamped, CPG clock (tripod offsets 0 and pi), Frozen CPG clock bug (_step_impl override never called), Sliding (belly) exploit, Stand-still exploit and stand_still_penalty, Domain randomization in S0b (D17), Left-right mirror augmentation (mdp/symmetry.py, D21), coxa_rest_reach measured in USD zero pose bug (+4 more)

### Community 40 - "Final pi_S0 & Lessons"
Cohesion: 0.23
Nodes (12): ContactSensor setup on tibia links (PhysxContactReportAPI), D30 back-coxa clip excess (revisit in S2), Lesson: gates are necessary, not sufficient, leg_proximity_penalty (capsule gap, -400 then -1600), Next steps per spec 12: N_v benchmark, S1b, S2 reward, O vs B0 headroom, Final pi_S0 (from scratch, model_12200), Plateau stop rule (<1.5% return gain per 2,000 iterations), Reward economy check (walk > stand > fall, reward_economy.py) (+4 more)

### Community 41 - "Terrain Material Tests"
Cohesion: 0.17
Nodes (4): Spatial terrain materials check for Hexapod-S1-v0 (phase2-progress §2.2;…, Feasibility spike (S1 plan §1, option A): per-step, per-foot friction through…, sink(), warp

### Community 42 - "S0 Reconditioning Spec"
Cohesion: 0.21
Nodes (12): Clock Reference f_ref (0.8 Hz floor), Friction Combine Mode Multiply, Invariants Identical for Every Arm (§1), Left-Right Mirror Data Augmentation, π_base (model_14998.pt), Reward Schedule across Stages (§5), Stage-0 Audit of π_base, Stage S0a (actuator & observations) (+4 more)

### Community 43 - "FAH & Proprio Encoder (v7)"
Cohesion: 0.21
Nodes (12): Deferred ablation A: proprioceptive receptive field, 50 Hz control loop, DreamWaQ (Nahrendra et al. 2023), Proprioceptive encoder E_p (dilated TCN), Deployment execution split (CPU control, GPU anticipation), Foothold Anticipation Head (FAH), Jetson Orin Nano 8 GB, L_fah Gaussian NLL (+4 more)

### Community 44 - "S1 Blind Trunk Summary"
Cohesion: 0.25
Nodes (11): 1,125-D union actor layout (mdp/layout.py), Contract P (683-D critic privileged input), E_p causal dilated TCN (rl/s1/tcn.py, RF 61 >= W), S1 blind trunk (Hexapod-S1-v0), S1PPO (rl/s1_ppo.py), Spatial terrain material field (Voronoi, 10-entry library), Universal probes with beta-NLL (beta 0.5), V_0 (5-frame o_p MLP velocity estimate, D2) (+3 more)

### Community 45 - "Blind Trunk Spec (S1)"
Cohesion: 0.20
Nodes (11): Contact Switches as Instrumentation Only, E_p Proprioceptive TCN Encoder (W=50), Paired Seed Branching (§10.1), Bai, Kolter, Koltun 2018 TCN, Residual Policy Fallback, Sequence Buffer for Auxiliary Losses (L=128), Stage S1 / S1b Blind Trunk, TCN Window W=50 & Receptive Field Assertion (+3 more)

### Community 46 - "Critic Contract & Citations"
Cohesion: 0.22
Nodes (11): Critic Contract P (683-D), Gradient Routing & CI Unit Tests, v2 Precedence Table (§0.5), Baisero & Amato 2022 Unbiased Asymmetric RL, Mesnard et al. 2021 Counterfactual Credit Assignment, PIE (Luo et al., RA-L 2024), Robot Parkour Learning (Zhuang et al., CoRL 2023), WMP: World Model-based Perception (Lai et al., ICRA 2025) (+3 more)

### Community 47 - "CPG Reward Defects & Camera"
Cohesion: 0.18
Nodes (11): Actuator saturation vs claim / static torque envelope, Camera mount h ~0.22 m, pitch 45-50 deg, cpg_coxa velocity/contact reward terms, cpg_force_target_penalty, CPG reward floors in Stage 2, cpg_stance_rigidity_penalty, cpg_z_target reward/penalty, duty_factor single source (+3 more)

### Community 48 - "Train Script"
Cohesion: 0.20
Nodes (10): main(), DirectMARLEnvCfg, DirectRLEnvCfg, hydra_task_config, ManagerBasedRLEnvCfg, RslRlBaseRunnerCfg, Train with RSL-RL agent., Tensor (+2 more)

### Community 49 - "Toe & Stance Geometry Rewards"
Cohesion: 0.22
Nodes (10): coxa_leg_geometry(), coxa_stroke_weight(), foot_clearance_reward(), Anti-slide: sum over feet of toe height clamped to [0, target_clearance],…, World-frame toe positions (N, 6, 3) for the given tibia body ids., Body-frame toe positions from the base, p_toe, and from each coxa joint, r…, Per leg (N, 6): sign × reach — the coxa rate multiplier that pays the toe…, Σ over legs in contact of | |r_xy| − rest |r_xy| | (m): horizontal toe distance… (+2 more)

### Community 50 - "Terrain Height & Curriculum"
Cohesion: 0.22
Nodes (8): pin_terrain_level(), ManagerBasedRLEnv, Tensor, Bilinear terrain height at world xy (..., 2) → (...)., Unit surface normal at world xy (..., 2) → (..., 3), central differences over…, Terrain curriculum on the commanded path length (Isaac Lab's…, Evaluation: put every env on terrain level ``level`` (its own column type);…, terrain_levels_cmd()

### Community 51 - "Vision Arms B2/B3/F"
Cohesion: 0.36
Nodes (10): Actor Observation Layouts (§2.9, union layout), Arm B2 (Foothold-attended, no predictor), Arm B3 (Generative-target control, optional), Arm F (VP-JEPA), Budgets and Pilot Gates (§6.1), E_v Vision Stem, Foothold Attention Head (FAH) + sampling + contact memory, Fusion Φ (B2, 3 learned queries) (+2 more)

### Community 52 - "Omniverse UI Extension"
Cohesion: 0.22
Nodes (3): omni_ext, omni_ui, ExampleExtension

### Community 53 - "S0a Reconditioning Summary"
Cohesion: 0.25
Nodes (9): Duty factor contradiction (5/6 vs tripod offsets) -> 0.5, AX-12A servo model (hexasim/actuators/ax12a.py), Dropped reads, ZOH and 18-bit validity mask (mdp/proprio.py), From-scratch control arm (spec 4.2), Lesson: inherited habits survive warm starts, Phase 2 Summary: S0 reconditioning to S1 blind trunk, PhysX enable_external_forces_every_iteration fix (D11), S0a reconditioning (Hexapod-S0a-v0) (+1 more)

### Community 54 - "Headroom Gate & Compute"
Cohesion: 0.22
Nodes (9): RTX A4000 Envelope / N_v Benchmark, Arm B0 (Blind), Arm O (Oracle), Compute Feasibility & Cut Order, Critical Path (§12), Envs x Batch Invariant (N_v x T_v >= 24,576), O vs B0 Headroom Gate, Material Strips M* (nominal foothold, 576-D) (+1 more)

### Community 55 - "KiVi Arms & Proprio Frame"
Cohesion: 0.31
Nodes (9): Arm B1-K (KiVi re-implemented), Arm B1-M (KiVi-matched), Concurrent Estimation + Asymmetric Critic Template, Dropped Reads Model (ZOH + 18-bit mask), Two-Sample Joint Velocity Difference, Per-step Frame o_p (111-D), Single Shared Paradigm (no separate replication), Ji et al. 2022 Concurrent Training of Policy and State Estimator (+1 more)

### Community 56 - "VP-JEPA Claim & Related Work"
Cohesion: 0.31
Nodes (9): Two-stage anticipation cascade (vision for leading legs, proprio latent for trailing legs), HIM, KiVi (2025) - nearest competitor, Miki et al. 2022, Modality dropout, Neural Volumetric Memory (Yang et al. 2023), V-JEPA 2-AC (action-conditioned JEPA), Anticipatory Visuo-Proprioceptive JEPA (VP-JEPA) (+1 more)

### Community 57 - "Terrain Materials Spec"
Cohesion: 0.25
Nodes (9): Appearance factorisation / material library, PhysX friction combine multiply, Gradual vs sharp transition conditions, Held-out harder-terrain suite, Matte low-friction primary test surface (PTFE/talc), Self-occlusion Tier 1 ray-vs-capsule test, Specularity-driven depth-validity dropout, Terrain conditions: friction primary, compliance secondary (+1 more)

### Community 58 - "S0 Lineage (CLAUDE.md)"
Cohesion: 0.25
Nodes (8): AX-12A servo actuator model (hexasim/actuators/ax12a.py), Hexapod-S0a-v0 task, Hexapod-S0b-v0 task, KL anchor to pi_S0 (void), Left-right mirror data augmentation (mdp/symmetry.py), pi_S0 final checkpoint (from scratch, model_12200), S0 reconditioning (S0a/S0b), S0B_TERRAIN_CFG color_scheme random (white terrain fix)

### Community 59 - "Play Env & Tmux Rules"
Cohesion: 0.39
Nodes (8): Run sims through tmux session 0, Play Environment and Teleoperation Updates, Hexapod-S1-Play-v0 registration, HexapodS1EnvCfg_PLAY, scripts/rsl_rl/record_gait.py updates (yaw_cmd, Play task), s1_milestones.sh opt-in plateau stop (STOP_PCT), touchdown_reach front_reach parameter, Uncapped Play episode lengths (10000 s)

### Community 60 - "List Envs Script"
Cohesion: 0.29
Nodes (7): hexasim_tasks_experimental, prettytable, _format_presets(), main(), Format a preset map returned by :func:`enumerate_task_presets` into a human-…, Script to print all the available environments in Isaac Lab. The script…, Print all environments registered in `hexasim` extension.

### Community 61 - "S1 PPO & Merge Tests"
Cohesion: 0.25
Nodes (4): runpy, Merge-gating checks for S1 (paradigm spec §4.3, §8.2, §9.1; phase2-progress…, S1 PPO (paradigm spec §4.3, §6, §8; phase2-progress §2.2). On top of…, Per-env ring buffer for the auxiliary losses (paradigm spec §8.3, v7 §4.12).…

### Community 62 - "JEPA Losses & Critic (v7)"
Cohesion: 0.25
Nodes (8): Deferred ablation B: raw history stack, Separate privileged critic, EMA target encoder E_p', L_inv inverse dynamics, L_jepa (EMA target, smooth-L1), Policy pi (PPO asymmetric actor), RMA, Decoupled SSL FIFO transition buffer with episode masking

### Community 63 - "30 Hz Anticipation Path"
Cohesion: 0.36
Nodes (8): 30 Hz anticipation loop, Phase-indexed contact memory, Vision encoder E_v (CNN stem + transformer), Foothold sampling with validity m_l, Retrieval scoped to front legs (odometry stack deleted), Predictor P (cross-attention, horizons 60/200/400 ms), RGB-aligned depth (CUDA rs2::align), Self-occlusion Tier 2 rasterized mask channel

### Community 64 - "Clock Anneal & Priv Probes"
Cohesion: 0.25
Nodes (8): CPG clock (command-referenced, 0.8 Hz floor), lambda_priv schedule tied to modality introduction, L_priv privileged probes (future probe 2x), 111-D proprioceptive frame o_p, Paired clock/CPG-reward anneal (first ~30% of Stage 2), Per-env phase reset jitter, Stall lock (0.8 Hz clock floor), Training stages 1-3

### Community 65 - "Pre-commit Hooks"
Cohesion: 0.29
Nodes (7): pre-commit configuration, codespell hook, insert-license hook (.github/LICENSE_HEADER.txt), pre-commit-hooks standard checks, pygrep rst hooks, ruff linter and formatter hooks, Broken insert-license pre-commit hook

### Community 66 - "S1 Budget & Seeds"
Cohesion: 0.29
Nodes (7): Batch 4,096 x 16 (D1), Held-out terrain levels 15-20 (D22), S1 budget 8,000 iterations (plateau rule), S1 trunks seeds 42-46, Terrain curriculum terrain_levels_cmd, levels 0-14 + stairs (D20), Terrain geometry depends on run seed, eval_terrain_sweep.sh terrain stress sweep

### Community 67 - "Policy Checkpoint Lineage"
Cohesion: 0.29
Nodes (7): eval_s0b_gates.sh / eval_s0.py gate evaluation, From-scratch control arm, KL anchor to pi_S0 (void), OPENBLAS_NUM_THREADS=1 Kit start-up fix, pi_S0 final (coordlift + proximity, model_12200.pt), pi_S0 old (from scratch, model_20700.pt), S1 pilot 3 (seed 42, W 50, model_7999.pt)

### Community 68 - "Play Script A"
Cohesion: 0.29
Nodes (7): main(), DirectMARLEnvCfg, DirectRLEnvCfg, hydra_task_config, ManagerBasedRLEnvCfg, RslRlBaseRunnerCfg, Play with RSL-RL agent.

### Community 69 - "Play Script B"
Cohesion: 0.29
Nodes (7): main(), DirectMARLEnvCfg, DirectRLEnvCfg, hydra_task_config, ManagerBasedRLEnvCfg, RslRlBaseRunnerCfg, Play with RSL-RL agent.

### Community 70 - "Foot Friction Writer"
Cohesion: 0.29
Nodes (3): FootFriction, Writes (μ_s, μ_d) per env per foot to the tibia shapes; restitution kept as…, mu_s, mu_d: (N, 6) tensors (any device).

### Community 71 - "Flat Env Config"
Cohesion: 0.29
Nodes (5): HexapodEnvCfg, HexapodEnvCfg_PLAY, ManagerBasedRLEnvCfg, PhantomX Mark IV hexapod — flat ground locomotion environment., Single-env evaluation variant — no observation noise.

### Community 72 - "Scene Configs"
Cohesion: 0.29
Nodes (7): HexapodS0aSceneCfg, HexapodS0bSceneCfg, HexapodSceneCfg, InteractiveSceneCfg, AX-12A servos, rest pose femur +30° / tibia +30°, and a base IMU., Scene with flat ground, dome/distant lighting, hexapod robot, and foot contact…, Generated terrain (μ 1, restitution 1, multiply: the per-env contact physics…

### Community 73 - "Hardware & Actuator Model"
Cohesion: 0.29
Nodes (7): AX-12A actuator model (sim), Dynamixel AX-12A servo, Dynamixel bus configuration (RDT 0, FTDI 1 ms, position-only), Dropped reads: ZOH fill + 18-bit validity mask, Dynamics randomisation, PhantomX MK-IV hexapod (18-DOF AX-12A), Rolling-shutter skew model on RGB

### Community 74 - "Paradigm Invariants (CLAUDE.md)"
Cohesion: 0.33
Nodes (6): CPG clock with 0.8 Hz floor and per-env phase jitter, Deployable actor inputs rule, Dropped reads model with ZOH and 18-bit mask, Paradigm invariants (spec §1), Toe point toe_pos_w definition, Residual velocity estimator v_kin + delta v

### Community 75 - "Teleop & Env State Loading"
Cohesion: 0.33
Nodes (6): Foot friction mu=1.0 combine mode multiply, load_policy.load_env_state, Custom USD environment deployment guide, scripts/rsl_rl/demo_teleop.py updates, S1 Pilot 2 checkpoint (model_8000), hexasim.rl.load_policy.task_for

### Community 76 - "Research Progress Summary"
Cohesion: 0.33
Nodes (6): Bent-knee local optimum, non_foot_body_ground_penalty (femur < 5 cm, -30), Principle: make the local optimum economically unviable instead of reseeding, Research Progress (running summary), Deviations from specs D1-D33, Phase 1 Summary: CPG-guided flat-ground tripod gait

### Community 77 - "Coxa Contact & pi_S0a"
Cohesion: 0.33
Nodes (6): cpg_coxa_contact_reward (contact-conditioned propulsion), cpg_stance_rigidity_penalty, Signed, clipped cpg_coxa_contact, Friction combine multiply, ground mu 0.75 (D7), pi_base (Phase 1 policy, model_14998), pi_S0a checkpoint (model_4498, seed 42)

### Community 78 - "Isaac Sim Screenshot"
Cohesion: 0.40
Nodes (6): Isaac Sim screenshot: pxmark4 USD in Robot Inspector, IsaacRobotAPI isaac:physics:robotJoints list (coxa/femur joints per leg), Leg/joint naming order: right_front, right_middle, right_back, left_front, left_middle, left_back coxa, then femur, pxmark4 USD defaultPrim (PhantomX Mark IV hexapod), NVIDIA RTX A4000 GPU (viewport 118 FPS at 1280x720), RTX Real-Time 2.0 renderer mode (Isaac Sim viewport, PhysX)

### Community 79 - "Probe Loss Spec"
Cohesion: 0.33
Nodes (6): β-NLL Probe Loss (β=0.5), Universal Probe Target Set (186-D), Constant Probe Weight S1-S3, Fu, Cheng, Pathak 2022 Deep Whole-Body Control (ROA), Seitzer et al. 2022 β-NLL, ROA Not Applicable

### Community 80 - "Warm Start & KL Anchor"
Cohesion: 0.33
Nodes (6): From-scratch Control Arm, Function-Preserving Input Widening W' = [W | 0], KL Anchor (void since 2026-09-23), Latent Drift Mitigation & Normaliser Reset, π_S0, Schmitt et al. 2018 Kickstarting

### Community 81 - "Diagnostics & Contact Switches"
Cohesion: 0.33
Nodes (6): Foot contact switches on all six legs, Lead-time metric, Per-horizon remaining lead time rem_k, Standing diagnostics table, Position tracking error q_cmd - q_meas, V trust tests (slip-stratified error, IMU dropout, slip weighting)

### Community 82 - "Arms & Stage Schedule"
Cohesion: 0.40
Nodes (5): Arms O, B0, B1-K, B1-M, B2, F, B3, Blind O vs B0 headroom test gate, Paired t-test on seed-level F - B2 differences, S1 blind trunks (5 seeds, W=50), Stage schedule S0 -> S1 -> S1b -> S2 -> S3

### Community 83 - "S1 Task Components"
Cohesion: 0.50
Nodes (5): Separate critic with contract P (683-D), Hexapod-S1-v0 task, Gradient-isolation and widening merge-gating tests, TCN window W=50, Universal probe set (186-D, beta-NLL)

### Community 84 - "Project Stack"
Cohesion: 0.40
Nodes (5): hexasim hexapod RL project, Isaac Lab / Isaac Sim, rsl_rl PPO, hexasim extension CHANGELOG, 0.1.0 initial Isaac Lab extension template

### Community 85 - "Crutch Legs & Mirror"
Cohesion: 0.50
Nodes (5): Right-side crutch legs inherited from pi_base, Friction combine multiply, ground mu 0.75, Left-right mirror augmentation (D21), pi_base (Phase 1 policy model_14998.pt), pi_S0a (S0a run 4, model_4498.pt)

### Community 86 - "Leg Proximity & Clip"
Cohesion: 0.50
Nodes (4): action_beyond_clip penalty, Coxa clip term imperfection (D30), leg_gap_probe.py capsule gap measurement, leg_proximity_penalty (capsule gap, 2 cm margin)

### Community 87 - "Aux Config"
Cohesion: 0.50
Nodes (3): AuxCfg, mlp(), configclass

### Community 88 - "Rewards Config"
Cohesion: 0.50
Nodes (4): Locomotion reward shaping for the PhantomX Mark IV. Term weights — design…, S0a's terms plus command tracking, the phase-free floors (spec §5) and the…, RewardsCfg, S0bRewardsCfg

### Community 89 - "Velocity Command"
Cohesion: 0.50
Nodes (3): HexapodVelocityCommand, Segments: ``rel_turn_envs`` turn on the spot (v_x = v_y = 0, |yaw| ~…, UniformVelocityCommand

### Community 90 - "Evaluation Protocol"
Cohesion: 0.50
Nodes (4): Evaluation Protocol (§10.3), Held-out Harder-Terrain Suite (levels 15-20), Agarwal et al. 2021 Statistical Precipice (rliable), Terrain Generator / Material Library / Curriculum (levels 0-14)

### Community 91 - "RSL-RL Arg Parser"
Cohesion: 0.67
Nodes (3): ArgumentParser, add_rsl_rl_args(), Add RSL-RL arguments to the parser. Args: parser: The parser to add the…

## Ambiguous Edges - Review These
- `leg_proximity_penalty (capsule gap, 2 cm margin)` → `Coxa clip term imperfection (D30)`  [AMBIGUOUS]
  research/phase2-progress.md · relation: conceptually_related_to

## Knowledge Gaps
- **103 isolated node(s):** `isaaclab-dev`, `record_all.sh script`, `eval_terrain_sweep.sh script`, `milestone_review.sh script`, `run_s1_sweep_and_seeds.sh script` (+98 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 467 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `leg_proximity_penalty (capsule gap, 2 cm margin)` and `Coxa clip term imperfection (D30)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Phase 2 Run Logs (archive)` connect `Phase 2 Run Log Findings` to `Obs Equivalence Diagnosis`, `S0 Gait Evaluation`, `Isaac Lab Utilities`, `Phase 2.0 Correctness Fixes`, `Gait Probe Scripts`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `Phase 2 Baseline Paradigm Development Log` connect `Phase 2.0 Correctness Fixes` to `Phase 2.1 S0b Development`, `Phase 2.2 S1 Infrastructure`, `Phase 2 Run Log Findings`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `Phase 2.1: S0b commands, clock, standing, terrain, DR` connect `Phase 2.1 S0b Development` to `S1 Budget & Seeds`, `Policy Checkpoint Lineage`, `Phase 2.0 Correctness Fixes`, `Phase 2.2 S1 Infrastructure`, `Crutch Legs & Mirror`, `Leg Proximity & Clip`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **What connects `isaaclab-dev`, `record_all.sh script`, `eval_terrain_sweep.sh script` to the rest of the system?**
  _103 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dashboard Architecture & Arms` be split into smaller, more focused modules?**
  _Cohesion score 0.0502283105022831 - nodes in this community are weakly interconnected._
- **Should `Phase 2 Run Log Findings` be split into smaller, more focused modules?**
  _Cohesion score 0.05704365079365079 - nodes in this community are weakly interconnected._