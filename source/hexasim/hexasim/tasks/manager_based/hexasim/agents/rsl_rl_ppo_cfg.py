# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..mdp.symmetry import mirror_augmentation, mirror_augmentation_s1


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16  # longer rollout horizon for legged locomotion
    max_iterations = 10000  # 10× more iterations for hexapod (was 150)
    save_interval = 50  # checkpoint every 50 iters — never lose a run
    experiment_name = "hexapod_flat"

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],  # 75-D obs (Hexapod-Flat-v0) / 18-D action
        activation="elu",
        obs_normalization=True,  # running-mean normalisation on observations
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class S0PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Extra knobs of :class:`hexasim.rl.s0_ppo.S0PPO` (paradigm spec §4.2, §8.1)."""

    class_name: str = "hexasim.rl.s0_ppo:S0PPO"
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    std_floor: float = 0.05
    norm_burnin_iters: int = 50
    actor_freeze_iters: int = 300
    actor_freeze_ev: float = 0.8


@configclass
class PPORunnerCfg_S0a(RslRlOnPolicyRunnerCfg):
    """S0a from π_base: 4,096 envs × 16 steps (the base run's batch; the spec's 8,192 × 16 is wrong for it)."""

    num_steps_per_env = 16
    max_iterations = 3000
    save_interval = 50
    experiment_name = "hexapod_s0a"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],  # must match π_base: only the input layer is widened (spec §4.3 item 6)
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),  # overwritten by the warm start
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],  # separate network, own normaliser (spec §3.5)
        activation="elu",
        obs_normalization=True,
    )
    algorithm = S0PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,  # unused: S0PPO sets per-network learning rates
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfg_S0b(PPORunnerCfg_S0a):
    """S0b from π_S0a (actor widened 114 → 117 by the twist_cmd block, zero columns); same batch and S0PPO.

    From run 14 on: left–right mirror data augmentation (``mdp.symmetry``, phase2-progress §2.1.27, D21) — every
    mini-batch is trained on its mirror image as well, so a turn to one side also teaches the other.
    """

    experiment_name = "hexapod_s0b"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=mirror_augmentation
        )


@configclass
class S1PpoAlgorithmCfg(S0PpoAlgorithmCfg):
    """Extra knobs of :class:`hexasim.rl.s1_ppo.S1PPO` (phase2-progress §2.2)."""

    class_name: str = "hexasim.rl.s1_ppo:S1PPO"
    v_test_every: int = 100
    v_test_passes: int = 3
    slip_ratio_max: float = 3.0  # v7 §4.3 test 1
    imu_zero_min: float = 1.5  # v7 §4.3 test 2 ("materially": provisional)
    v0_retire_delay: int = 1000
    v0_retire_iters: int = 1000
    clock_anneal_iters: int = 0  # S2 only: per-episode clock dropout 0 → 1 over this many iterations
    reburn_iters: int = 50  # S2 only: actor frozen after the branch
    ramp_terms: list = []  # redesign: reward terms whose weight ramps 1 → ``ramp_to`` × over ``ramp_iters`` iterations
    ramp_iters: int = 0
    ramp_to: float = 2.0


@configclass
class PPORunnerCfg_S1(PPORunnerCfg_S0b):
    """S1 from π_S0 (actor widened 117 → 1,125, zero columns; std carried at its 0.05 floor), fresh critic on
    sg[actor obs] ⊕ contract P, actor frozen until EV ≥ 0.8 and ≥ 300 iterations; mirror augmentation with the
    latents recomputed on the mirrored history."""

    experiment_name = "hexapod_s1"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "contract"]}
    algorithm = S1PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,  # unused: per-network learning rates
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=mirror_augmentation_s1
        )



@configclass
class PPORunnerCfg_S1R(PPORunnerCfg_S1):
    """Headroom redesign, S1 stage (phase2-progress §2.3.15): as S1, plus the redesign terms ramped 1 → 2 × over the
    first 30 % of the 8,000-iteration run."""

    experiment_name = "hexapod_s1r"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.ramp_terms = ["swing_clearance", "ang_vel_xy", "tilt_ground", "stance_slip", "touchdown_speed"]
        self.algorithm.ramp_iters = 2400


@configclass
class PPORunnerCfg_S2(PPORunnerCfg_S1):
    """Blind S2 arms (O / B0), branched from an S1 trunk with ``train.py --branch_from``."""

    experiment_name = "hexapod_s2"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.clock_anneal_iters = 1800  # 30 % of the 6,000-iteration headroom pilot (spec §5)
