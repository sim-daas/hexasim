# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import hexasim.tasks  # noqa: F401
from hexasim.tasks.manager_based.hexasim.mdp.rewards import toe_pos_w  # noqa: E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.get_observations()
        timestep = 0
        max_timesteps = 500  # 10 s at 50 Hz

        import csv
        import math
        import statistics

        debug_log_path = os.path.join(log_dir, "debug_full_episode.csv")
        csv_file = open(debug_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)

        header = [
            "timestep",
            "time_s",
            "pos_x",
            "pos_y",
            "pos_z",
            "disp_x",
            "disp_y",
            "disp_norm",
            "vel_b_x",
            "vel_b_y",
            "vel_b_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
            "a_z",
            "alpha_x",
            "alpha_y",
            "delta_phase_actual",
            "delta_phase_expected",
        ]
        for i in range(6):
            header.extend(
                [
                    f"leg_{i}_phase",
                    f"leg_{i}_tibia_z",
                    f"leg_{i}_force_z",
                    f"leg_{i}_is_swing",
                    f"leg_{i}_target_z",
                    f"leg_{i}_z_error",
                ]
            )
        csv_writer.writerow(header)

        # Unwrap env handles once
        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene["contact_forces"]
        tibia_body_ids = [i for i, n in enumerate(robot.data.body_names) if "tibia" in n]

        # Accumulators for post-loop summary
        logged_rows = []
        initial_pos = None
        prev_vz = None
        prev_ang_vel = None
        prev_phase_0 = None

        # Constants
        STEP_HEIGHT = 0.03
        swing_end = math.pi  # phase < pi → swing  (duty=0.5)
        dt = env.unwrapped.step_dt
        STRIDE_LEN = 0.20  # stride_length matching training
        MAX_F = 2.5

        # simulate environment
        try:
            while timestep < max_timesteps:
                start_time = time.time()

                with torch.inference_mode():
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

                # --- extract state (env 0 only) ---
                pos_w = robot.data.root_pos_w[0].cpu().tolist()
                vel_b = robot.data.root_lin_vel_b[0].cpu().tolist()
                ang_vel = robot.data.root_ang_vel_b[0].cpu().tolist()
                cpg_phase = env.unwrapped.cpg_phase[0].cpu().tolist()

                tibia_z_w = toe_pos_w(robot, tibia_body_ids)[0, :, 2].cpu().tolist()  # toe point, not tibia origin
                forces_w = contact_sensor.data.net_forces_w[0]
                force_z = [forces_w[j, 2].item() for j in range(6)]

                baseline = env.unwrapped.cpg_foot_baseline[0].cpu().tolist()

                if initial_pos is None:
                    initial_pos = pos_w[:]

                # --- derived scalars ---
                disp_x = pos_w[0] - initial_pos[0]
                disp_y = pos_w[1] - initial_pos[1]
                disp_norm = math.hypot(disp_x, disp_y)
                time_s = timestep * dt

                # Finite-difference accelerations
                if prev_vz is None:
                    a_z = alpha_x = alpha_y = 0.0
                else:
                    a_z = (vel_b[2] - prev_vz) / dt
                    alpha_x = (ang_vel[0] - prev_ang_vel[0]) / dt
                    alpha_y = (ang_vel[1] - prev_ang_vel[1]) / dt
                prev_vz = vel_b[2]
                prev_ang_vel = ang_vel[:]

                # CPG phase-speed verification
                if prev_phase_0 is None:
                    delta_phase_actual = 0.0
                else:
                    delta_phase_actual = (cpg_phase[0] - prev_phase_0 + 2 * math.pi) % (2 * math.pi)
                prev_phase_0 = cpg_phase[0]
                vx_pos = max(0.0, vel_b[0])
                f_expected = max(0.2, min(MAX_F, vx_pos / STRIDE_LEN))
                delta_phase_expected = 2 * math.pi * f_expected * dt

                # --- per-leg CPG metrics ---
                leg_cols = []
                for i in range(6):
                    phi = cpg_phase[i]
                    is_swing = int(phi < swing_end)
                    if is_swing:
                        swing_norm = (phi / swing_end) * math.pi
                        target_z = STEP_HEIGHT * math.sin(swing_norm)
                        z_error = abs(tibia_z_w[i] - (target_z + baseline[i])) / STEP_HEIGHT
                    else:
                        target_z = 0.0
                        z_error = 0.0
                    leg_cols.extend([phi, tibia_z_w[i], force_z[i], is_swing, target_z, z_error])

                # --- write CSV row ---
                row = [
                    timestep,
                    round(time_s, 4),
                    *[round(v, 5) for v in pos_w],
                    round(disp_x, 5),
                    round(disp_y, 5),
                    round(disp_norm, 5),
                    *[round(v, 5) for v in vel_b],
                    *[round(v, 5) for v in ang_vel],
                    round(a_z, 5),
                    round(alpha_x, 5),
                    round(alpha_y, 5),
                    round(delta_phase_actual, 5),
                    round(delta_phase_expected, 5),
                    *[round(v, 5) for v in leg_cols],
                ]
                csv_writer.writerow(row)

                # accumulate for post-loop summary
                logged_rows.append(
                    {
                        "vx": vel_b[0],
                        "disp_x": disp_x,
                        "disp_y": disp_y,
                        "disp_norm": disp_norm,
                        "a_z": a_z,
                        "alpha_x": alpha_x,
                        "alpha_y": alpha_y,
                        "dp_actual": delta_phase_actual,
                        "dp_expected": delta_phase_expected,
                        "swing": [leg_cols[i * 6 + 3] for i in range(6)],
                        "force_z": [leg_cols[i * 6 + 2] for i in range(6)],
                        "z_err": [leg_cols[i * 6 + 5] for i in range(6)],
                    }
                )

                timestep += 1

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass

        csv_file.close()

        # --- post-episode summary ---
        if logged_rows:
            n = len(logged_rows)
            last = logged_rows[-1]
            mean_vx = sum(r["vx"] for r in logged_rows) / n
            max_vx = max(r["vx"] for r in logged_rows)

            dp_act = [r["dp_actual"] for r in logged_rows[1:]]
            dp_exp = [r["dp_expected"] for r in logged_rows[1:]]
            mean_dpa = sum(dp_act) / len(dp_act) if dp_act else 0.0
            mean_dpe = sum(dp_exp) / len(dp_exp) if dp_exp else 0.0
            ratio = mean_dpa / mean_dpe if mean_dpe else 0.0

            az_vals = [r["a_z"] for r in logged_rows[1:]]
            ax_vals = [r["alpha_x"] for r in logged_rows[1:]]
            ay_vals = [r["alpha_y"] for r in logged_rows[1:]]
            az_std = statistics.stdev(az_vals) if len(az_vals) > 1 else 0.0
            ax_range = (min(ax_vals), max(ax_vals)) if ax_vals else (0.0, 0.0)
            ay_range = (min(ay_vals), max(ay_vals)) if ay_vals else (0.0, 0.0)

            leg_names = ["right_front", "right_middle", "right_back", "left_front", "left_middle", "left_back"]
            leg_tripods = ["A", "B", "A", "B", "A", "B"]

            print("\n" + "=" * 80)
            print(f"  FULL-EPISODE KINEMATICS & GAIT METRICS  ({n} steps / {n * dt:.1f} s)")
            print("=" * 80)
            print(
                f"Displacement: dX={last['disp_x']:.3f} m  dY={last['disp_y']:.3f} m  |total|={last['disp_norm']:.3f} m"
            )
            print(f"Mean forward velocity: {mean_vx:.3f} m/s  (max {max_vx:.3f} m/s)")
            print()
            print("--- CPG Phase-Speed Verification ---")
            print(f"Mean delta_phase_actual:   {mean_dpa:.4f} rad/step")
            print(f"Mean delta_phase_expected: {mean_dpe:.4f} rad/step  (stride={STRIDE_LEN})")
            print(f"Ratio actual/expected:     {ratio:.3f}  (1.00 = matched; >1.10 → stride too short)")
            print()
            print("--- Stability Metrics (research-paper thresholds) ---")
            print(f"Vertical acceleration  a_z  : std={az_std:.4f} m/s²   [target < 0.148]")
            print(f"Roll  acceleration alpha_x  : min={ax_range[0]:.3f}  max={ax_range[1]:.3f} rad/s²  [target ±0.9]")
            print(f"Pitch acceleration alpha_y  : min={ay_range[0]:.3f}  max={ay_range[1]:.3f} rad/s²  [target ±0.8]")
            print()
            print("--- Per-Leg Gait Activity ---")
            print(f"{'Leg':>3}  {'Name':<16} {'Tripod':>6} {'Swing%':>7} {'MeanF_z(N)':>10} {'MeanSwingZ_err':>14}")
            print("-" * 62)
            for i in range(6):
                swing_steps = [r["swing"][i] for r in logged_rows]
                force_steps = [r["force_z"][i] for r in logged_rows]
                swing_err = [r["z_err"][i] for r in logged_rows if r["swing"][i]]
                sw_pct = sum(swing_steps) / n * 100
                mean_fz = sum(force_steps) / n
                mean_err = sum(swing_err) / len(swing_err) if swing_err else float("nan")
                print(
                    f"{i:>3}  {leg_names[i]:<16} {leg_tripods[i]:>6} {sw_pct:>6.1f}% {mean_fz:>10.2f} {mean_err:>14.3f}"
                )
            print()

        env.close()


if __name__ == "__main__":
    main()
