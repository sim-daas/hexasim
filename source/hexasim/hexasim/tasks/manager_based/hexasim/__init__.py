# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Hexasim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexasim_env_cfg:HexasimEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

##
# PhantomX Mark IV hexapod — flat ground environment
##

gym.register(
    id="Hexapod-Flat-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Flat-Play-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Hexapod-Flat-Resume-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodFlatResumeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
gym.register(
    id="Hexapod-S0a-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS0aEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS0aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S0a",
    },
)

gym.register(
    id="Hexapod-S0a-Play-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS0aEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS0aEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S0a",
    },
)

gym.register(
    id="Hexapod-S0b-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS0bEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS0bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S0b",
    },
)

gym.register(
    id="Hexapod-S0b-Play-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS0bEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS0bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S0b",
    },
)

gym.register(
    id="Hexapod-S1-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S1",
    },
)

gym.register(
    id="Hexapod-S2-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S2",
    },
)

gym.register(
    id="Hexapod-S1-Play-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodS1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodS1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_S1",
    },
)
