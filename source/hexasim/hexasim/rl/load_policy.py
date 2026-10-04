# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared checkpoint → env wiring for evaluation and probe scripts.

``task_for`` picks the task from the actor input width; ``configure`` makes the env cfg match the checkpoint (S1: the
aux window); ``load_env_state`` loads what the env owns (V_0, S1 aux, ê gate, V_0 dropout). Call it **before**
``env.reset()``: the first observation and the per-episode V_0 keep draws use the loaded state.
"""

from __future__ import annotations

TASK_BY_DIM = {
    75: "Hexapod-Flat-v0",
    114: "Hexapod-S0a-v0",
    117: "Hexapod-S0b-v0",
    1125: "Hexapod-S1-v0",
    1269: "Hexapod-S2-v0",
}


def task_for(ck: dict) -> str:
    return TASK_BY_DIM[ck["actor_state_dict"]["mlp.0.weight"].shape[1]]


def configure(cfg, ck: dict) -> None:
    if "s1_state" in ck:
        cfg.aux.window = ck["s1_state"]["window"]
        cfg.arm = ck["s1_state"].get("arm", "B0")  # S1 trunks predate the field (they are B0-layout)


def load_env_state(env, ck: dict) -> None:
    if "v0_state_dict" in ck:
        env.v0.load_state_dict(ck["v0_state_dict"])
        env.v0_train = False
    if "aux_state_dict" in ck:
        env.aux.load_state_dict(ck["aux_state_dict"])
        env.e_gate = 1.0 if ck["s1_state"]["gate_iter"] >= 0 else 0.0
        env.v0_drop_p = ck["s1_state"]["v0_drop_p"]
        env.clock_drop_p = ck["s1_state"].get("clock_drop_p", 0.0)


def eval_physx_buffers(cfg) -> None:
    """PhysX GPU buffers sized for evaluation (≤ 1,024 envs) instead of the 4,096-env training defaults, so evals fit
    next to a running S1 training (the defaults pre-allocate ≈ 1 GB regardless of env count, e.g. 671 MB of contact
    pairs). Physics is unchanged while the capacities suffice — PhysX warns on overflow; check eval logs for it."""
    p = cfg.sim.physics
    if p is None:  # Flat task: Isaac Lab defaults (S0a+ set a PhysxCfg)
        return
    p.gpu_max_rigid_contact_count = 2**20
    p.gpu_found_lost_pairs_capacity = 2**19
    p.gpu_found_lost_aggregate_pairs_capacity = 2**22
    p.gpu_total_aggregate_pairs_capacity = 2**19
    p.gpu_collision_stack_size = 2**24
    p.gpu_heap_capacity = 2**24
    p.gpu_temp_buffer_capacity = 2**22
