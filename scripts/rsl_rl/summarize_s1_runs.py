# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarise final S1 evaluations of several runs side by side (phase2-progress §2.2.9–2.2.10).

Per run, from files written by `eval_s0b_gates.sh` (logs/eval/<tag>_*.json), `gait_by_material.py`
(<milestones>/it8000_matgait.txt), `leg_gap_probe.py` / `swing_radial_probe.py` (<milestones>/it7999_*.txt) and the
run's TensorBoard events (V tests and probe losses over iterations 6,000–8,000). Prints one markdown table with the
S0b gate numbers, V (v7 §4.3), probe NLLs and the gait probes, plus mean ± sd over the runs flagged as seeds.

Usage: python scripts/rsl_rl/summarize_s1_runs.py name=<run_dir>:<eval_tag> [...] [--seeds name,name,...]
No simulator needed.
"""

import argparse
import glob
import json
import os
import re
import statistics as st

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help="name=<run_dir>:<eval_tag>")
parser.add_argument("--seeds", default="", help="comma-separated run names to aggregate as seeds")
args = parser.parse_args()

WALK = [f"v{v}_L{lv}" for v in ("0.10", "0.15") for lv in (0, 5, 9)]
TB = [
    "Train/mean_reward",
    "Loss/vtest_v_rmse",
    "Loss/vtest_v_rmse_rigid",
    "Loss/vtest_v_rmse_slip",
    "Loss/vtest_v_rmse_imu_zero",
    "Loss/aux_probe_cur",
    "Loss/aux_probe_td",
]


def tb(run_dir):
    by = {}
    for f in sorted(glob.glob(run_dir + "/events*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in TB:
            if t in ea.Tags()["scalars"]:
                by.setdefault(t, {}).update({s.step: s.value for s in ea.Scalars(t)})
    out = {}
    for t in TB:
        v = [x for s, x in by.get(t, {}).items() if 6000 <= s < 8000]
        out[t] = st.mean(v) if v else float("nan")
    gate = [s for s, v in by.get("Loss/vtest_v_rmse", {}).items()]
    out["first_vtest"] = min(gate) if gate else -1
    return out


def row(name, run_dir, tag):
    g = lambda c: json.load(open(f"logs/eval/{tag}_{c}.json"))  # noqa: E731
    files = glob.glob(f"logs/eval/{tag}_*.json")
    r = {"name": name, "cells": len(files)}
    r["falls_max"] = max(json.load(open(f))["fall_rate"] for f in files) if files else float("nan")
    walk = [g(c) for c in WALK]
    r["stride_err_010"] = [100 * (m["stride"] / m["stride_ref"] - 1) for m in walk[:3]]
    r["stride_err_015"] = [100 * (m["stride"] / m["stride_ref"] - 1) for m in walk[3:]]
    r["rmse_max"] = max(m["vx_rmse_cycle"] for m in walk)
    nd = [g(f"v{v}_L0_nodrops")["vx_rmse_cycle"] for v in ("0.10", "0.15")]
    r["drops_vs_nodrops"] = (walk[0]["vx_rmse_cycle"], nd[0], walk[3]["vx_rmse_cycle"], nd[1])
    t = [g("turn+0.3_L0"), g("turn-0.3_L0"), g("walkturn_L0")]
    r["turn"] = (t[0]["wz_mean"], t[1]["wz_mean"], max(t[0]["wz_rmse_cycle"], t[1]["wz_rmse_cycle"]))
    r["walkturn_yaw_rmse"] = t[2]["wz_rmse_cycle"]
    r["stand"] = [g(f"stand_L{lv}")["duty_factor"] for lv in (0, 5, 9)]
    r["apex"] = (walk[0]["clearance"], walk[2]["clearance"])
    v = walk + [g(c) for c in ("turn+0.3_L0", "walkturn_L0")]
    r["V_rmse"] = (min(m["vtest_v_rmse"] for m in walk), max(m["vtest_v_rmse"] for m in walk))
    r["V_slip_ratio"] = max(m["vtest_v_rmse_slip"] / m["vtest_v_rmse_rigid"] for m in walk)
    r["V_imu_ratio"] = min(m["vtest_v_rmse_imu_zero"] / m["vtest_v_rmse"] for m in walk)
    del v
    ms = f"logs/milestones/{os.path.basename(run_dir)}"
    mg = open(f"{ms}/it8000_matgait.txt").read()
    lifts = re.findall(r"grippy\s+lift f/m/b ([\d.]+) ([\d.]+) ([\d.]+) cm \| coxa range f/m/b\s+([\d.]+)\s+([\d.]+)"
                       r"\s+([\d.]+) deg \| gap min ([-+\d.]+) p1 [-+\d.]+ cm <2cm ([\d.]+)% \| clip ([\d.]+)", mg)
    if lifts:
        x = [float(a) for a in lifts[0]]
        r["lift_fmb"], r["coxa_fmb"], r["gap_mat"], r["lt2_mat"], r["clip"] = x[0:3], x[3:6], x[6], x[7], x[8]
    land = re.findall(r"landing \|r\| f/m/b\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", mg)
    r["landing"] = [float(a) for a in land[0]] if land else None
    gap = open(f"{ms}/it7999_gap.txt").read()
    wg = re.findall(r"worst surface gap ([-+\d.]+) cm", gap)
    r["gap_worst"] = float(wg[0]) if wg else float("nan")
    lt = re.findall(r"fwd0\.15.*", gap)
    r["lt2_015"] = max(float(p) for p in re.findall(r"<2cm\s+([\d.]+)%", lt[0])) if lt else float("nan")
    rad = open(f"{ms}/it7999_radial.txt").read()
    lf = re.findall(r"lift\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", rad)
    r["lift_swing"] = [float(a) for a in lf[0]] if lf else None
    r.update(tb(run_dir))
    return r


rows = []
for spec in args.runs:
    name, rest = spec.split("=", 1)
    run_dir, tag = rest.rsplit(":", 1)
    rows.append(row(name, run_dir, tag))

f2 = lambda xs: " / ".join(f"{x:+.1f}" for x in xs)  # noqa: E731
print("| Metric | " + " | ".join(r["name"] for r in rows) + " |")
print("| --- | " + " | ".join("---" for _ in rows) + " |")
lines = [
    ("Gate cells / max fall rate", lambda r: f"{r['cells']} / {100 * r['falls_max']:.1f} %"),
    ("Stride err 0.10 L0/L5/L9 (%)", lambda r: f2(r["stride_err_010"])),
    ("Stride err 0.15 L0/L5/L9 (%)", lambda r: f2(r["stride_err_015"])),
    ("Cycle RMSE max (m/s)", lambda r: f"{r['rmse_max']:.4f}"),
    ("Drops vs no drops 0.10, 0.15", lambda r: "{:.4f}/{:.4f}, {:.4f}/{:.4f}".format(*r["drops_vs_nodrops"])),
    ("Turn ±0.3 ω̄_z; RMSE max", lambda r: "{:+.3f} / {:+.3f}; {:.3f}".format(*r["turn"])),
    ("Walk-and-turn yaw RMSE", lambda r: f"{r['walkturn_yaw_rmse']:.3f}"),
    ("Standing contact L0/L5/L9", lambda r: " / ".join(f"{x:.2f}" for x in r["stand"])),
    ("Gate apex clearance L0 / L9 (cm)", lambda r: f"{100 * r['apex'][0]:.1f} / {100 * r['apex'][1]:.1f}"),
    ("V v̂ RMSE walking (m/s)", lambda r: "{:.3f}–{:.3f}".format(*r["V_rmse"])),
    ("V slip/rigid (max); IMU-zero ratio (min)", lambda r: f"{r['V_slip_ratio']:.2f} ×; {r['V_imu_ratio']:.1f} ×"),
    ("Train 6k–8k: V slip RMSE", lambda r: f"{r['Loss/vtest_v_rmse_slip']:.4f}"),
    ("Train 6k–8k: V rigid RMSE", lambda r: f"{r['Loss/vtest_v_rmse_rigid']:.4f}"),
    ("Train 6k–8k: probe cur / td NLL", lambda r: f"{r['Loss/aux_probe_cur']:.4f} / {r['Loss/aux_probe_td']:.4f}"),
    ("Train 6k–8k: mean reward", lambda r: f"{r['Train/mean_reward']:.1f}"),
    ("Worst leg gap (cm); <2 cm at 0.15", lambda r: f"{r['gap_worst']:+.1f}; {r['lt2_015']:.1f} %"),
    ("Lift f/m/b, material probe (cm)", lambda r: " / ".join(f"{x:.1f}" for x in r.get("lift_fmb", [0, 0, 0]))),
    ("Lift RF RM RB LF LM LB, swing probe", lambda r: " ".join(f"{x:.1f}" for x in (r["lift_swing"] or []))),
    ("Landing reach f/m/b (cm)", lambda r: " / ".join(f"{x:.1f}" for x in (r["landing"] or []))),
    ("Coxa range f/m/b (°)", lambda r: " / ".join(f"{x:.0f}" for x in r.get("coxa_fmb", [0, 0, 0]))),
    ("Coxa clip term (rad/step)", lambda r: f"{r.get('clip', float('nan')):.4f}"),
]
for label, fn in lines:
    print(f"| {label} | " + " | ".join(fn(r) for r in rows) + " |")

seeds = [r for r in rows if r["name"] in args.seeds.split(",")]
if len(seeds) >= 2:
    print(f"\nAcross {len(seeds)} seeds ({', '.join(r['name'] for r in seeds)}): mean ± sd [min, max]")
    def agg(label, vals):
        print(f"  {label:36s} {st.mean(vals):8.4f} ± {st.stdev(vals):.4f}  [{min(vals):.4f}, {max(vals):.4f}]")
    agg("max fall rate", [r["falls_max"] for r in seeds])
    agg("stride err 0.10 L0 (%)", [r["stride_err_010"][0] for r in seeds])
    agg("stride err 0.15 L0 (%)", [r["stride_err_015"][0] for r in seeds])
    agg("cycle RMSE max", [r["rmse_max"] for r in seeds])
    agg("turn +0.3 ω̄_z", [r["turn"][0] for r in seeds])
    agg("turn −0.3 ω̄_z", [r["turn"][1] for r in seeds])
    agg("standing contact L9", [r["stand"][2] for r in seeds])
    agg("V slip RMSE (train 6k–8k)", [r["Loss/vtest_v_rmse_slip"] for r in seeds])
    agg("V slip/rigid (eval max)", [r["V_slip_ratio"] for r in seeds])
    agg("V IMU-zero ratio (eval min)", [r["V_imu_ratio"] for r in seeds])
    agg("probe cur NLL (train 6k–8k)", [r["Loss/aux_probe_cur"] for r in seeds])
    agg("probe td NLL (train 6k–8k)", [r["Loss/aux_probe_td"] for r in seeds])
    agg("mean reward (train 6k–8k)", [r["Train/mean_reward"] for r in seeds])
    agg("worst leg gap (cm)", [r["gap_worst"] for r in seeds])
    agg("< 2 cm share at 0.15 (%)", [r["lt2_015"] for r in seeds])
    agg("front lift, material probe (cm)", [r["lift_fmb"][0] for r in seeds])
    agg("front landing reach (cm)", [r["landing"][0] for r in seeds])
    agg("front coxa range (°)", [r["coxa_fmb"][0] for r in seeds])
    agg("coxa clip term", [r["clip"] for r in seeds])
