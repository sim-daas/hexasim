#!/usr/bin/env bash
# S0b gate evaluation (paradigm spec §4.2): 256 deterministic episodes per condition, JSON to logs/eval/.
# 256 episodes per condition meets the spec's ≥ 200 floor; the final π_S0 evaluation uses 1,024 (95 % upper bound on
# a 0 % fall rate: 1.2 % → 0.3 %), which costs no extra wall time — the episodes run in parallel.
#   tracking + stride + falls at v_x ∈ {0.10, 0.15} on terrain levels 0, 5 and 9; drops vs no drops at level 0;
#   standing (zero command) on levels 0, 5 and 9; turning (D19): on the spot ±0.3 rad/s and walk 0.10 m/s + 0.2 rad/s
#   on levels 0 and 5.
# Usage: scripts/rsl_rl/eval_s0b_gates.sh <checkpoint> <tag> [num_envs]   (num_envs = episodes per condition, 256)
# EXTRA="..." appends eval_s0.py flags to every condition, e.g. EXTRA="--servo max_delay=3 --drop_k_mult 2".
set -euo pipefail
CK=$1
TAG=$2
N=${3:-256}
mkdir -p logs/eval
ev() { python scripts/rsl_rl/eval_s0.py --headless --task "${TASK:-Hexapod-S0b-v0}" --ckpt "$CK" --num_envs "$N" \
  --out "logs/eval/${TAG}_$1.json" "${@:2}" ${EXTRA:-} > /dev/null 2>&1; }
for L in 0 5 9; do
  for V in 0.10 0.15; do
    ev "v${V}_L${L}" --cmd "$V" 0 0 --terrain_level "$L"
  done
  ev "stand_L${L}" --cmd 0 0 0 --terrain_level "$L"
done
for L in 0 5; do
  ev "turn+0.3_L${L}" --cmd 0 0 0.3 --terrain_level "$L"
  ev "turn-0.3_L${L}" --cmd 0 0 -0.3 --terrain_level "$L"
  ev "walkturn_L${L}" --cmd 0.10 0 0.2 --terrain_level "$L"
done
for V in 0.10 0.15; do
  ev "v${V}_L0_nodrops" --cmd "$V" 0 0 --terrain_level 0 --no_drops
done
python - "$TAG" <<'EOF'
import glob, json, sys
tag = sys.argv[1]
keys = ["fall_rate", "vx_mean", "vx_rmse_cycle", "vx_rmse_step", "stride", "stride_ref", "duty_factor", "clearance",
        "coxa_hf_power", "v0_rmse", "stand_drift_m", "stand_joint_speed", "wz_mean", "wz_rmse_cycle"]
for f in sorted(glob.glob(f"logs/eval/{tag}_*.json")):
    m = json.load(open(f))
    print(f.split(tag + "_")[1][:-5].ljust(18), " ".join(f"{k}={m[k]:.4f}" for k in keys if k in m))
EOF
