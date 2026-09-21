#!/usr/bin/env bash
# S0b gate evaluation (paradigm spec §4.2): 256 deterministic episodes per condition, JSON to logs/eval/.
#   tracking + stride + falls at v_x ∈ {0.10, 0.15} on terrain levels 0 and 5; drops vs no drops at level 0;
#   standing (zero command) on levels 0 and 5.
# Usage: scripts/rsl_rl/eval_s0b_gates.sh <checkpoint> <tag>
set -euo pipefail
CK=$1
TAG=$2
mkdir -p logs/eval
ev() { python scripts/rsl_rl/eval_s0.py --headless --task Hexapod-S0b-v0 --ckpt "$CK" --out "logs/eval/${TAG}_$1.json" "${@:2}" > /dev/null 2>&1; }
for L in 0 5; do
  for V in 0.10 0.15; do
    ev "v${V}_L${L}" --cmd "$V" 0 0 --terrain_level "$L"
  done
  ev "stand_L${L}" --cmd 0 0 0 --terrain_level "$L"
done
for V in 0.10 0.15; do
  ev "v${V}_L0_nodrops" --cmd "$V" 0 0 --terrain_level 0 --no_drops
done
python - "$TAG" <<'EOF'
import glob, json, sys
tag = sys.argv[1]
keys = ["fall_rate", "vx_mean", "vx_rmse_cycle", "vx_rmse_step", "stride", "stride_ref", "duty_factor", "clearance",
        "coxa_hf_power", "v0_rmse", "stand_drift_m", "stand_joint_speed"]
for f in sorted(glob.glob(f"logs/eval/{tag}_*.json")):
    m = json.load(open(f))
    print(f.split(tag + "_")[1][:-5].ljust(18), " ".join(f"{k}={m[k]:.4f}" for k in keys if k in m))
EOF
