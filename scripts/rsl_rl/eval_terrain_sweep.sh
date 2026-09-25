#!/usr/bin/env bash
# Terrain stress sweep (phase2-progress §2.1.20), evaluation only: each terrain type on its own, 15 levels (0–9 as
# in training, 10–14 harder), v_x ∈ {0.10, 0.15}, 128 deterministic episodes per cell. JSON to logs/eval/sweep_<tag>_*.
# Usage: scripts/rsl_rl/eval_terrain_sweep.sh <checkpoint> <tag> [levels...]
set -euo pipefail
CK=$1
TAG=$2
shift 2
LEVELS=${*:-0 3 6 9 10 11 12 13 14}
mkdir -p logs/eval
for T in rough slope_up slope_down bumps stairs_up stairs_down; do
  for L in $LEVELS; do
    for V in 0.10 0.15; do
      python scripts/rsl_rl/eval_s0.py --headless --task "${TASK:-Hexapod-S0b-v0}" --ckpt "$CK" --num_envs 128 \
        --terrain_type "$T" --terrain_level "$L" --cmd "$V" 0 0 \
        --out "logs/eval/sweep_${TAG}_${T}_L${L}_v${V}.json" > /dev/null 2>&1 || true
      # Kit can exit 0 after a Python error: judge a cell by its JSON, not the exit code.
      [ -f "logs/eval/sweep_${TAG}_${T}_L${L}_v${V}.json" ] || echo "MISSING ${T} L${L} v${V}"
    done
  done
done
python - "$TAG" <<'PY'
import glob, json, re, sys
tag = sys.argv[1]
rows = {}
for f in glob.glob(f"logs/eval/sweep_{tag}_*.json"):
    t, lvl, v = re.search(rf"sweep_{tag}_(.+)_L(\d+)_v([\d.]+)\.json", f).groups()
    m = json.load(open(f))
    ok = m["fall_rate"] <= 0.05 and m["vx_rmse_cycle"] <= 0.03 and abs(m["stride"] / m["stride_ref"] - 1) <= 0.10
    rows[(t, int(lvl), v)] = f"{m['fall_rate']*100:4.0f}% {m['vx_rmse_cycle']:.3f} {m['stride']/m['stride_ref']-1:+.0%} {'PASS' if ok else 'FAIL'}"
print("type         level  v     falls  cycRMSE stride  gates")
for k in sorted(rows):
    print(f"{k[0]:12s} {k[1]:5d}  {k[2]}  {rows[k]}")
PY
