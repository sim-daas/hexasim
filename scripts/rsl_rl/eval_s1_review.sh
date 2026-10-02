#!/usr/bin/env bash
# S1 review evaluations (phase2-progress §2.3), per S1 trunk, JSON to logs/eval/:
#   lat15   all 17 gate conditions at 15 ms command latency (3 physics steps, above the trained 5–10 ms) and 2× drop slope
#   lay<S>  walking 0.10 / 0.15 on levels 0, 5, 9 with material-field layout seed S (0 = training layout, 1 = held out),
#           incl. V tests and probe skill vs the mean predictor
# Usage: scripts/rsl_rl/eval_s1_review.sh <tag>=<checkpoint> ...
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 TASK=Hexapod-S1-v0
for spec in "$@"; do
  T=${spec%%=*}
  CK=${spec#*=}
  EXTRA="--servo min_delay=3 max_delay=3 --drop_k_mult 2" scripts/rsl_rl/eval_s0b_gates.sh "$CK" "review_${T}_lat15"
  for S in 0 1; do
    for L in 0 5 9; do
      for V in 0.10 0.15; do
        python scripts/rsl_rl/eval_s0.py --headless --task $TASK --ckpt "$CK" --num_envs 256 --cmd "$V" 0 0 \
          --terrain_level "$L" --layout_seed "$S" --out "logs/eval/review_${T}_lay${S}_v${V}_L${L}.json" > /dev/null 2>&1
      done
    done
  done
done
