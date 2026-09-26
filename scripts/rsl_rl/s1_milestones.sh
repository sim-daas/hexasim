#!/usr/bin/env bash
# S1 pilot milestones (phase2-progress §2.2.5, §2.2.7). At every 2,000 iterations: the standard review (videos,
# swing-radial and leg-gap probes, reward slopes; milestone_review.sh), the S0b gate set on Hexapod-S1-v0 with the
# checkpoint's own aux / ê gate / V_0 dropout (V tests and probe losses in each JSON), and the gait-by-material probe
# (coxa range, lift per leg type, leg gaps, per-term reward). The π_S0 baseline on the S1 env runs first if missing.
#
# Plateau stop (user decision, §2.2.7): from MIN_STOP on, if Train/mean_reward gained < STOP_PCT % over the last
# 2,000 iterations, training on tmux 0:2 is stopped and <milestones>/BUDGET records the iteration (the S1 budget for
# every seed). Gait regressions are reviewed by hand at every milestone.
# Usage: scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0.pt> [last_iter=12000] [min_stop=4000] [stop_pct=1.5]
set -u
RUN=$1; PI=$2; LAST=${3:-12000}; MIN_STOP=${4:-4000}; STOP_PCT=${5:-1.5}
NAME=$(basename "$RUN"); OUT=logs/milestones/$NAME; mkdir -p "$OUT"
export TASK=Hexapod-S1-v0
export OPENBLAS_NUM_THREADS=1  # Kit start-up fork vs OpenBLAS at-fork crash (phase2-progress §2.2.5)
[ -f logs/eval/s1base_v0.15_L0_nodrops.json ] || scripts/rsl_rl/eval_s0b_gates.sh "$PI" s1base 256 > logs/eval_s1base.txt 2>&1
for IT in $(seq 2000 2000 "$LAST"); do
  CKIT=$IT; [ "$IT" -eq "$LAST" ] && CKIT=$((LAST - 1))
  until [ -f "$RUN/model_$CKIT.pt" ]; do sleep 60; done
  sleep 10
  scripts/rsl_rl/milestone_review.sh "$RUN" "$CKIT"
  if [ "$IT" -ge "$MIN_STOP" ]; then
    GAIN=$(awk '/Train\/mean_reward/ {printf "%.3f", ($4 / $8 - 1) * 100}' "$OUT/it${CKIT}_slope.txt")
    if awk -v g="$GAIN" -v p="$STOP_PCT" 'BEGIN {exit !(g < p)}'; then
      tmux send-keys -t 0:2 C-c
      echo "$CKIT (mean_reward gain ${GAIN} % < ${STOP_PCT} % over 2,000 iterations)" > "$OUT/BUDGET"
    fi
  fi
  scripts/rsl_rl/eval_s0b_gates.sh "$RUN/model_$CKIT.pt" "$(basename "$OUT")_it$IT" 256 > "$OUT/it${IT}_gates.txt" 2>&1
  python scripts/tests/gait_by_material.py --headless --cmd 0.15 "it$IT=$RUN/model_$CKIT.pt" 2>&1 \
    | grep -E 'MATGAIT|Traceback' > "$OUT/it${IT}_matgait.txt"
  touch "$OUT/.done_it$IT"
  [ -f "$OUT/BUDGET" ] && break
done
