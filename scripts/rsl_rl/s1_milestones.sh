#!/usr/bin/env bash
# S1 pilot milestones (phase2-progress §2.2): the π_S0 baseline on the S1 env first, then at every 2,000 iterations the
# standard review (videos, swing-radial and leg-gap probes, reward slopes; milestone_review.sh) and the S0b gate set
# on Hexapod-S1-v0 with the checkpoint's own aux / ê gate / V_0 dropout (V tests and probe losses in each JSON).
# Usage: scripts/rsl_rl/s1_milestones.sh <run_dir> <π_S0.pt> [last_iter=10000]
set -u
RUN=$1; PI=$2; LAST=${3:-10000}; NAME=$(basename "$RUN")
export TASK=Hexapod-S1-v0
export OPENBLAS_NUM_THREADS=1  # Kit start-up fork vs OpenBLAS at-fork crash (phase2-progress §2.2.5)
[ -f logs/eval/s1base_v0.15_L0_nodrops.json ] || scripts/rsl_rl/eval_s0b_gates.sh "$PI" s1base 256 > logs/eval_s1base.txt 2>&1
for IT in $(seq 2000 2000 "$LAST"); do
  CKIT=$IT; [ "$IT" -eq "$LAST" ] && CKIT=$((LAST - 1))
  until [ -f "$RUN/model_$CKIT.pt" ]; do sleep 60; done
  sleep 10
  scripts/rsl_rl/milestone_review.sh "$RUN" "$CKIT"
  scripts/rsl_rl/eval_s0b_gates.sh "$RUN/model_$CKIT.pt" "s1p_it$IT" 256 > "logs/milestones/$NAME/it${IT}_gates.txt" 2>&1
  touch "logs/milestones/$NAME/.done_it$IT"
done
