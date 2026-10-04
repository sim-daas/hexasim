#!/usr/bin/env bash
# O vs B0 headroom evaluation (phase2-progress §2.3.7, soft-ground conditions D39), JSON to logs/eval/headroom_*:
# every arm on the same conditions with the same seed, 256 deterministic episodes each, clock as in the checkpoint
# (dropped after the anneal), material layout seed 1, then the pre-registered go / no-go against the arm tagged b0.
#   material  μ 0.25 strip on 15° slopes up / down (+ μ 0.9 control) and on flat (negative control)
#   soft      foam (k 3.2e3), soft_soil (k 6e2), wet_clay (k 1e3) strips at matched μ 0.8, flat and 15° up / down
#   shape     bumps, stairs up / down, rough at levels 9, 14 (training) and 15–20 (held out); the verdict uses only
#             the pre-registered 9, 14, 17, the other held-out levels are reported as secondary
#   gates     the 17 S0b gate conditions (eval_s0b_gates.sh)
# Usage: scripts/rsl_rl/eval_headroom.sh b0=<ckpt> omat=<ckpt> ogeo=<ckpt>
#        SUMMARY_ONLY=1 … reprints the verdict from the JSONs.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
mkdir -p logs/eval
ev() { python scripts/rsl_rl/eval_headroom.py --headless --ckpt "$CK" --out "logs/eval/headroom_${T}_$1.json" \
  "${@:2}" > /dev/null 2>&1 || true; }  # Kit can exit non-zero after writing; cells are judged by their JSON
for spec in "$@"; do
  T=${spec%%=*}
  CK=${spec#*=}
  [ -n "${SUMMARY_ONLY:-}" ] && continue
  for S in flat slope_up slope_down; do
    ev "mat_${S}_mu0.25" --terrain $S --strip 0.25 0.22 6
    ev "mat_${S}_mu0.90" --terrain $S --strip 0.9 0.85 6
    for K in foam=3.505 soft_soil=2.778 wet_clay=3.0; do
      ev "soft_${S}_${K%%=*}" --terrain $S --strip 0.8 0.75 "${K#*=}" --base 0.8 0.75 6
    done
    ev "soft_${S}_rigid" --terrain $S --strip 0.8 0.75 6 --base 0.8 0.75 6
  done
  for S in bumps stairs_up stairs_down rough; do
    for L in 9 14 15 16 17 18 19 20; do
      ev "shape_${S}_L${L}" --terrain $S --level $L
    done
  done
  TASK=$(python -c "import torch,sys; print({1125:'Hexapod-S1-v0',1269:'Hexapod-S2-v0'}[torch.load(sys.argv[1],map_location='cpu',weights_only=False)['actor_state_dict']['mlp.0.weight'].shape[1]])" "$CK") \
    scripts/rsl_rl/eval_s0b_gates.sh "$CK" "headroom_${T}_gate" > "logs/eval/headroom_${T}_gates.txt"
done
python - "$@" <<'PY'
import glob, json, sys
import numpy as np

rng = np.random.default_rng(0)
tags = [s.split("=")[0] for s in sys.argv[1:]]
load = lambda t: {f.split(f"headroom_{t}_")[1][:-5]: json.load(open(f)) for f in sorted(glob.glob(f"logs/eval/headroom_{t}_*.json")) if "_gate_" not in f}
runs = {t: load(t) for t in tags}


def rel(o, b):
    """(B0 − oracle) / B0 of the episode mean (> 0 = oracle better), with an episode-bootstrap 95 % CI."""
    o, b = np.array(o, float), np.array(b, float)
    o, b = o[~np.isnan(o)], b[~np.isnan(b)]
    # Relative to B0; where B0's mean is 0 (falls) the plain difference B0 − oracle is reported instead.
    f = lambda o, b: (b.mean() - o.mean()) / (abs(b.mean()) if abs(b.mean()) > 1e-6 else 1.0)
    bs = [f(rng.choice(o, len(o)), rng.choice(b, len(b))) for _ in range(2000)]
    return f(o, b), *np.percentile(bs, [2.5, 97.5])


METRICS = ("fall", "vx_err", "slip", "impact", "td_speed", "stumble", "sink")
for t in tags:
    for c, m in runs[t].items():
        print(f"{t:5s} {c:26s} " + " ".join(f"{k}={m[k]:.4f}" for k in METRICS + ("height", "pitch", "stride")))
# Pre-registered rule: ≥ 10 % better than B0 on a target metric of the oracle's own conditions, CI excluding 0.
TARGET = {
    "omat": (("vx_err", "slip"), lambda c: c in ("mat_slope_up_mu0.25", "mat_slope_down_mu0.25")),
    "ogeo": (("impact", "stumble"), lambda c: c.startswith(("shape_bumps", "shape_stairs")) and c.endswith(("_L9", "_L14", "_L17"))),
}
SOFT = (("vx_err", "sink"), lambda c: c.startswith("soft_") and not c.endswith("rigid"))
HELD = (("impact", "stumble", "fall"), lambda c: c.startswith("shape_") and c.endswith(("_L15", "_L16", "_L18", "_L19", "_L20")))
NEG = (("vx_err", "slip"), lambda c: c.startswith("mat_flat"))
for t in tags:
    if t == "b0":
        continue
    for label, (metrics, sel) in (("target", TARGET.get(t, ((), lambda c: False))), ("soft", SOFT), ("flat-control", NEG), ("held-out", HELD)):
        wins = 0
        for c in filter(sel, runs[t]):
            for k in metrics:
                d, lo, hi = rel(runs[t][c]["episodes"][k], runs["b0"][c]["episodes"][k])
                win = d >= 0.10 and lo > 0
                wins += win
                print(f"{t:5s} vs b0 {label:12s} {c:26s} {k:8s} {100 * d:+6.1f} % [{100 * lo:+6.1f}, {100 * hi:+6.1f}]" + ("  WIN" if win else ""))
        if label == "target":
            print(f"VERDICT {t}: {'GO' if wins else 'NO-GO'} on its own conditions ({wins} winning cells) — gates: logs/eval/headroom_{t}_gates.txt")
PY
