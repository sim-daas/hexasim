#!/usr/bin/env bash
# One training milestone review (phase2-progress §2.1.38): waits for <run_dir>/model_<iter>.pt, then records side and
# front gait videos, the swing radial probe, the leg-gap probe and the mean-reward slope into
# logs/milestones/<run>/it<iter>*. Runs next to training.
# Usage: scripts/rsl_rl/milestone_review.sh <run_dir> <iter>
set -u
RUN=$1; IT=$2; NAME=$(basename "$RUN"); OUT=logs/milestones/$NAME; mkdir -p "$OUT"
until [ -f "$RUN/model_$IT.pt" ]; do sleep 30; done
sleep 10  # let the checkpoint finish writing
CK=$RUN/model_$IT.pt
for v in side front; do
  python scripts/rsl_rl/record_gait.py --headless --ckpt "$CK" --tag "it${IT}_$v" --view $v --video_dir "$OUT" 2>&1 \
    | grep -E '^(right|left)_|stance half-width' >> "$OUT/it${IT}_geometry.txt"
done
python scripts/tests/swing_radial_probe.py --headless "it$IT=$CK" 2>&1 | grep -E '^RADIAL|^  |Traceback' > "$OUT/it${IT}_radial.txt"
python scripts/tests/leg_gap_probe.py --headless "it$IT=$CK" 2>&1 | grep -E '^GAP|Traceback' > "$OUT/it${IT}_gap.txt"
python - "$RUN" > "$OUT/it${IT}_slope.txt" <<'PY'
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# All event files of the run dir, oldest first: a resumed run's dir holds a copy of its predecessor's (later wins).
files = []
for f in sorted(glob.glob(sys.argv[1] + "/events*")):
    ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload(); files.append(ea)
for k in ("Train/mean_reward", "Episode_Reward/track_lin_vel", "Episode_Reward/leg_proximity", "Curriculum/terrain_levels"):
    by_step = {}
    for ea in files:
        by_step.update({s.step: s.value for s in ea.Scalars(k)})
    v = [by_step[t] for t in sorted(by_step)]
    now, before = sum(v[-500:]) / 500, sum(v[-2500:-2000]) / 500
    print(f"SLOPE {k:32s} last500 {now:9.3f}  2000 its earlier {before:9.3f}  change {100 * (now - before) / abs(before):+6.1f} %")
PY
