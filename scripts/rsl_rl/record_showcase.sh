#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Showcase renders of a trained policy (phase2-progress §2.4). Run in tmux 0:3 with no training on the GPU:
#
#   bash scripts/rsl_rl/record_showcase.sh                 # default: O-mat final (S2 headroom, seed 43)
#   bash scripts/rsl_rl/record_showcase.sh <ckpt> <quality> <res> <sky> <terrain_mdl> <seconds> <orbit_turns> \
#                                          <terrain_level> <cmd>
#   EQUIV=1 bash scripts/rsl_rl/record_showcase.sh     # also run the equivalence check (4 extra short cells)
#
# quality: high = RT2 path tracing (needs the GPU to itself), mid = RT1 (runs next to a training job).
# Skies: see demo_look.SKIES (noon_grass, partly_cloudy, evening, studio, …).
# terrain_mdl: one ground material everywhere (dirt, sand, gravel, grass, concrete, …) or "multi" for the
#   per-material patchwork that shows the physics. See demo_look.GROUND_MDL.
# First run only: the HDRI and the MDLs stream from the Isaac asset root, so start-up is slow once.
# Cell A is the equivalence check: the same take with and without --pretty must print the SAME gait table, which is
# what proves the showcase rendering is visual only. Everything after it is footage.
#
# Writes mp4s to logs/videos/showcase, gait tables and stills to logs/showcase, and logs/.done_showcase at the end.
set -u

CKPT=${1:-logs/rsl_rl/hexapod_s2/2026-10-03_02-09-32_s2_headroom_O-mat_seed43/model_13998.pt}  # O-mat, seed 43
QUALITY=${2:-mid}
RES=${3:-1920x1080}
SKY=${4:-noon_grass}
GROUND=${5:-granite}
SECONDS_=${6:-30}        # not SECONDS: bash makes that a magic counter
TURNS=${7:-0.75}
LEVEL=${8:-12}
CMD=${9:-0.12}
EQUIV=${EQUIV:-0}   # EQUIV=1 also runs the plain repeats and the equivalence check

VID=logs/videos/showcase
OUT=logs/showcase
mkdir -p "$VID" "$OUT"
rm -f logs/.done_showcase
export OPENBLAS_NUM_THREADS=1   # Kit start-up forks and races numpy's OpenBLAS at-fork handler (exit 139)

run () {  # run <tag> <extra record_gait args…>
  local tag=$1; shift
  echo; echo "=================== $tag ==================="
  python scripts/rsl_rl/record_gait.py --headless --ckpt "$CKPT" --tag "$tag" --video_dir "$VID" "$@" \
    2>&1 | tee "$OUT/$tag.log"
}

# ---------------------------------------------------------------------------------------------------------------
# A. Equivalence check, OFF by default — EQUIV=1 to run it. PhysX on the GPU is not bit-reproducible, so the
# plain command is repeated three times to get its own envelope and the --pretty run has to land inside it;
# equiv_check.py does that comparison. Already run and passed on 2026-10-03 (phase2-progress §2.4.6).
# ---------------------------------------------------------------------------------------------------------------
if [ "$EQUIV" = "1" ]; then
  EQ=(--seconds 4 --terrain_type rough --terrain_level "$LEVEL" --seed 42)
  run equiv_plain  "${EQ[@]}"
  run equiv_plain2 "${EQ[@]}"
  run equiv_plain3 "${EQ[@]}"
  run equiv_pretty "${EQ[@]}" --pretty --quality "$QUALITY" --sky "$SKY" --terrain_mdl "$GROUND"
  python scripts/rsl_rl/equiv_check.py "$OUT"/equiv_pretty.log \
    "$OUT"/equiv_plain.log "$OUT"/equiv_plain2.log "$OUT"/equiv_plain3.log | tee "$OUT/equiv.txt"
fi

# ---------------------------------------------------------------------------------------------------------------
# No --robot_mdl: the robot keeps its own material, the look of the showcase_rough_L14 take. Override with
#   --robot_mdl gunmetal (or grey / light_grey / abs) if it reads too dark against the ground.
# --orbit_turns sets revolutions per take, so the orbit rate follows --seconds: longer take = slower orbit.
# --uniform_friction CHANGES THE PHYSICS: the material field is cut to 'concrete' (mu 0.8-1.0, rigid) so the
# robot never crosses a low-friction patch. The visual material never affected friction — these are separate
# systems (phase2-progress §2.4.9). Showcase footage only; drop the flag for a take that represents training.
P=(--pretty --quality "$QUALITY" --sky "$SKY" --res "$RES" --terrain_mdl "$GROUND" --uniform_friction
   --seconds "$SECONDS_" --orbit_turns "$TURNS")

run hero_flat              "${P[@]}" --view hero  --cmd "$CMD"
run "orbit_rough_L$LEVEL"  "${P[@]}" --view orbit --cmd "$CMD" --terrain_type rough --terrain_level "$LEVEL"
# Stairs and slopes descend far below z = 0, so the backdrop slab has to drop with them or it cuts through.
run "orbit_stairs_L$LEVEL" "${P[@]}" --view orbit --cmd "$CMD" --terrain_type stairs_up \
                           --terrain_level "$LEVEL" --backdrop_z -0.7
run "orbit_slope_L$LEVEL"  "${P[@]}" --view orbit --cmd "$CMD" --terrain_type slope_up \
                           --terrain_level "$LEVEL" --backdrop_z -0.7
# Turning on the spot, camera held still: the clean look at the turning gait and where the feet land.
run "spin_rough_L$LEVEL"   "${P[@]}" --view side  --cmd 0.0 --yaw_cmd 0.3 --terrain_type rough \
                           --terrain_level "$LEVEL"

# ---------------------------------------------------------------------------------------------------------------
echo; echo "=================== outputs ==================="
for f in "$VID"/*.mp4; do
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames \
    -of csv=p=0 "$f" | sed "s|^|$(basename "$f")  |"
done
mkdir -p "$OUT/stills"
for f in "$VID"/*.mp4; do
  ffmpeg -nostdin -loglevel error -y -i "$f" -vf fps=1/3 "$OUT/stills/$(basename "${f%.mp4}")_%02d.png"
done
echo "stills → $OUT/stills  (look at them: sky the right way up and separated from the ground, terrain texture"
echo "reading at the robot's scale, highlights on the shells, grounded shadows, no magenta missing-MDL fallback)"

touch logs/.done_showcase
echo; echo "DONE → $VID, $OUT"
