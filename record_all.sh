#!/bin/bash
set -e
CKPT="logs/rsl_rl/hexapod_s1/2026-09-26_02-33-23_s1_pilot2_seed42/model_8000.pt"

for level in 16 17 18; do
    for terrain in rough bumps slope_up slope_down stairs_up; do
        tag="L${level}_${terrain}"
        echo "Recording $tag..."
        /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/record_gait.py --ckpt $CKPT --terrain_type $terrain --terrain_level $level --seconds 15.0 --tag $tag
    done
done

# Yaw videos
for i in 1 2; do
    tag="L16_rough_yaw_0.3_run${i}"
    echo "Recording $tag..."
    /home/ubuntu/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/record_gait.py --ckpt $CKPT --terrain_type rough --terrain_level 16 --yaw_cmd 0.3 --seconds 15.0 --tag $tag
done
echo "DONE" > logs/.done_record_all
