#!/usr/bin/env bash
# Run S1 TCN window sweep and multi-seed training runs.
# Usage:
#   scripts/rsl_rl/run_s1_sweep_and_seeds.sh sweep
#   scripts/rsl_rl/run_s1_sweep_and_seeds.sh seeds <W>

set -e
export OPENBLAS_NUM_THREADS=1

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  $0 sweep"
    echo "  $0 seeds <W>"
    exit 1
fi

MODE=$1
PI_S0="logs/rsl_rl/hexapod_s0b/2026-09-24_00-35-51_s0_coordlift_prox_fromscratch_seed42/model_12200.pt"

if [ "$MODE" = "sweep" ]; then
    echo "Starting S1 W-sweep (W=25, 75) on seed 42..."
    
    # W=25
    echo "Running W=25... logging to logs/s1_sweep_w25.log"
    nohup python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 8000 \
        --warmstart "$PI_S0" \
        --run_name s1_sweep_w25_seed42 env.aux.window=25 > logs/s1_sweep_w25.log 2>&1
    
    # W=75
    echo "Running W=75... logging to logs/s1_sweep_w75.log"
    nohup python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed 42 --max_iterations 8000 \
        --warmstart "$PI_S0" \
        --run_name s1_sweep_w75_seed42 env.aux.window=75 > logs/s1_sweep_w75.log 2>&1

elif [ "$MODE" = "seeds" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Error: 'seeds' mode requires the chosen window size (W)."
        echo "Usage: $0 seeds <W>"
        exit 1
    fi
    W=$2

    # Seed 42 is pilot 3 (phase2-progress §2.2.8). 5 seeds total (user decision 2026-09-28): 43-46.
    # Override with SEEDS="43 44 ..." for more (B2/F target 8 paired seeds).
    SEEDS=${SEEDS:-"43 44 45 46"}
    echo "Starting S1 multi-seed training at W=$W, seeds: $SEEDS"

    for SEED in $SEEDS; do
        LOG=logs/s1_w${W}_seed${SEED}.log
        # Up to 3 attempts per seed: Kit can still die at start-up (exit 139); a crashed seed must not stop the rest.
        for TRY in 1 2 3; do
            echo "$(date '+%F %T') seed $SEED attempt $TRY -> $LOG"
            set +e
            nohup python scripts/rsl_rl/train.py --task Hexapod-S1-v0 --headless --num_envs 4096 --seed "$SEED" \
                --max_iterations 8000 --warmstart "$PI_S0" \
                --run_name s1_w${W}_seed${SEED} env.aux.window="$W" > "$LOG" 2>&1
            RC=$?
            set -e
            # Done when the last checkpoint exists (Kit can exit 0 after a Python error).
            LAST=$(ls -d logs/rsl_rl/hexapod_s1/*_s1_w${W}_seed${SEED} 2>/dev/null | tail -1)
            if [ -n "$LAST" ] && [ -f "$LAST/model_7999.pt" ]; then
                echo "$(date '+%F %T') seed $SEED finished (rc $RC): $LAST"
                break
            fi
            echo "$(date '+%F %T') seed $SEED attempt $TRY failed (rc $RC); see $LOG"
            [ -n "$LAST" ] && [ -f "$LAST/model_0.pt" ] && { echo "seed $SEED died mid-run; not retrying"; break; }
        done
    done
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {sweep|seeds}"
    exit 1
fi

echo "Done."
