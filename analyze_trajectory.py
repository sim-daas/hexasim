import pandas as pd
import os

csv_path = "logs/rsl_rl/hexapod_flat/2026-08-30_21-17-33/debug_robot_kinematics.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

print("=== TRANSIENT SPIKE ANALYSIS ===")
force_cols = ["force_z_RF", "force_z_RM", "force_z_RB", "force_z_LF", "force_z_LM", "force_z_LB"]
max_forces = df[force_cols].max()
print("Max forces per leg (N):")
print(max_forces)
print("\nForces > 15N detected:")
for col in force_cols:
    spikes = df[df[col] > 15.0]
    if len(spikes) > 0:
        print(f"{col}: {len(spikes)} steps > 15N (Max: {spikes[col].max()}N)")

print("\n=== GAIT SYMMETRY ANALYSIS (Tripod A vs B) ===")
# Tripod A: RF, LM, RB
# Tripod B: LF, RM, LB
print("Tripod A (RF, LM, RB) Average Forces:", df[["force_z_RF", "force_z_LM", "force_z_RB"]].mean().values)
print("Tripod B (LF, RM, LB) Average Forces:", df[["force_z_LF", "force_z_RM", "force_z_LB"]].mean().values)

# Let's count how many steps each leg is "airborne" (< 0.5 N)
airborne_counts = (df[force_cols] < 0.5).sum()
print("\nSteps airborne (< 0.5 N) out of 100 total steps:")
print(airborne_counts)

print("\n=== BASE HEIGHT ANALYSIS ===")
print(f"Min base Z-height: {df['pos_z'].min():.4f} m")
print(f"Max base Z-height: {df['pos_z'].max():.4f} m")
print(f"Mean base Z-height: {df['pos_z'].mean():.4f} m")

