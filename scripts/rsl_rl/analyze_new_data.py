import pandas as pd
import numpy as np

csv_path = "logs/rsl_rl/hexapod_flat/2026-09-06_14-35-10/debug_cpg_trajectory.csv"
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Total steps: {len(df)}")

print("\n=== FORCE STATS (Mean) ===")
force_cols = [f'leg_{i}_force_z' for i in range(6)]
print(df[force_cols].mean())

print("\n=== ERROR STATS (Mean) ===")
err_cols = [f'leg_{i}_error' for i in range(6)]
print(df[err_cols].mean())

print("\n=== PHASE STATS (Freq) ===")
# Phase diff to freq
df['phase_diff'] = df['leg_0_phase'].diff().apply(lambda x: x + 2*np.pi if x < 0 else x)
df['freq'] = df['phase_diff'] / (2 * np.pi * 0.02)
print(f"Mean CPG Frequency: {df['freq'].mean():.4f} Hz")
print(f"Min CPG Frequency: {df['freq'].min():.4f} Hz")
print(f"Max CPG Frequency: {df['freq'].max():.4f} Hz")

print("\n=== TRIPOD A (Leg 0, 4, 2 -> RF, LM, RB) ===")
print(df[['timestep', 'leg_0_phase', 'leg_0_force_z', 'leg_4_force_z', 'leg_2_force_z', 'leg_0_target_z', 'leg_0_tibia_z', 'leg_0_baseline']].iloc[50:70].to_string(index=False))

print("\n=== TRIPOD B (Leg 3, 1, 5 -> LF, RM, LB) ===")
print(df[['timestep', 'leg_3_phase', 'leg_3_force_z', 'leg_1_force_z', 'leg_5_force_z', 'leg_3_target_z', 'leg_3_tibia_z', 'leg_3_baseline']].iloc[50:70].to_string(index=False))

