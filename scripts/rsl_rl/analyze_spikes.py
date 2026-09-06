import pandas as pd
import glob
import os

files = glob.glob('logs/rsl_rl/hexapod_flat/*/debug_robot_kinematics.csv')
latest_csv = max(files, key=os.path.getctime)
df = pd.read_csv(latest_csv)
force_cols = ['force_z_RF', 'force_z_RM', 'force_z_RB', 'force_z_LF', 'force_z_LM', 'force_z_LB']

print("=== SPIKES (>15N) ===")
for col in force_cols:
    spikes = df[df[col] > 15.0]
    print(f"{col}: {len(spikes)} spikes. Max: {df[col].max():.2f}N")

print("\n=== LEFT SIDE TRIPOD BEHAVIOR (Steps 5-19) ===")
# Phase offset: Tripod A (LM) starts swing at 0. Tripod B (LF, LB) starts swing at pi.
print(df[['timestep', 'cpg_phase_0', 'force_z_LM', 'force_z_LF', 'force_z_LB']].iloc[5:20].to_string(index=False))

print("\n=== RIGHT SIDE DRAGGING ANALYSIS ===")
print(df[['timestep', 'cpg_phase_0', 'force_z_RF', 'force_z_RM', 'force_z_RB']].iloc[5:20].to_string(index=False))
