from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import hexasim.tasks.manager_based.hexasim

env_cfg = parse_env_cfg("Hexapod-Flat-v0")
env = gym.make("Hexapod-Flat-v0", cfg=env_cfg)
env.reset()

sensor = env.unwrapped.scene["contact_forces"]
print("Sensor body names:", sensor.body_names)

env.close()
app.close()
