from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import the package to register environments
import hexasim.tasks.manager_based.hexasim

env_cfg = parse_env_cfg("Hexapod-Flat-v0")
env = gym.make("Hexapod-Flat-v0", cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
print("Robot body names:", robot.data.body_names)
print("Robot body IDs for .*tibia.*:", robot.find_bodies(".*tibia.*"))

sensor = env.unwrapped.scene["contact_forces"]
print("Sensor body names:", sensor.data.body_names)

env.close()
app.close()
