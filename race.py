import gym
import numpy as np
from ppo_agent import PPOAgent
import torch
import os
from scipy.interpolate import interp1d
from utils import *

# --- Car Physics Parameters ---
params_dict = {'mu': 1.0489,
               'C_Sf': 4.718,
               'C_Sr': 5.4562,
               'lf': 0.15875,
               'lr': 0.17145,
               'h': 0.074,
               'm': 3.74,
               'I': 0.04712,
               's_min': -0.4189,
               's_max': 0.4189,
               'sv_min': -3.2,
               'sv_max': 3.2,
               'v_switch':7.319,
               'a_max': 9.51,
               'v_min':-5.0,
               'v_max': 20.0,
               'width': 0.31,
               'length': 0.58
               }

# --- Race Parameters ---
NUM_AGENTS = 2
MAP_NAME = "YasMarina"
NUM_RACES = 5 # How many races to run
MAX_EPISODE_TIME = 25.0 # Max time in seconds before a race resets

# --- IMPORTANT: These MUST match your trained model ---
LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

# --- Model Paths ---
ACTOR_WEIGHTS_PATH = "actor_best.pth"
CRITIC_WEIGHTS_PATH = "critic_best.pth"

# --- Environment Setup ---
env = gym.make(
    "f110_gym:f110-v0",
    map=MAP_NAME,
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)

# --- Agent Setup ---
agent = PPOAgent(num_agents=NUM_AGENTS)
device = agent.device

# --- Load Trained Weights ---
try:
    print(f"Loading actor weights from: {ACTOR_WEIGHTS_PATH}")
    agent.actor_module.module.load_state_dict(torch.load(ACTOR_WEIGHTS_PATH, map_location=device))
    
    print(f"Loading critic weights from: {CRITIC_WEIGHTS_PATH}")
    agent.critic_module.module.load_state_dict(torch.load(CRITIC_WEIGHTS_PATH, map_location=device))
    
    # Set models to evaluation mode (disables dropout, etc.)
    agent.actor_module.eval()
    agent.critic_module.eval()
    
    print("Weights loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find model weights. Make sure 'actor_best.pth' and 'critic_best.pth' are in the same directory.")
    env.close()
    exit()

# --- Generate Starting Poses ---
INITIAL_POSES = generate_start_poses(MAP_NAME, NUM_AGENTS)

# --- Main Race Loop ---
for race in range(NUM_RACES):
    print(f"\n--- Starting Race {race+1} / {NUM_RACES} ---")
    
    obs, _, _, _ = env.reset(poses=INITIAL_POSES)
    current_physics_time = 0.0
    done = False

    while not done:
        # --- RENDER THE VIEW ---
        env.render(mode='human')
        
        # Get Action from Agent
        scan_tensors, state_tensor = agent._obs_to_tensors(obs)
        
        # --- Use deterministic=True for racing ---
        action_tensor, _, _ = agent.get_action_and_value(
            scan_tensors, state_tensor, deterministic=True
        )
        
        # Convert to NumPy for the Gym environment
        action_np = action_tensor.cpu().numpy()
        
        # Step the Environment
        next_obs, step_reward, done_from_env, info = env.step(action_np)
        
        # Handle Time Limit
        current_physics_time += step_reward
        is_time_up = current_physics_time >= MAX_EPISODE_TIME
        
        # Check for Episode End
        is_collision = done_from_env
        done = is_collision or is_time_up
        
        if done:
            print(f"Race {race+1} finished: Time limit reached.")
        
        # Update observation for next loop
        obs = next_obs

# --- END OF RACING ---
env.close()
print("All races complete.")
