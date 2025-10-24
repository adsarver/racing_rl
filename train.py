import gym
import numpy as np
from ppo_agent import PPOAgent
from utils import *
import torch
import random

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

# --- Main Training Parameters ---
NUM_AGENTS = 25
MAP_NAMES = ["YasMarina", "Catalunya", "Monza", "Silverstone", "Mexico City"]
TOTAL_TIMESTEPS = 1_000_000
STEPS_PER_GENERATION = 1024 # How long we "play" before "coaching"
MAX_EPISODE_TIME = 9.0 # Max time in seconds before an episode resets
LIDAR_BEAMS = 1080  # Default is 1080
LIDAR_FOV = 4.7   # Default is 4.7 radians (approx 270 deg)
INITIAL_POSES = generate_start_poses(MAP_NAMES[0], NUM_AGENTS)
CURRENT_MAP = MAP_NAMES[0]

env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)

# --- Agent Setup ---
agent = PPOAgent(num_agents=NUM_AGENTS, map_name=CURRENT_MAP)

# --- Reset Environment ---
obs, _, _, _ = env.reset(poses=INITIAL_POSES)
agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2]) # Pass X, Y only
current_physics_time = 0.0

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

best_avg_reward = -float('inf')
num_generations = TOTAL_TIMESTEPS // STEPS_PER_GENERATION
gen_per_map = 50
for gen in range(num_generations):
    wall_collisions = 0
    print(f"\n--- Generation {gen+1} / {num_generations} ---")
    total_reward_this_gen = 0.0
    
    for step in range(STEPS_PER_GENERATION):
        # env.render(mode='human_fast')
        
        # Get Action from Agent
        scan_tensors, state_tensor = agent._obs_to_tensors(obs)
        action_tensor, log_prob_tensor, value_tensor = agent.get_action_and_value(
            scan_tensors, state_tensor
        )
                
        # Convert to NumPy for the Gym environment
        action_np = action_tensor.cpu().numpy()
        
        # Step the Environment
        next_obs, timestep, done_from_env, info = env.step(action_np)
        
        # Calculate Reward
        rewards_list, avg_reward = agent.calculate_reward(done_from_env, next_obs)
        total_reward_this_gen += avg_reward

        # Handle Time Limit
        current_physics_time += timestep
        is_time_up = current_physics_time >= MAX_EPISODE_TIME
        
        # Store Experience
        agent.store_transition(
            obs=obs,
            next=next_obs,
            action=action_tensor,
            log_prob=log_prob_tensor,
            reward=rewards_list,
            done=np.array([done_from_env or is_time_up] * NUM_AGENTS), # Broadcast done to all agents
            value=value_tensor
        )
        
        
        # Check for Episode End (Reset)
        is_collision = next_obs['collisions'][0]  # Ego agent collision
        if is_collision or is_time_up:
            wall_collisions += 1
            obs, _, _, _ = env.reset(poses=INITIAL_POSES)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2]) # Pass X, Y only
            current_physics_time = 0.0
        else:
            # Only update obs if not done
            obs = next_obs
            
    # --- END OF GENERATION ---
    reward_avg = total_reward_this_gen / STEPS_PER_GENERATION
    print(f"Generation {gen+1} finished. Avg reward: {reward_avg:.3f}. Ego collisions: {wall_collisions}")
    
    agent.learn()
    
    if reward_avg > best_avg_reward:
        torch.save(agent.actor_module.module.state_dict(), f"models/actor_gen_{gen+1}.pt")
        torch.save(agent.critic_module.module.state_dict(), f"models/critic_gen_{gen+1}.pt")
        best_avg_reward = reward_avg
        print(f"New best model saved with avg reward: {best_avg_reward:.3f}")
        
    if (gen+1) % gen_per_map == 0:
        CURRENT_MAP = random.choice(MAP_NAMES)
        print(f"Changing map to {CURRENT_MAP} for next generation.")
        # Reset environment with new map and poses
        INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
        env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
        # Reset agent raceline data
        agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = agent._load_waypoints(new_map_name)
        agent.last_cumulative_distance = np.zeros(agent.num_agents) 
        agent.last_wp_index = np.zeros(agent.num_agents, dtype=np.int32)
        
        # Reset agent trackers
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        env.reset(poses=INITIAL_POSES)
        
# --- END OF TRAINING ---
env.close()
print("Training complete.")