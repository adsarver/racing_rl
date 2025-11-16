import time
import gym
import numpy as np
from ppo_agent import PPOAgent
from control_handler import ControlHandler
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
               'v_min': -5.0,
               'v_max': 20.0,
               'width': 0.31,
               'length': 0.58
               }

# --- Main Training Parameters ---
NUM_AGENTS = 15
EASY_MAPS = ["BrandsHatch", "Monza", "Hockenheim", "Melbourne"]
MEDIUM_MAPS = ["Oschersleben", "Sakhir", "Sepang", "SaoPaulo", "Budapest", "Catalunya", "Silverstone"]
HARD_MAPS = ["Zandvoort", "MoscowRaceway", "Austin", "Nuerburgring", "Spa", "YasMarina", "Sochi",]
TOTAL_TIMESTEPS = 4_000_000
STEPS_PER_GENERATION = 2048 # How long we "play" before "coaching"
MAX_EPISODE_TIME = 60.0 # Max time in seconds before an episode resets
LIDAR_BEAMS = 1080  # Default is 1080
LIDAR_FOV = 4.7   # Default is 4.7 radians (approx 270 deg)
INITIAL_POSES = None # Generated later
CURRENT_MAP = EASY_MAPS[0]
PATIENCE = 200  # Early stopping patience
GEN_PER_MAP = 50  # Increased from 30 - reduce distribution shift from map changes

def get_curriculum_map_pool(generation):
    """Returns the appropriate map pool based on training progress."""
    if generation < len(EASY_MAPS) * GEN_PER_MAP:
        return EASY_MAPS  # Phase 1: Learn basics
    elif generation < len(EASY_MAPS) * GEN_PER_MAP + len(MEDIUM_MAPS) * GEN_PER_MAP:
        return EASY_MAPS + MEDIUM_MAPS  # Phase 2: Add complexity
    else:
        return EASY_MAPS + MEDIUM_MAPS + HARD_MAPS  # Phase 3: Full curriculum

# -- Environment Setup ---
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)

# --- Agent Setup ---
num_generations = TOTAL_TIMESTEPS // STEPS_PER_GENERATION
agent = PPOAgent(
    num_agents=NUM_AGENTS, 
    map_name=CURRENT_MAP,
    steps=STEPS_PER_GENERATION,
    # transfer=["actor_gen_5.pt", "critic_gen_5.pt"]
)

# --- Reset Environment ---
current_physics_time = 0.0

INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)    
obs, timestep, _, _ = env.reset(poses=INITIAL_POSES)
agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

# Render first to create the window/renderer
env.render(mode="human")

# --- Control Handler Setup (must be after render) ---
# Set use_gamepad=True to enable controller support (requires 'inputs' package)
control_handler = ControlHandler(env, CURRENT_MAP, params_dict, max_demo_buffer=10000, use_gamepad=True)

best_avg_reward = -float('inf')
patience = 0
done_mat = np.ones((NUM_AGENTS,), dtype=bool)

for gen in range(num_generations):
    collisions = 0
    print(f"\n--- Generation {gen+1} / {num_generations} ---")
    total_reward_this_gen = []
    ego_reward_this_gen = []
    current_gen_time = 0.0
    
    for step in range(STEPS_PER_GENERATION):
        env.render(mode="human")
        
        # Get Action from Agent
        scan_tensors, state_tensor = agent._obs_to_tensors(obs)
        action_tensor, log_prob_tensor, value_tensor, scaled_values = agent.get_action_and_value(
            scan_tensors, state_tensor, params_dict
        )
        
        # --- Human Control Override ---
        scaled_values, action_tensor = control_handler.override_action(
            scaled_values, action_tensor, params_dict
        )
                
        # Convert to NumPy for the Gym environment
        action_np = scaled_values.cpu().numpy()
        
        # Stop episode for agents that are collided
        action_np = np.column_stack((done_mat, done_mat)) * action_np
        
        # Step the Environment
        next_obs, timestep, _, _ = env.step(action_np)
        next_obs = check_nan(next_obs)
                
        # Make copy for comparison
        done_mat_before_update = done_mat.copy()
        
        # Update Done Matrix
        done_mat = (1 - next_obs['collisions']) * done_mat
        
        # Just crashed
        just_crashed = (done_mat_before_update == 1) & (done_mat == 0)
        
        # Calculate Reward
        rewards_list, avg_reward = agent.calculate_reward(next_obs, step, just_crashed, params_dict)
        
        # --- Store Human Demonstrations ---
        control_handler.store_demonstration(
            scan_tensors, state_tensor, action_tensor, just_crashed
        )
        
        total_reward_this_gen.append(avg_reward)
        ego_reward_this_gen.append(rewards_list[0])
        
        # Calculate time
        current_gen_time += timestep
        
        # Store Experience
        agent.store_transition(
            obs=obs,
            next=next_obs,
            action=action_tensor,
            log_prob=log_prob_tensor,
            reward=rewards_list,
            done=just_crashed,
            value=value_tensor,
        )

        print(f"{step+1}/{STEPS_PER_GENERATION}: Max vel: {np.max(next_obs['linear_vels_x']):.1f} m/s, Max actor_vel: {torch.max(scaled_values[:,1]).item():.1f} m/s, Ego Speed: {next_obs['linear_vels_x'][0]:.2f} Avg Reward: {sum(total_reward_this_gen) / len(total_reward_this_gen):.2f}", end='\r')

        # Check for Episode End (Reset)
        if just_crashed.any():
            collisions += just_crashed.sum()
            just_crashed_idxs = np.where(just_crashed)[0]
            done_mat[just_crashed_idxs] = 1
            
            poses = np.array([[x, y, theta] for x, y, theta in zip(next_obs['poses_x'], next_obs['poses_y'], next_obs['poses_theta'])])
            
            INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS, agent_poses=poses)
            next_obs, _, _, _ = env.reset(poses=INITIAL_POSES, agent_idxs=just_crashed_idxs)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2], agent_idxs=just_crashed_idxs)
            
            
        obs = next_obs
    
    print() # Finish the carriage return line
    current_physics_time = 0.0
    
    # --- END OF GENERATION ---
    reward_avg = sum(total_reward_this_gen) / len(total_reward_this_gen)
    current_avg_ego_reward = sum(ego_reward_this_gen) / len(total_reward_this_gen)
    print(f"Generation {gen+1} finished.\n Avg Reward (All): {reward_avg:.3f}, Avg Reward (Ego): {current_avg_ego_reward:.3f}. Collision Exits: {collisions}")    
    
    # --- Pretrain from Demonstrations if Available ---
    if control_handler.should_pretrain():
        print(f"\n🎯 Using {len(control_handler.demonstration_buffer)} demonstrations for supervised learning...")
        agent.pretrain_from_demonstrations(control_handler.demonstration_buffer, epochs=5, batch_size=64)
        control_handler.save_demonstrations()
        control_handler.reset_pretrain_mode()
        print("Returning to normal RL training.\n")
    else:
        agent.learn(reward_avg)
    
    if reward_avg > best_avg_reward:
        torch.save(agent.actor_module.module.state_dict(), f"models/actor/actor_gen_{gen+1}.pt")
        torch.save(agent.critic_module.module.state_dict(), f"models/critic/critic_gen_{gen+1}.pt")
        best_avg_reward = reward_avg
        print(f"New best model saved with avg reward: {best_avg_reward:.3f}")
        patience = 0
    else:
        patience += 1
        print(f"No improvement in avg reward. Patience: {patience}")


    # if patience >= PATIENCE:
    #     print("Early stopping triggered due to no improvement.")
    #     break
        
    if (gen+1) % GEN_PER_MAP == 0:
        available_maps = get_curriculum_map_pool(gen+1)
        CURRENT_MAP = random.choice(available_maps)
        print(f"Gen {gen+1}: Map={CURRENT_MAP}, Pool size={len(available_maps)}")
        
        INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
        env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
        
        agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = agent._load_waypoints(CURRENT_MAP)
        agent.last_cumulative_distance = np.zeros(agent.num_agents) 
        agent.last_wp_index = np.zeros(agent.num_agents, dtype=np.int32)
        
        env.reset(poses=INITIAL_POSES)
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        
        # Update control handler with new map
        control_handler.update_map(CURRENT_MAP)
        
# --- END OF TRAINING ---
control_handler.cleanup()
env.close()
print("Training complete.")