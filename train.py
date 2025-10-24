import gym
import numpy as np
from ppo_agent import PPOAgent
import torch

MAX_PHYSICS_TIME = 10.0

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
NUM_AGENTS = 2
MAP_NAME = "vegas"
TOTAL_TIMESTEPS = 1_000_000
STEPS_PER_GENERATION = 2048 # How long we "play" before "coaching"
MAX_EPISODE_TIME = 15.0 # Max time in seconds before an episode resets
LIDAR_BEAMS = 1080  # Default is 1080
LIDAR_FOV = 4.7     # Default is 4.7 radians (approx 270 deg)
INITIAL_POSES = np.array([
    [0., 0., 0],  # Pose for Ego agent [x, y, theta]
    [1., 0., 0]   # Pose for Opponent agent [x, y, theta]
])

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

# --- Reset Environment ---
obs, step_reward, done, info = env.reset(poses=INITIAL_POSES)
current_physics_time = 0.0

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

best_avg_reward = -float('inf')
num_generations = TOTAL_TIMESTEPS // STEPS_PER_GENERATION
for gen in range(num_generations):
    
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
        next_obs, step_reward, done_from_env, info = env.step(action_np)
        
        # Calculate Reward
        rewards_list, avg_reward = agent.calculate_reward(done_from_env, next_obs)
        total_reward_this_gen += avg_reward

        # Handle Time Limit
        current_physics_time += step_reward
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
        is_agent_collision = next_obs['collisions'][0]
        is_wall_collision = done_from_env and not is_agent_collision
        if is_wall_collision or is_time_up:
            print(f"Episode finished. Reason: {'Wall' if done_from_env else 'Time'}. Resetting.")
            obs, _, _, _ = env.reset(poses=INITIAL_POSES)
            current_physics_time = 0.0
        else:
            # Only update obs if not done
            obs = next_obs
            
    # --- END OF GENERATION ---
    reward_avg = total_reward_this_gen / STEPS_PER_GENERATION
    print(f"Generation {gen+1} finished. Avg reward: {reward_avg:.3f}")
    
    agent.learn()
    
    if reward_avg > best_avg_reward:
        torch.save(agent.actor_module.module.state_dict(), f"models/actor_gen_{gen+1}.pt")
        torch.save(agent.critic_module.module.state_dict(), f"models/critic_gen_{gen+1}.pt")
        best_avg_reward = reward_avg
        print(f"New best model saved with avg reward: {best_avg_reward:.3f}")
        
# --- END OF TRAINING ---
env.close()
print("Training complete.")