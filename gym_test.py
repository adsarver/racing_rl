import gym
import numpy as np
from planner import AgentPlanner

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

map_name = "vegas"
num_agents = 2
lidar_beams = 1080  # Default is 1080
lidar_fov = 4.7     # Default is 4.7 radians (approx 270 deg)
initial_poses = np.array([
    [0., 0., 3.14],  # Pose for Ego agent [x, y, theta]
    [1., 0., 3.14]   # Pose for Opponent agent [x, y, theta]
])


# 2. Create the environment using keyword arguments
#    (as shown in the 'Customized Usage' documentation)
env = gym.make(
    "f110_gym:f110-v0",
    map=map_name,
    num_agents=num_agents,
    num_beams=lidar_beams,
    fov=lidar_fov,
    params=params_dict
)
planner = AgentPlanner(num_agents=num_agents)

# 3. Reset the environment and set the initial poses
obs, physics_step, done, _ = env.reset(poses=initial_poses)
planner.reset()
env.render()

current_physics_time = 0.0
total_epoch_reward = 0.0

try:
    while not master_done:
        action = planner.get_action(obs)
        next_obs, step_reward, done_from_env, info = env.step(action)
        current_physics_time += step_reward
        
        is_time_up = current_physics_time >= MAX_PHYSICS_TIME

        current_step_reward = planner.calculate_reward(
            done_from_env=done_from_env,
            info=info
        )
        total_epoch_reward += current_step_reward

        # Check for lap completion
        if done_from_env:
            print("Two laps completed! Ending epoch.")
            master_done = True
        
        # Check for agent-agent collision
        if info['collisions'][0]:
            print("Agent-agent collision detected! Ending epoch.")
            master_done = True
        
        # Check for time limit
        if is_time_up:
            print(f"Time limit reached ({current_physics_time:.2f}s). Ending epoch.")
            master_done = True
        
        env.render(mode='human')
        obs = next_obs

except KeyboardInterrupt:
    print("\nSimulation interrupted.")
finally:
    print(f"Epoch finished. Total reward: {total_epoch_reward:.2f}")
    env.close()
    print("Environment closed.")