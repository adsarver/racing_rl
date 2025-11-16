import numpy as np
import os
from scipy.interpolate import interp1d
import random
import gym
from torchrl.envs.libs.gym import GymEnv

def get_map_dir(map_name):
    """
    Returns the directory path for the specified map within the f1tenth_gym package.
    """
    map_dir = os.path.join('maps', map_name)
    return map_dir

def _get_wrapped_distance(s1, s2, track_length):
    """Calculates the shortest distance between two points on a circular track."""
    diff = abs(s1 - s2)
    return min(diff, track_length - diff)

def generate_start_poses(map_name, num_agents, theta_jitter=0.05, verbose=False, agent_poses=None):
    """
    Generates safe starting poses evenly distributed along the map's raceline,
    using the format from the f1tenth_racetracks repository.
    
    Args:
        map_name (str): The name of the map (e.g., "Spielberg").
        num_agents (int): The number of agents to generate poses for.

    Returns:
        numpy.ndarray: An array of shape (num_agents, 3) with [x, y, theta] poses.
                       Returns default poses if raceline loading fails.
    """
    try:
        # Get map directory
        map_dir = get_map_dir(map_name)
        # Assuming the files are placed correctly in the gym's maps folder
        waypoint_file = os.path.join(map_dir, f"{map_name}_raceline.csv")
        # Format: [s; x; y; psi; kappa; vx; ax]
        waypoints = np.loadtxt(waypoint_file, delimiter=';') # <-- Use semicolon
        
        # 2. Extract Positions and Calculate Cumulative Distance
        positions = waypoints[:, 1:3] # <-- x is index 1, y is index 2
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        total_raceline_length = cumulative_distances[-1]

        # 3. Determine Target Distances for Each Agent
        min_safe_distance = 1.0 
        required_length = (num_agents - 1) * min_safe_distance
        
        if total_raceline_length < required_length:
            print(f"Warning: Raceline is too short ({total_raceline_length:.2f}m) for {num_agents} agents with {min_safe_distance}m spacing. Placing closer.")
            spacing = total_raceline_length / max(1, num_agents) # Avoid division by zero if num_agents=0
        else:
            spacing = total_raceline_length / max(1, num_agents) # Use max for safety

        target_distances = [(i * spacing * random.uniform(0.8, 1.0)) % total_raceline_length for i in range(num_agents)]
        random.shuffle(target_distances) # Shuffle to avoid overfitting

        # 4. Interpolate Poses at Target Distances
        interp_x = interp1d(cumulative_distances, waypoints[:, 1], kind='linear') # <-- x index 1
        interp_y = interp1d(cumulative_distances, waypoints[:, 2], kind='linear') # <-- y index 2
        
        if agent_poses is not None:
            # We only care about the (x, y) coordinates for distance checks
            avoid_list_xy = list(agent_poses[:, :2])
        else:
            avoid_list_xy = []
        
        generated_poses = [] # This will store the final [x, y, theta] poses
        
        for _ in range(num_agents): # We need to generate this many new poses
            
            # Keep re-rolling until we find a safe spot
            is_safe = False
            while not is_safe:
                # 2. Generate one new random pose (x, y, theta)
                target_s = random.uniform(0, total_raceline_length)
                
                # Interpolate x, y, theta from this target_s
                x = interp_x(target_s)
                y = interp_y(target_s)
                
                interp_theta_index = np.searchsorted(cumulative_distances, target_s, side='right') - 1
                interp_theta_index = max(0, min(interp_theta_index, len(waypoints) - 1))
                theta = waypoints[interp_theta_index, 3] + random.uniform(-theta_jitter, theta_jitter)
                
                new_pose_xy = np.array([x, y])
                
                # 3. Assume it's safe and check against the *entire* avoid list
                #    (This includes active agents AND newly spawned agents)
                is_safe = True
                for existing_xy in avoid_list_xy:
                    dist = np.linalg.norm(new_pose_xy - existing_xy)
                    
                    if dist < 2.0:  # Minimum safe distance
                        # Collision found, re-roll
                        is_safe = False
                        break
            
            # 4. We found a safe spot!
            # Add the full pose to our list of generated poses
            generated_poses.append([x, y, theta])
            # Add just the (x, y) to the avoid list for future checks
            avoid_list_xy.append(new_pose_xy)

        if verbose: print(f"Generated {num_agents} safe start poses.")
        return np.array(generated_poses)

    except Exception as e:
        print(f"Warning: Could not generate poses from raceline. Using default fallback. Error: {e}")
        # Fallback just in case
        poses = np.array([
            [0., i * -1.5, 3.14] for i in range(num_agents) 
        ])
        return poses

# Fixes NAN and INF values
def check_nan(obs):
    temp_obs = obs.copy()
    for k, v in temp_obs.items():
        if isinstance(v, list):
            v = np.array(v)
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                print("NAN/INF DETECTED!!\n")
                if k == 'scans':
                    obs[k] = np.clip(v, 0.0, 30.0).tolist()
                else:
                    obs[k] = np.nan_to_num(v).tolist()
        elif isinstance(v, np.ndarray):
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                print("NAN/INF DETECTED!!\n")
                if k == 'scans':
                    obs[k] = np.clip(v, 0.0, 30.0)
                else:
                    obs[k] = np.nan_to_num(v)
    return obs

def buffer_to_pkl(buffer, filename):
    """
    Saves the demonstration buffer to a pickle file.
    
    Args:
        buffer (list): List of demonstration tuples.
        filename (str): Path to the output pickle file.
    """
    import pickle
    
    if not os.path.exists("demonstrations"):
        os.makedirs("demonstrations")

    with open("demonstrations/" + filename, mode='wb') as file:
        pickle.dump(buffer, file)
            
def load_buffer_from_pkl(filename):
    """
    Loads the demonstration buffer from a pickle file.
    
    Args:
        filename (str): Path to the input pickle file.
    """
    import pickle
    
    filename = "demonstrations/" + filename

    if not os.path.exists(filename):
        print("No demonstration buffer found at", filename)
        return []
    
    print("Loading demonstration buffer from", filename)
    with open(filename, mode='rb') as file:
        buffer = pickle.load(file)
    
    return buffer