import numpy as np
import os
from scipy.interpolate import interp1d

def get_map_dir(map_name):
    """
    Returns the directory path for the specified map within the f1tenth_gym package.
    """
    map_dir = os.path.join('maps', map_name)
    return map_dir

def generate_start_poses(map_name, num_agents):
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

        target_distances = [(i * spacing) % total_raceline_length for i in range(num_agents)]
        target_distances.sort()

        # 4. Interpolate Poses at Target Distances
        interp_x = interp1d(cumulative_distances, waypoints[:, 1], kind='linear') # <-- x index 1
        interp_y = interp1d(cumulative_distances, waypoints[:, 2], kind='linear') # <-- y index 2
        
        # Find closest waypoint index *before* target distance for theta
        interp_theta_indices = np.searchsorted(cumulative_distances, target_distances, side='right') - 1
        
        start_poses = []
        for i, target_s in enumerate(target_distances):
            x = interp_x(target_s)
            y = interp_y(target_s)
            closest_wp_index = interp_theta_indices[i]
             # Ensure index is valid
            closest_wp_index = max(0, min(closest_wp_index, len(waypoints) - 1))
            theta = waypoints[closest_wp_index, 3] # <-- theta (psi_rad) index 3
            start_poses.append([x, y, theta])

        print(f"Generated {num_agents} start poses spaced along the raceline.")
        return np.array(start_poses)

    except Exception as e:
        print(f"Warning: Could not generate poses from raceline. Using default fallback. Error: {e}")
        # Fallback just in case
        poses = np.array([
            [0., i * -1.5, 3.14] for i in range(num_agents) 
        ])
        return poses

