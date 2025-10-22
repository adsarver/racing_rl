import numpy as np

class AgentPlanner:
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def get_action(self, obs):
        # TODO: This logic will be replaced by your neural network's 'forward' pass
        
        actions = []
        for i in range(self.num_agents):
            action = [0.5, 0.0]  
            actions.append(action)
        
        return np.array(actions)

    def calculate_reward(self, done_from_env, info):
        """
        Calculates the reward based on the step's results.
        """
        
        # Goal 2: Wall collision
        is_wall_collision = done_from_env
        
        # Goal 1: Agent-agent collision
        # info['collisions'][0] is True if the ego agent (0) hit another agent
        is_agent_collision = info['collisions'][0]
        
        # --- Goal 3: Your Custom Reward Function ---
        reward = 0.0
        
        if is_wall_collision:
            # Heavy penalty for hitting a wall
            reward = -100.0
        elif is_agent_collision:
            # Heavy penalty for hitting another agent
            reward = -50.0
        else:
            # Small "survival" reward for each step taken
            reward = 0.1 
            
        return reward
            
    def reset(self):
        pass