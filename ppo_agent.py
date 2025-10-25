import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ValueOperator, ProbabilisticActor
from torch.distributions import Normal, Independent

from model import ActorNetwork, CriticNetwork

class PPOAgent:
    def __init__(self, num_agents, map_name, total_generations):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- Target LRs for Scheduler ---
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99  # Discount factor for future rewards
        self.gae_lambda = 0.95 # Lambda for GAE (Advantage calculation)
        self.clip_epsilon = 0.2 # PPO clip parameter
        self.state_dim = 3 # x_vel, y_vel, z_ang_vel
        self.num_scan_beams = 1080
        self.minibatch_size = 8192
        self.epochs = 10
        
        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents) 
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        
        # --- Reward Scalars ---
        self.SPEED_REWARD_SCALAR = 0.5
        self.PROGRESS_REWARD_SCALAR = 0.5
        self.AGENT_PENALTY = 5.0
        self.SLIDE_PENALTY_SCALAR = 0.05 # Penalty for sliding sideways
        
        # --- Networks & Wrappers ---
        actor = ActorNetwork(self.num_scan_beams, self.state_dim, 2).to(self.device)
        critic = CriticNetwork(self.num_scan_beams, self.state_dim).to(self.device)
        self.actor_module = ProbabilisticActor(
            module=TensorDictModule(
                actor,
                in_keys=["observation_scan", "observation_state"],
                out_keys=["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            distribution_class=lambda loc, scale: Independent(Normal(loc, scale), 1),
            out_keys=["action"],
            return_log_prob=True
        )
        self.critic_module = ValueOperator(
            module=critic,
            in_keys=["observation_scan", "observation_state"],
            out_keys=["state_value"]
        )
        
        # --- Optimizers ---
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr_critic)
        
        # --- Loss Modules ---
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor_module,
            critic_network=self.critic_module,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=0.02,
            loss_critic_type="l2"
        )
        
        self.advantage_module = GAE(
            gamma=self.gamma, 
            lmbda=self.gae_lambda, 
            value_network=self.critic_module,
            device=self.device
        )

        # --- Storage ---
        self.buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=1024) # 2048 steps per generation
        )
        
        # --- Diagnostics ---
        self.plot_save_path = "plots/training_diagnostics_history.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Initialize storage for historical averages
        self.diagnostic_keys = ["loss_objective", "loss_entropy", "loss_critic",
                                "entropy", "kl_approx", "clip_fraction"]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        self.generation_counter = 0 # Track generation for x-axis

        
    def _load_waypoints(self, map_name):
        """
        Loads waypoints from a CSV file for the given map.
        """
        waypoint_file = f"maps/{map_name}/{map_name}_raceline.csv"
        waypoints = np.loadtxt(waypoint_file, delimiter=';')
        waypoints_xy = waypoints[:, 1:3]
        
        # 2. Calculate Cumulative Distance (s)
        positions = waypoints[:, 1:3]
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        waypoints_s = np.insert(np.cumsum(distances), 0, 0)
        raceline_length = waypoints_s[-1]

        return waypoints_xy, waypoints_s, raceline_length

    def _obs_to_tensors(self, obs):
        scans = obs['scans']
        scan_tensors = torch.from_numpy(np.array(scans)).float().to(self.device)
        scan_tensors = scan_tensors.unsqueeze(1) 
        
        state_data = np.stack(
            (obs['linear_vels_x'], obs['linear_vels_y'], obs['ang_vels_z']), 
            axis=1
        )
        state_tensor = torch.from_numpy(state_data).float().to(self.device)
        
        return scan_tensors, state_tensor

    def get_action_and_value(self, scan_tensor, state_tensor, deterministic=False):
        """
        Gets an action from the Actor and a value from the Critic.
        """
        with torch.no_grad():
            input_td = TensorDict({
                "observation_scan": scan_tensor,
                "observation_state": state_tensor
            }, batch_size=scan_tensor.shape[0])
            
            if deterministic:
                # 'mean' or 'mode' for deterministic action
                self.actor_module.interaction_mode = "mean" 
            else:
                # 'random' (sample) for exploration
                self.actor_module.interaction_mode = "random"
            
            # Get the action distribution
            self.actor_module(input_td)
            
            # Get the state-value
            self.critic_module(input_td)
            
            action = input_td["action"]
            log_prob = input_td["action_log_prob"]
            value = input_td["state_value"]
            
        return action, log_prob, value

    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        """
        Stores a single step of experience for ALL agents.
        This is a bit complex as we must convert from "list of obs" to "batch."
        """
        
        # Prepare data for TensorDict
        scans, states = self._obs_to_tensors(obs)
        next_scans, next_states = self._obs_to_tensors(next)
        
        # `reward` and `done` need to be converted
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(-1)
        done_tensor = torch.tensor(done, dtype=torch.bool).to(self.device).unsqueeze(-1)

        # This dict contains a *batch* of experiences (one for each agent)
        step_data = TensorDict({
            "observation_scan": scans,
            "observation_state": states,
            "action": action,
            "action_log_prob": log_prob,
            "state_value": value,
            "next": TensorDict({
                "observation_scan": next_scans,
                "observation_state": next_states,
                "state_value": torch.zeros_like(value),
                "reward": reward_tensor,
                "done": done_tensor,
            }, batch_size=[self.num_agents]).to(self.device)
        }, batch_size=[self.num_agents])
        
        # Add the whole batch to the buffer
        self.buffer.add(step_data.cpu())
    
    def _project_to_raceline(self, current_pos, start_idx, lookahead):
        """
        Projects the agent's current position onto the raceline segment defined
        by the search window to get the most accurate, continuous s-distance.
        
        Returns: projected_s (float), global_wp_index (int)
        """
        wp_count = len(self.waypoints_xy)
        
        # Create a wrapped search slice for the waypoints
        search_indices = np.arange(start_idx, start_idx + lookahead) % wp_count
        search_waypoints = self.waypoints_xy[search_indices]
        
        # Find the closest waypoint (W_curr) within the lookahead window
        distances_in_window = np.linalg.norm(search_waypoints - current_pos, axis=1)
        closest_wp_in_window = np.argmin(distances_in_window)
        
        # Map the local index back to the global index (Index C)
        closest_wp_index_global = search_indices[closest_wp_in_window]
        
        # Define the segment W_prev -> W_curr for projection
        W_curr = self.waypoints_xy[closest_wp_index_global]
        W_prev_index = (closest_wp_index_global - 1 + wp_count) % wp_count
        W_prev = self.waypoints_xy[W_prev_index]
        
        # Vector V: Segment direction (W_prev -> W_curr)
        V = W_curr - W_prev
        V_len_sq = np.dot(V, V)
        
        # Vector W: Vector from W_prev to Agent's Pos
        W = current_pos - W_prev
        
        # Calculate projection length (L) of W onto V. L is a scalar.
        if V_len_sq > 1e-6:
            L = np.dot(W, V) / V_len_sq
        else:
            L = 0.0

        # Clamp L to ensure the projected point P' is within the segment [0, 1]
        L_clamped = np.clip(L, 0.0, 1.0) 
        
        # Calculate the true continuous s-value
        s_prev = self.waypoints_s[W_prev_index]
        s_curr = self.waypoints_s[closest_wp_index_global]
        
        segment_distance = s_curr - s_prev
        
        # Handle the lap wrap-around condition where s_curr is near 0 and s_prev is near max_length
        if segment_distance < 0:
            segment_distance += self.raceline_length
        
        # Projected S value: s(P') = s(W_prev) + L_clamped * segment_distance
        projected_s = s_prev + L_clamped * segment_distance
        
        return projected_s, closest_wp_index_global
        
    def calculate_reward(self, next_obs):
        rewards = []
        for i in range(self.num_agents):
            # -- Initialization and Speed Reward --
            current_speed = next_obs['linear_vels_x'][i]            
            reward = current_speed * self.SPEED_REWARD_SCALAR # Encourages forward movement while discouraging backwards movement
            
            
            # -- Raceline Reward --
            # Logic: Find closest waypoint ahead of last achieved waypoint AND within lookahead distance
            #        Then calculate progress along raceline at waypoint, subtract from last distance_s
            current_pos = np.array([next_obs['poses_x'][i], next_obs['poses_y'][i]])
            start_idx = self.last_wp_index[i]
            
            # Define a search window: from the last achieved WP to the next 50.
            # This prevents the car from constantly locking onto the same, passed point.
            lookahead = 50 # 50 is a middle-ground
            
            current_s, global_wp_index = self._project_to_raceline(current_pos, start_idx, lookahead)
            
            # Calculate progress, handling lap wrap-around
            progress = current_s - self.last_cumulative_distance[i]
            if progress < -self.raceline_length / 2 and self.raceline_length > 0: 
                progress += self.raceline_length

            # Update tracker and add shaped reward
            self.last_cumulative_distance[i] = current_s
            self.last_wp_index[i] = global_wp_index
            waypoint_weight = self.generation_counter / 20 if self.generation_counter < 20 else 1.0
            # print(f"Speed: {reward}")
            # print(f"Progress: {progress * self.PROGRESS_REWARD_SCALAR * waypoint_weight}")
            reward += progress * self.PROGRESS_REWARD_SCALAR * waypoint_weight
            
            
            
            # -- Sideways Penalty --
            sideways_speed = next_obs['linear_vels_y'][i]
            reward -= abs(sideways_speed) * self.SLIDE_PENALTY_SCALAR
            
            
            
            # -- Collision Penalty --
            if next_obs['collisions'][i]:
                reward -= self.AGENT_PENALTY
                
            rewards.append(reward)
        return rewards, np.array(rewards).mean() # Return list and avg
    
    def reset_progress_trackers(self, initial_poses_xy):
        """Resets the cumulative distance tracker for all agents after an episode reset."""
        new_last_cumulative_distance = np.zeros(self.num_agents)
        new_last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        
        # Iterate over all starting positions
        for i in range(self.num_agents):
            current_pos = initial_poses_xy[i]
            
            # 1. Find the globally closest waypoint (no lookahead needed here)
            distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
            closest_wp_index = np.argmin(distances)

            # 2. Set the initial cumulative distance and index
            new_last_cumulative_distance[i] = self.waypoints_s[closest_wp_index]
            new_last_wp_index[i] = closest_wp_index
            
        self.last_cumulative_distance = new_last_cumulative_distance
        self.last_wp_index = new_last_wp_index

    def learn(self):
        """
        The "Coaching Session" where we train the networks.
        """
        print("Starting learning phase...")
        # Get all data from the "generation"
        # .sample() with no args returns the *entire* buffer
        data = self.buffer.sample(batch_size=len(self.buffer))

        # Calculate advantages (GAE)
        # how much "better" or "worse" each action was
        with torch.no_grad():
            self.critic_module.to("cpu")
            self.advantage_module.device = "cpu"
            
            self.advantage_module(data) 
            
            self.critic_module.to(self.device)
            self.advantage_module.device = self.device
        
        data = data.flatten(0, 1)
        total_samples = data.batch_size[0]
        
        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        
        # Train for several epochs on this same data
        for _ in range(self.epochs): # PPO repeats training on the same data            
            # Shuffle data indices for this epoch
            indices = torch.randperm(total_samples)
            
            # Loop over all samples in minibatches
            for start in range(0, total_samples, self.minibatch_size):
                end = start + self.minibatch_size
                if end > total_samples:
                    continue # Skip the last, incomplete minibatch
                
                minibatch_indices = indices[start:end]
                
                # Sample the minibatch from the full dataset
                minibatch_data = data[minibatch_indices].to(self.device)
                                
                # Get the loss from torchrl's PPO module
                loss_td = self.loss_module(minibatch_data) # <--- Run on the minibatch
                
                # Sum the actor and critic losses
                actor_loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
                critic_loss = loss_td["loss_critic"]
                loss = actor_loss + critic_loss
                
                # -- Backpropagation --
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                for key in self.diagnostic_keys:
                    if key in loss_td.keys():
                            value = loss_td[key].detach().cpu().item()
                            current_gen_diagnostics[key].append(value)
        
        self.generation_counter += 1
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key, [])
            if values: # Check if list is not empty
                avg_value = np.mean(values)
                self.diagnostics_history[key].append(avg_value)
            else:
                # Append NaN or None if no data was collected for this key in this gen
                self.diagnostics_history[key].append(np.nan)


        current_lr_actor = self.actor_optimizer.param_groups[0]['lr']
        current_lr_critic = self.critic_optimizer.param_groups[0]['lr']
        self.diagnostics_history.setdefault("lr_actor", []).append(current_lr_actor)
        self.diagnostics_history.setdefault("lr_critic", []).append(current_lr_critic)
        if self.generation_counter > 0: self._plot_historical_diagnostics()
             
        # self.actor_scheduler.step()
        # self.critic_scheduler.step()
        print(f"Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.6f}, Critic LR: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
        
        # Clear the buffer for the next "generation"
        self.buffer.empty()
        print("Learning complete.")
        
    def _plot_historical_diagnostics(self):
        """
        Generates and saves a plot showing the trend of average diagnostics
        across all completed generations. Overwrites the file each time.
        """
        # Define keys to plot (exclude generation if it's not in history dict)
        keys_to_plot = [k for k in self.diagnostic_keys if k != "generation" and k in self.diagnostics_history]
        num_metrics = len(keys_to_plot)

        if num_metrics == 0 or self.generation_counter == 0:
            print("No diagnostics data to plot yet.")
            return

        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 3 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # Ensure axes is always iterable

        # X-axis: Generation number
        x_axis = np.arange(1, self.generation_counter + 1) # Generations 1, 2, 3...

        # Plot each metric's history
        for idx, key in enumerate(keys_to_plot):
            values = self.diagnostics_history.get(key, [])
            ax = axes[idx] # Get the correct subplot axis

            if not values: # Skip if no data for this key
                ax.set_ylabel(key)
                ax.grid(True)
                continue

            # Convert to numpy array, handling potential NaNs if some generations had errors
            values_np = np.array(values)

            # Plot only valid (non-NaN) points
            valid_indices = ~np.isnan(values_np)
            if np.any(valid_indices): # Check if there are any valid points to plot
                 ax.plot(x_axis[valid_indices], values_np[valid_indices], marker='.', linestyle='-', label=f'Avg {key}')
            ax.set_ylabel(key)
            ax.legend(loc='upper right')
            ax.grid(True)

        axes[-1].set_xlabel("Generation Number")
        fig.suptitle("Training Diagnostics History", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

        try:
            plt.savefig(self.plot_save_path)
            print(f"Diagnostics history plot saved to {self.plot_save_path}")
        except Exception as e:
            print(f"Error saving diagnostics history plot: {e}")
        plt.close(fig) # Close the figure to free memory