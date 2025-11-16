import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ValueOperator, ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from model import ActorNetwork, CriticNetwork, VisionEncoder

class PPOAgent:
    def __init__(self, num_agents, map_name, steps, transfer=None):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 5e-5  # Back to original conservative rate
        self.lr_critic = 5e-5  # Back to original conservative rate
        self.gamma = 0.99  # Discount factor for future rewards
        self.gae_lambda = 0.95 # Lambda for GAE (Advantage calculation)
        self.clip_epsilon = 0.15 # Reduced from 0.2 - prevent excessive clipping
        self.state_dim = 3 # x_vel, y_vel, z_ang_vel
        self.num_scan_beams = 1080
        self.minibatch_size = 2048
        self.epochs = 5
        self.epochs_with_demos = 3  # Reduce epochs when BC is active to prevent forgetting
        
        # --- Demonstration Retention ---
        self.demo_buffer = None  # Store demonstrations for continual learning
        self.demo_bc_weight = 1.0  # Initial weight for behavior cloning loss (prevents forgetting)
        self.demo_bc_weight_initial = 1.0  # Starting BC weight after pretraining (equal to PPO)
        self.demo_bc_weight_final = 0.3  # Final BC weight after decay
        self.demo_bc_decay_gens = 20  # Generations to decay BC weight
        self.demo_pretrain_generation = None  # Track when pretraining occurred
        
        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents) 
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        
        # --- Reward Scalars ---
        self.PROGRESS_REWARD_SCALAR = 2.0  # Main reward for following raceline
        self.LAP_REWARD = 10.0  # Bonus for lap completion
        self.SPEED_REWARD_SCALAR = 0.05  # Positive reward for forward speed
        self.TURN_REWARD_SCALAR = 0.02  # Small positive reward for turning (prevents signal dropout in corners)
        self.COLLISION_PENALTY = -3.0  # Penalty for crashing
        self.SLIDE_PENALTY_SCALAR = -0.3  # Moderate penalty for excessive sliding
        self.CORNER_SPEED_BONUS = 0.03  # Reward for appropriate speed in sharp turns
        
        # --- Networks & Wrappers ---
        if transfer is None: transfer = [None, None]
        shared_encoder = self._transfer_vision(transfer[0])
        actor_encoder = shared_encoder
        critic_encoder = shared_encoder

        actor = ActorNetwork(self.state_dim, 2, encoder=actor_encoder).to(self.device)
        critic = CriticNetwork(self.state_dim, encoder=critic_encoder).to(self.device)
        
        self.actor_module = ProbabilisticActor(
            module=TensorDictModule(
                actor,
                in_keys=["observation_scan", "observation_state"],
                out_keys=["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            out_keys=["action"],
            return_log_prob=True
        )
        self.critic_module = ValueOperator(
            module=critic,
            in_keys=["observation_scan", "observation_state"],
            out_keys=["state_value"]
        )
        
        # --- Optimizers ---
        self.actor_optimizer = optim.AdamW(self.actor_module.parameters(), lr=self.lr_actor, weight_decay=0.01)
        self.critic_optimizer = optim.AdamW(self.critic_module.parameters(), lr=self.lr_critic, weight_decay=0.01)
        
        # --- Loss Modules ---
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor_module,
            critic_network=self.critic_module,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=0.35,  # Increased from 0.25 - combat entropy collapse
            normalize_advantage=True,
            critic_coeff=1.0,  # Increased from 0.5 - value network needs more weight
            clip_value=True,
            separate_losses=True,
            reduction="mean"
        )
        
        self.loss_module.set_keys(
            sample_log_prob="action_log_prob",
            value="state_value",
        )
                
        self.advantage_module = GAE(
            gamma=self.gamma, 
            lmbda=self.gae_lambda, 
            value_network=self.critic_module,
            device=self.device
        )

        # --- Storage ---
        self.buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=steps) # 2048 steps per generation
        )
        
        # --- Diagnostics ---
        self.plot_save_path = "plots/training_diagnostics_history.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Initialize storage for historical averages
        self.diagnostic_keys = ["loss_objective", "loss_entropy", "loss_critic",
                                "entropy", "kl_approx", "clip_fraction", "reward_avg"]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        self.generation_counter = 0 # Track generation for x-axis
        
    def _transfer_weights(self, path, network):

        if path is None:
            return network.to(self.device)

        checkpoint = torch.load(path)
        
        prefix = "0.module."

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                state_dict[new_key] = v
            else: state_dict[k] = v

        if state_dict:
            network.load_state_dict(state_dict)
            print("Successfully loaded pre-trained weights!")
            
        return network.to(self.device)

    def _transfer_vision(self, path):
        new_encoder = VisionEncoder(num_scan_beams=self.num_scan_beams)
        if path is None:
            return new_encoder.to(self.device)
        
        checkpoint = torch.load(path)
        prefix = "conv_layers."

        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                encoder_state_dict[new_key] = v
            elif k.startswith("0.module." + prefix):
                new_key = k[len("0.module." + prefix):]
                encoder_state_dict[new_key] = v

        if encoder_state_dict:
            new_encoder.load_state_dict(encoder_state_dict)
            print("Successfully loaded pre-trained encoder weights!")
        else:
            print(checkpoint.keys())
            print(f"Warning: No weights found with prefix '{prefix}'. Starting with a random encoder.")

        return new_encoder.to(self.device)

    def _map_range(self, value, in_min, in_max, out_min=-1, out_max=1):
        if in_max == in_min:
            return out_min if value <= in_min else out_max

        return out_min + (float(value - in_min) / float(in_max - in_min)) * (out_max - out_min)

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
        scan_tensors = torch.from_numpy(np.array(scans, dtype=np.float64)).float().to(self.device)
        scan_tensors = scan_tensors.unsqueeze(1) 
        
        state_data = np.stack(
            (obs['linear_vels_x'], obs['linear_vels_y'], obs['ang_vels_z']), 
            axis=1
        )
        state_tensor = torch.from_numpy(state_data).float().to(self.device)
        
        return scan_tensors, state_tensor

    def get_action_and_value(self, scan_tensor, state_tensor, params, deterministic=False):
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
            
            # Scale values to environment's action space
            # Note: Environment expects steering velocity (s_min/s_max)
            steer_scale = (params['s_max'] - params['s_min']) / 2
            steer_shift = (params['s_max'] + params['s_min']) / 2
            speed_scale = (params['v_max'] - params['v_min']) / 2
            speed_shift = (params['v_max'] + params['v_min']) / 2
            
            steering = steer_scale * input_td["action"][..., 0].unsqueeze(-1) + steer_shift
            speed = speed_scale * input_td["action"][..., 1].unsqueeze(-1)  + speed_shift
            
            if steering.isnan().any() or speed.isnan().any() or speed.max().item() > params['v_max'] or speed.min().item() < params['v_min'] or steering.max().item() > params['s_max'] or steering.min().item() < params['s_min']:
                raise ValueError("get_action_and_value produced NaN or invalid values.")
            
            if steering.max() > params['s_max'] or steering.min() < params['s_min']:
                steering = torch.clamp(steering, max=params['s_max'], min=params['s_min'])
            if speed.max() > params['v_max'] or speed.min() < params['v_min']:
                speed = torch.clamp(speed, max=params['v_max'], min=params['v_min'])

            action = input_td["action"]
            log_prob = input_td["action_log_prob"]
            value = input_td["state_value"]
            scaled_action = torch.cat((steering, speed), dim=-1)

        return action, log_prob, value, scaled_action

    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        """
        Stores a single step of experience for ALL agents.
        This is a bit complex as we must convert from "list of obs" to "batch."
        """
        
        # Prepare data for TensorDict
        scans, states = self._obs_to_tensors(obs)
        next_scans, next_states = self._obs_to_tensors(next)
        
        _, _, next_value, _ = self.get_action_and_value(
            next_scans, next_states, params={'s_max': 0, 's_min': 0, 'v_max': 0, 'v_min': 0}
        )
        
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
                "state_value": next_value,
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
        
    def calculate_reward(self, next_obs, step, just_crashed, params):
        rewards = []
        for i in range(self.num_agents):
            collided = just_crashed[i] == 1
            reward = 0.0
            
            # -- Forward Speed Reward (positive incentive) --
            forward_speed = next_obs['linear_vels_x'][i]
            normalized_speed = np.clip(forward_speed / params['v_max'], 0, 1)
            reward += normalized_speed * self.SPEED_REWARD_SCALAR
            
            # -- Turning Reward (prevents reward signal dropout in corners) --
            angular_vel = abs(next_obs['ang_vels_z'][i])
            normalized_turn = np.clip(angular_vel / params['s_max'], 0, 1)
            reward += normalized_turn * self.TURN_REWARD_SCALAR
            
            # -- Corner Speed Management (reward appropriate speed for turn sharpness) --
            # For sharp turns (high angular velocity), reward slower speeds
            if normalized_turn > 0.3:  # If turning moderately to sharply
                # Ideal speed decreases as turn sharpness increases
                ideal_corner_speed = 1.0 - (normalized_turn * 0.5)  # Sharp turn = slower ideal speed
                speed_appropriateness = 1.0 - abs(normalized_speed - ideal_corner_speed)
                reward += speed_appropriateness * self.CORNER_SPEED_BONUS
            
            # -- Raceline Progress (main reward signal) --
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
            
            if progress < -self.raceline_length / 2:
                # Agent crossed finish line FORWARD
                progress += self.raceline_length
            elif progress > self.raceline_length / 2:
                # Agent crossed finish line BACKWARD
                progress -= self.raceline_length
                
            # Update tracker and add shaped reward
            self.last_cumulative_distance[i] = current_s
            self.last_wp_index[i] = global_wp_index
            
            # if step != 0 and progress > 0.0:
            #     print(f"Agent {i} made progress: {progress * self.PROGRESS_REWARD_SCALAR:.2f} m at step {step}")
            #     reward += progress * self.PROGRESS_REWARD_SCALAR # Waypoint reached incentive
            
            total_distance = current_s - self.start_s[i]
            
            if total_distance < 0:
                total_distance = 0.0
            
            total_distance += self.raceline_length * self.current_lap_count[i]
            
            new_lap_count = total_distance / self.raceline_length
                
            if new_lap_count > self.current_lap_count[i] + 1:
                print(new_lap_count, self.current_lap_count[i], total_distance, self.raceline_length)
                self.current_lap_count[i] = int(new_lap_count)
                reward += self.LAP_REWARD
                print(f"Lap {new_lap_count} completed by agent {i}! Step: {step} Bonus: {self.LAP_REWARD}")
            else:
                reward += new_lap_count * self.PROGRESS_REWARD_SCALAR
            
            
            # -- Collision Penalty --
            if collided:
                rewards.append(self.COLLISION_PENALTY)
                continue # No rewards if collided
            
            # -- Sideways Slide Penalty --
            sideways_speed = abs(next_obs['linear_vels_y'][i])
            reward += sideways_speed * self.SLIDE_PENALTY_SCALAR

            # -- Reward Clipping --
            if reward <= -1000.0:
                print(f"Large negative reward detected: {reward:.2f}")
                reward = -1.0
            elif reward >= 1000.0:
                print(f"Large positive reward detected: {reward:.2f}")
                reward = 1.0
                
            rewards.append(reward)
                
        return np.array(rewards), np.array(rewards).mean() # Return list and avg
    
    def reset_progress_trackers(self, initial_poses_xy, agent_idxs=None):
        """Resets the cumulative distance tracker for all agents after an episode reset."""
        if agent_idxs is not None:
            for i in agent_idxs:
                current_pos = initial_poses_xy[i]
                
                # Find the globally closest waypoint (no lookahead needed here)
                distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
                closest_wp_index = np.argmin(distances)
                
                start_s_val = self.waypoints_s[closest_wp_index]
                self.last_cumulative_distance[i] = start_s_val
                self.last_wp_index[i] = closest_wp_index
                
                self.start_s[i] = start_s_val
                self.current_lap_count[i] = 0
            return
        
        new_last_cumulative_distance = np.zeros(self.num_agents)
        new_last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        
        # Iterate over all starting positions
        for i in range(self.num_agents):
            current_pos = initial_poses_xy[i]
            
            # Find the globally closest waypoint (no lookahead needed here)
            distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
            closest_wp_index = np.argmin(distances)

            # Set the initial cumulative distance and index            
            start_s_val = self.waypoints_s[closest_wp_index]
            new_last_cumulative_distance[i] = start_s_val
            new_last_wp_index[i] = closest_wp_index
            
            self.start_s[i] = start_s_val
            self.current_lap_count[i] = 0
            
        self.last_cumulative_distance = new_last_cumulative_distance
        self.last_wp_index = new_last_wp_index

    def pretrain_from_demonstrations(self, demo_buffer, epochs=10, batch_size=64):
        """
        Supervised learning from human demonstrations using behavior cloning.
        Also stores demos for continual learning (prevents catastrophic forgetting).
        """
        if len(demo_buffer) < batch_size:
            print(f"Not enough demos ({len(demo_buffer)} < {batch_size}). Skipping pretraining.")
            return
        
        # Store demos for continual BC regularization during RL training
        self.demo_buffer = demo_buffer
        self.demo_pretrain_generation = self.generation_counter  # Mark when pretraining occurred
        print(f"\nPretraining from {len(demo_buffer)} human demonstrations...")
        print(f"Stored {len(demo_buffer)} demos for continual learning")
        print(f"   BC weight: {self.demo_bc_weight_initial} → {self.demo_bc_weight_final} over {self.demo_bc_decay_gens} generations")
        
        # Convert demos to tensors
        scans = torch.stack([torch.from_numpy(d['scan']) for d in demo_buffer]).float().to(self.device)
        states = torch.stack([torch.from_numpy(d['state']) for d in demo_buffer]).float().to(self.device)
        actions = torch.stack([torch.from_numpy(d['action']) for d in demo_buffer]).float().to(self.device)
        
        total_loss = 0.0
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(demo_buffer))
            epoch_loss = 0.0
            
            for i in range(0, len(demo_buffer), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_scans = scans[batch_indices]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                
                # Get actor's predicted action distribution
                input_td = TensorDict({
                    "observation_scan": batch_scans,
                    "observation_state": batch_states
                }, batch_size=batch_scans.shape[0])
                
                self.actor_module(input_td)
                predicted_loc = input_td["loc"]
                
                # MSE loss between predicted and human actions
                loss = torch.nn.functional.mse_loss(predicted_loc, batch_actions)
                
                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / (len(demo_buffer) // batch_size)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        print(f"Pretraining complete. Avg loss: {total_loss/epochs:.4f}\n")

    def learn(self, reward_avg):
        """
        The "Coaching Session" where we train the networks.
        """
        print("Starting learning phase...")
        # Get all data from the "generation"
        # .sample() with no args returns the *entire* buffer
        data = self.buffer.sample(batch_size=len(self.buffer))
        minibatch_size = self.minibatch_size
        
        if len(data) < minibatch_size:
            minibatch_size = len(data)
            with torch.no_grad():                
                self.advantage_module(data.to(self.device)) 
        else:
            with torch.no_grad():
                self.critic_module.to("cpu")
                self.advantage_module.device = "cpu"
                data.to("cpu")
                
                self.advantage_module(data) 
                
                data.to(self.device)
                self.critic_module.to(self.device)
                self.advantage_module.device = self.device
        
        data = data.flatten(0, 1)
        total_samples = data.batch_size[0]
        
        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["reward_avg"] = [reward_avg]
        
        # Determine number of epochs based on whether we have demonstrations
        # Fewer epochs with demos to prevent PPO from overwhelming BC regularization
        num_epochs = self.epochs_with_demos if (self.demo_buffer is not None and len(self.demo_buffer) > 0) else self.epochs
        if self.demo_buffer is not None and len(self.demo_buffer) > 0:
            print(f"Training with BC regularization: {num_epochs} epochs (reduced from {self.epochs})")
        
        # Train for several epochs on this same data
        for _ in range(num_epochs): # PPO repeats training on the same data            
            # Shuffle data indices for this epoch
            indices = torch.randperm(total_samples)
            
            # Loop over all samples in minibatches
            for start in range(0, total_samples, minibatch_size):
                end = start + minibatch_size
                if end > total_samples:
                    continue # Skip the last, incomplete minibatch
                
                minibatch_indices = indices[start:end]
                
                # Sample the minibatch from the full dataset
                minibatch_data = data[minibatch_indices].to(self.device)
                
                # Get the loss from torchrl's PPO module
                loss_td = self.loss_module(minibatch_data) # <--- Run on the minibatch

                # # Sum the actor and critic losses
                actor_loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
                critic_loss = loss_td["loss_critic"]
                
                # -- Backpropagation for PPO --
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=2.0)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=2.0)
                self.critic_optimizer.step()
                
                # --- Behavior Cloning Regularization (Prevent Catastrophic Forgetting) ---
                # Separate gradient step to avoid mixing loss scales
                if self.demo_buffer is not None and len(self.demo_buffer) > 0:
                    # Calculate current BC weight with decay schedule
                    if self.demo_pretrain_generation is not None:
                        gens_since_pretrain = self.generation_counter - self.demo_pretrain_generation
                        if gens_since_pretrain < self.demo_bc_decay_gens:
                            # Linear decay from initial to final weight
                            decay_progress = gens_since_pretrain / self.demo_bc_decay_gens
                            current_bc_weight = self.demo_bc_weight_initial - decay_progress * (self.demo_bc_weight_initial - self.demo_bc_weight_final)
                        else:
                            current_bc_weight = self.demo_bc_weight_final
                    else:
                        current_bc_weight = self.demo_bc_weight_initial
                    
                    # Sample a minibatch from demonstration buffer (50% of minibatch)
                    demo_batch_size = min(len(self.demo_buffer), minibatch_size // 2)
                    demo_indices = torch.randint(0, len(self.demo_buffer), (demo_batch_size,)).tolist()
                    
                    demo_scans = torch.stack([torch.from_numpy(self.demo_buffer[idx]["scan"]).float() for idx in demo_indices]).to(self.device)
                    demo_states = torch.stack([torch.from_numpy(self.demo_buffer[idx]["state"]).float() for idx in demo_indices]).to(self.device)
                    demo_actions = torch.stack([torch.from_numpy(self.demo_buffer[idx]["action"]).float() for idx in demo_indices]).to(self.device)
                    
                    # Get actor's prediction for demonstration observations
                    demo_input_td = TensorDict({
                        "observation_scan": demo_scans,
                        "observation_state": demo_states
                    }, batch_size=demo_scans.shape[0])
                    
                    self.actor_module(demo_input_td)
                    predicted_actions = demo_input_td["loc"]
                    
                    # MSE loss between predicted and human actions
                    bc_loss = torch.nn.functional.mse_loss(predicted_actions, demo_actions)
                    
                    # Apply BC loss as separate gradient step
                    self.actor_optimizer.zero_grad()
                    bc_loss_weighted = current_bc_weight * bc_loss
                    bc_loss_weighted.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=2.0)
                    self.actor_optimizer.step()
                    
                    # Log BC activity occasionally
                    if start == 0:  # First minibatch of each epoch
                        print(f"  BC: weight={current_bc_weight:.3f}, loss={bc_loss.item():.4f}, weighted={bc_loss_weighted.item():.4f}")

                for key in self.diagnostic_keys:
                    if key in loss_td.keys():
                            value = loss_td[key].detach().cpu().item()
                            current_gen_diagnostics[key].append(value)
        
        # Adaptive entropy coefficient - update ONCE per generation based on average entropy
        if current_gen_diagnostics.get("entropy"):
            avg_entropy = np.mean(current_gen_diagnostics["entropy"])
            # Target entropy for continuous action spaces (positive values)
            target_entropy = 0.8  # Healthy exploration level for 2D continuous actions

            # Adjust entropy coefficient based on average entropy across generation
            if avg_entropy < target_entropy:
                # Aggressive increase when entropy is too low
                if avg_entropy < 0.5:  # Critically low
                    adjustment = 1.15  # 15% increase
                elif avg_entropy < 0.7:  # Low
                    adjustment = 1.10  # 10% increase
                else:  # Slightly low
                    adjustment = 1.05  # 5% increase
                self.loss_module.entropy_coeff.data *= adjustment
            elif avg_entropy > target_entropy + 0.3:  # Too high (>1.1)
                self.loss_module.entropy_coeff.data *= 0.95  # Slow decrease when too high
            # Otherwise maintain current coefficient

            # Clamp to reasonable range
            self.loss_module.entropy_coeff.data.clamp_(0.1, 0.8)  # Increased minimum from 0.05 to 0.1
            print(f"Entropy coeff adjusted to {self.loss_module.entropy_coeff.data.item():.4f} (avg entropy: {avg_entropy:.3f})")
                
        self.generation_counter += 1
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key)
            if values: # Check if list is not empty
                avg_value = np.mean(values)
                self.diagnostics_history[key].append(avg_value)
        
        if self.generation_counter > 0: self._plot_historical_diagnostics()
        
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

        plt.style.use('dark_background')
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 4 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # Ensure axes is always iterable
        
        # Set global font size
        plt.rcParams['font.size'] = 16  # Adjust as needed

        # Set global line width
        plt.rcParams['lines.linewidth'] = 3 
        
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