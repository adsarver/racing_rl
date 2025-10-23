import numpy as np
import torch
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ValueOperator, ProbabilisticActor
from torch.distributions import Normal, Independent

from model import ActorNetwork, CriticNetwork

class PPOAgent:
    def __init__(self, num_agents):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-4
        self.lr_critic = 1e-3
        self.gamma = 0.99  # Discount factor for future rewards
        self.gae_lambda = 0.95 # Lambda for GAE (Advantage calculation)
        self.clip_epsilon = 0.2 # PPO clip parameter
        self.state_dim = 3 # x_vel, y_vel, z_ang_vel
        self.num_scan_beams = 1080
        
        # --- Reward Scalars ---
        self.SPEED_REWARD_SCALAR = 1.0
        self.WALL_PENALTY = 100.0
        self.AGENT_PENALTY = 50.0
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
            entropy_coef=0.01,
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
            storage=ListStorage(max_size=2048) # 2048 steps per generation
        )

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
        self.buffer.add(step_data)
        
    def calculate_reward(self, done_from_env, next_obs):
        rewards = []
        for i in range(self.num_agents):
            # -- Speed Reward --
            current_speed = next_obs['linear_vels_x'][i]            
            reward = current_speed * self.SPEED_REWARD_SCALAR # Encourages forward movement while discouraging backwards movement
            
            # -- Sideways Penalty --
            sideways_speed = next_obs['linear_vels_y'][i]
            reward -= abs(sideways_speed) * self.SLIDE_PENALTY_SCALAR

            is_agent_collision = next_obs['collisions'][i]
            
            # We must check the ego agent (i == 0) differently,
            # since its 'done' flag is special.
            if i == 0:
                if done_from_env:
                    if is_agent_collision:
                        reward -= self.AGENT_PENALTY
                    else:
                        reward -= self.WALL_PENALTY
                elif is_agent_collision:
                    reward -= self.AGENT_PENALTY
            
            # For all other agents (i != 0)
            elif is_agent_collision:
                reward -= self.AGENT_PENALTY
                
            rewards.append(reward)
        return rewards, np.array(rewards).mean() # Return list and avg

    def learn(self):
        """
        The "Coaching Session" where we train the networks.
        """
        print("Starting learning phase...")
        
        # Get all data from the "generation"
        # .sample() with no args returns the *entire* buffer
        data = self.buffer.sample(batch_size=len(self.buffer)).to(self.device)

        # Calculate advantages (GAE)
        # how much "better" or "worse" each action was
        with torch.no_grad():
            data = data.permute(1, 0)
            self.advantage_module(data) # This adds "advantage" and "value_target" to the data
                    
        # Train for several epochs on this same data
        for _ in range(10): # PPO repeats training on the same data            
            # Get the loss from torchrl's PPO module
            loss_td = self.loss_module(data)
            
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
            
        # Clear the buffer for the next "generation"
        self.buffer.empty()
        print("Learning complete.")