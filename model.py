# model.py
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    """
    A 1D CNN-based policy network (Actor).
    It takes a LIDAR scan and outputs a probability distribution
    over the continuous actions (steering and speed).
    """
    def __init__(self, num_scan_beams=1080, state_dim=2, action_dim=2):
        super(ActorNetwork, self).__init__()
        
        # Input shape: (num_agents, 1, num_scan_beams)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        dummy_input = torch.randn(1, 1, num_scan_beams)
        conv_output_size = self._get_conv_output_size(dummy_input)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # We need two outputs for each action: one for the mean (the 'best' guess)
        # and one for the standard deviation (the 'confidence').

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(128, action_dim)
        
        # Head for the standard deviation (log_std) of the action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, scan_tensor, state_tensor):
        # NN Layers
        vision_features = self.conv_layers(scan_tensor)
        vision_features = vision_features.view(vision_features.size(0), -1)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        x = self.fc_layers(combined_features)
        
        # Heads
        action_mean = self.mean_head(x)
        
        # Apply Tanh to steering_mean to keep it between [-1, 1]
        steering_mean = torch.tanh(action_mean[:, 0].unsqueeze(1)) 
        
        # Apply Sigmoid to speed_mean to keep it between [0, 1]
        speed_mean = torch.sigmoid(action_mean[:, 1].unsqueeze(1))
        
        # Combine them
        mu = torch.cat((steering_mean, speed_mean), dim=1)
        
        # Get the standard deviation (std)
        action_std = torch.exp(self.log_std)
        
        # 'expand' std to match the batch size
        action_std = action_std.expand_as(mu)
        
        # Create the Normal (Gaussian) distribution
        action_dist = Normal(mu, action_std)
        
        return action_dist
    
# In model.py, add this new class:

class CriticNetwork(nn.Module):
    def __init__(self, num_scan_beams=1080, state_dim=3):
        super(CriticNetwork, self).__init__()
        
        # Vision Stream (LIDAR)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        dummy_input = torch.randn(1, 1, num_scan_beams)
        conv_output_size = self._get_conv_output_size(dummy_input)
        
        # Combined Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Output Head
        # Outputs a single value, no activation function
        self.value_head = nn.Linear(128, 1)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, scan_tensor, state_tensor):
        if scan_tensor.ndim == 4:
            T, B = scan_tensor.shape[0:2]
            
            scan_tensor = scan_tensor.view(T * B, 1, -1) 
            state_tensor = state_tensor.view(T * B, -1)
            
            unflatten_output = True
        else:
            unflatten_output = False
        
        vision_features = self.conv_layers(scan_tensor)
        vision_features = vision_features.view(vision_features.size(0), -1)
        
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        x = self.fc_layers(combined_features)
        
        value = self.value_head(x)
        
        if unflatten_output:
            return value.view(T, B, 1)
        else:
            return value