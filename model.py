# model.py
import torch
import torch.nn as nn
import numpy as np

class ActorNetwork(nn.Module):
    """
    A 1D CNN-based policy network (Actor).
    It takes a LIDAR scan and outputs a probability distribution
    over the continuous actions (steering and speed).
    """
    def __init__(self, num_scan_beams=1080, state_dim=3, action_dim=2, max_speed=20.0, min_speed=-5.0, max_steering=0.4189):
        super(ActorNetwork, self).__init__()
        self.MIN_SPEED = min_speed
        self.SPEED_RANGE = max_speed - min_speed
        self.MAX_STEERING = max_steering
        
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
        if scan_tensor.ndim == 4:
            T, B = scan_tensor.shape[0:2]
            # Flatten Time and Batch dims for Conv1D
            scan_tensor = scan_tensor.reshape(T * B, 1, -1) 
            state_tensor = state_tensor.reshape(T * B, -1)
            unflatten_output = True
        else:
            unflatten_output = False
            
        # NN Layers            
        vision_features = self.conv_layers(scan_tensor)
        vision_features = vision_features.view(vision_features.size(0), -1)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        x = self.fc_layers(combined_features)
        
        # Heads
        action_mean = self.mean_head(x)
        
        # Apply Tanh to steering_mean to keep it between [-1, 1]
        steering_tanh = torch.tanh(action_mean[..., 0].unsqueeze(-1)) 
        steering_mean = steering_tanh * self.MAX_STEERING
        
        # Apply Sigmoid to speed_mean to keep it between [0, 1]
        speed_tanh = torch.sigmoid(action_mean[..., 1].unsqueeze(1))
        speed_mean = (speed_tanh + 1.0) * 0.5 * self.SPEED_RANGE + self.MIN_SPEED
        
        # Combine them
        loc = torch.cat((steering_mean, speed_mean), dim=-1)
        
        # Get the standard deviation (std)
        action_std = torch.exp(self.log_std)
        
        # 'expand' std to match the batch size
        scale = action_std.expand_as(loc)
        
        if unflatten_output:
            loc = loc.view(T, B, -1)
            scale = scale.view(T, B, -1)

        return loc, scale

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
            # Flatten Time and Batch dims for Conv1D
            scan_tensor = scan_tensor.reshape(T * B, 1, -1) 
            state_tensor = state_tensor.reshape(T * B, -1)
            unflatten_output = True
        else:
            unflatten_output = False
        
        # NN Layers            
        vision_features = self.conv_layers(scan_tensor)
        vision_features = vision_features.view(vision_features.size(0), -1)
        
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        x = self.fc_layers(combined_features)
        
        value = self.value_head(x)
        
        if unflatten_output:
            return value.view(T, B, 1)
        else:
            return value