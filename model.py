# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionEncoder(nn.Module):
    def __init__(self, num_scan_beams=1080):
        super(VisionEncoder, self).__init__()
        
        # Input shape: (batch_size, 1, num_scan_beams)
        # Based off of TinyLidarNet from: https://arxiv.org/pdf/2410.07447
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4),
            nn.BatchNorm1d(24, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4),
            nn.BatchNorm1d(36, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2),
            nn.BatchNorm1d(48, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.ReLU(),            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the output size of the conv layers
        dummy_input = torch.randn(1, 1, num_scan_beams)
        self.output_size = self._get_conv_output_size(dummy_input)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, scan_tensor):
        return self.conv_layers(scan_tensor)

class ActorNetwork(nn.Module):
    """
    A 1D CNN-based policy network (Actor).
    It takes a LIDAR scan and outputs a probability distribution
    over the continuous actions (steering and speed).
    """
    def __init__(
        self, 
        state_dim=3, 
        action_dim=2,
        encoder=None
        ):
        super(ActorNetwork, self).__init__()
        
        # Input shape: (num_agents, 1, num_scan_beams)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        fc_input_size = conv_output_size + state_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size),
            nn.BatchNorm1d(fc_input_size, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(fc_input_size, 100),
            nn.BatchNorm1d(100, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10, track_running_stats=False),
            nn.ReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(10, action_dim)
        
        # log_std head
        self.log_std_head = nn.Linear(10, action_dim)

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
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        x = self.fc_layers(combined_features)
        
        # Heads
        loc = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = F.softplus(log_std)
        
        # Get the standard deviation (std)
        scale = torch.exp(log_std)
        
        if unflatten_output:
            loc = loc.view(T, B, -1)
            scale = scale.view(T, B, -1)

        return loc, scale


class CriticNetwork(nn.Module):
    def __init__(self, state_dim=3, encoder=None):
        super(CriticNetwork, self).__init__()
        
        # Vision Stream (LIDAR)
        self.conv_layers = encoder
        
        conv_output_size = self.conv_layers.output_size
        fc_input_size = conv_output_size + state_dim
        
        # Combined Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size),
            nn.BatchNorm1d(fc_input_size, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(fc_input_size, 100),
            nn.BatchNorm1d(100, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

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
        
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        value = self.fc_layers(combined_features)
        
        if unflatten_output:
            return value.view(T, B, 1)
        else:
            return value