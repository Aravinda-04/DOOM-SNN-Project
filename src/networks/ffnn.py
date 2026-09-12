import torch
import torch.nn as nn

class FeedForwardQNetwork(nn.Module):
    def __init__(self, action_size=3):
        super(FeedForwardQNetwork, self).__init__()
        
        self.action_size = action_size
        
        # 1. Feature Extraction (CNN) - standard ANNs use continuous activations like ReLU
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        
        # 2. Flatten and Fully Connected
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        """
        x: Raw float tensor from environment of shape (Batch, 1, 84, 84)
        In an FFNN, there is no time dimension or spike generation.
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        x = self.flatten(x)
        
        x = self.relu4(self.fc1(x))
        q_values = self.fc2(x)
        
        # We return a dummy zero for total_spikes so that train.py can unpack it
        # without crashing when training the FFNN.
        total_spikes = torch.tensor(0.0, device=x.device)
        
        return q_values, total_spikes
