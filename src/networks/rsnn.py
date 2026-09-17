import torch
import torch.nn as nn
import snntorch as snn

class RSNNQNetwork(nn.Module):
    def __init__(self, action_size=3, num_steps=10, beta=0.9):
        super(RSNNQNetwork, self).__init__()
        
        self.num_steps = num_steps
        self.action_size = action_size
        
        # 1. Feature Extraction (CNN)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.flatten = nn.Flatten()
        
        # 2. Recurrent Fully Connected Layer
        self.fc1 = nn.Linear(3136, 512)
        # This linear layer feeds the spikes of lif4 back into lif4 in the next time step
        self.recurrent = nn.Linear(512, 512)
        self.lif4 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(512, action_size)
        self.lif5 = snn.Leaky(beta=beta, reset_mechanism="none")
        
    def forward(self, x):
        """
        x: Raw float tensor from environment of shape (Batch, 1, 84, 84)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        q_values = torch.zeros(x.size(0), self.action_size, device=x.device)
        total_spikes = torch.tensor(0.0, device=x.device)
        
        # Initialize an empty spike tensor for the recurrent connection
        spk4 = torch.zeros(x.size(0), 512, device=x.device)
        
        for step in range(self.num_steps):
            # Deterministic Direct Coding
            cur_conv1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur_conv1, mem1)
            
            cur_conv2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur_conv2, mem2)
            
            cur_conv3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur_conv3, mem3)
            
            cur_flat = self.flatten(spk3)
            
            # The input to lif4 is the feed-forward input PLUS the recurrent input from the previous step
            cur_fc1 = self.fc1(cur_flat) + self.recurrent(spk4)
            spk4, mem4 = self.lif4(cur_fc1, mem4)
            
            cur_fc2 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur_fc2, mem5)
            
            q_values += mem5
            
            total_spikes += spk1.sum() + spk2.sum() + spk3.sum() + spk4.sum()
            
        return q_values, total_spikes
