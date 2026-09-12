import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen

class SpikingQNetwork(nn.Module):
    def __init__(self, action_size=3, num_steps=10, beta=0.9):
        super(SpikingQNetwork, self).__init__()
        
        self.num_steps = num_steps
        self.action_size = action_size
        
        # 1. Feature Extraction (CNN)
        # Input shape: (Batch, Channels, Height, Width) -> (B, 1, 84, 84)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4) # Out: (16, 20, 20)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # Out: (32, 9, 9)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # Out: (64, 7, 7)
        self.lif3 = snn.Leaky(beta=beta)
        
        # 2. Flatten and Fully Connected
        self.flatten = nn.Flatten()
        
        # 64 channels * 7 * 7 spatial dimensions = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.lif4 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(512, action_size)
        # The final layer does not need to spike if we use its accumulated membrane potential for Q-values
        self.lif5 = snn.Leaky(beta=beta, reset_mechanism="none") # Disable reset to accumulate voltage
        
    def forward(self, x):
        """
        x: Raw float tensor from environment of shape (Batch, 1, 84, 84)
        """
        # Initialize membrane potentials for all LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        # Generate spike train using rate coding (Pixel intensity -> Spike probability)
        # Resulting shape: (num_steps, Batch, 1, 84, 84)
        spike_in = spikegen.rate(x, num_steps=self.num_steps)
        
        # We will accumulate the membrane potential of the output layer over time
        # to represent the Q-values.
        q_values = torch.zeros(x.size(0), self.action_size, device=x.device)
        total_spikes = torch.tensor(0.0, device=x.device)
        
        for step in range(self.num_steps):
            # Pass the spikes for the current time step through the network
            cur_in = spike_in[step]
            
            cur_conv1 = self.conv1(cur_in)
            spk1, mem1 = self.lif1(cur_conv1, mem1)
            
            cur_conv2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur_conv2, mem2)
            
            cur_conv3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur_conv3, mem3)
            
            # Flatten spatial dimensions
            cur_flat = self.flatten(spk3)
            
            cur_fc1 = self.fc1(cur_flat)
            spk4, mem4 = self.lif4(cur_fc1, mem4)
            
            cur_fc2 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur_fc2, mem5)
            
            # Accumulate the final layer's membrane potential
            q_values += mem5
            
            # Tally spikes for sparsity constraint
            total_spikes += spk1.sum() + spk2.sum() + spk3.sum() + spk4.sum()
            
        # Return the accumulated membrane potential (Q-values) and total spike count
        return q_values, total_spikes


if __name__ == "__main__":
    # Small test block to verify the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the network
    net = SpikingQNetwork().to(device)
    
    # Create a dummy batch of 2 frames (Batch, Channels, Height, Width)
    dummy_input = torch.rand(2, 1, 84, 84).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    q_out, spk_count = net(dummy_input)
    
    print(f"Output shape: {q_out.shape}")
    print(f"Output values (Q-values):\n{q_out.detach().cpu().numpy()}")
    print(f"Total Spikes generated in pass: {spk_count.item()}")
    
    print("\nNetwork structure is working successfully!")
