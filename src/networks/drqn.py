import torch
import torch.nn as nn

class DRQN(nn.Module):
    def __init__(self, action_size=3, **kwargs):
        super(DRQN, self).__init__()
        
        self.action_size = action_size
        self.is_recurrent = True
        
        # 1. Feature Extraction (CNN)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # 2. LSTM layer
        self.lstm = nn.LSTM(input_size=3136, hidden_size=512, batch_first=True)
        
        # 3. Output layer
        self.fc = nn.Linear(512, action_size)
        
    def forward(self, x, hidden=None):
        """
        x: Raw float tensor from environment.
           Can be shape (Batch, Seq_Len, 1, 84, 84) during training,
           or (Batch, 1, 84, 84) during evaluation.
        hidden: Tuple of (h_n, c_n), each of shape (1, Batch, 512).
                If None, the LSTM initializes with zeros.
        """
        batch_size = x.size(0)
        
        if len(x.shape) == 4:
            # (Batch, Channels, Height, Width) -> Add sequence dimension
            seq_len = 1
            x = x.unsqueeze(1)
        else:
            # (Batch, Seq_Len, Channels, Height, Width)
            seq_len = x.size(1)
            
        # Fold sequence into batch dimension for CNN processing
        x = x.view(batch_size * seq_len, 1, 84, 84)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        
        # Unfold back into (Batch, Seq_Len, Features)
        x = x.view(batch_size, seq_len, -1)
        
        # Process sequence through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Compute Q-values
        q_values = self.fc(lstm_out)
        
        # If seq_len == 1, we can optionally squeeze it out for compatibility
        if seq_len == 1:
            q_values = q_values.squeeze(1)
            
        total_spikes = torch.tensor(0.0, device=x.device)
        return q_values, hidden, total_spikes
