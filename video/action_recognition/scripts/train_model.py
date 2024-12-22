"""
Model definition and training code for action recognition
"""

import torch
import torch.nn as nn

class PoseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # LSTM with hidden_size=128, input_size=51 (17 keypoints * 3)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 51
            hidden_size=hidden_size,  # 128
            num_layers=2,
            batch_first=True,
            bidirectional=False  # Single direction to match state dict
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 128 -> 128
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 128 -> 101
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take the last time step
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out