import torch
import torch.nn as nn

class Spectra1DCNN(nn.Module):
    def __init__(self, input_dim=2000, num_classes=31, dropout=0.5):
        """
        Args:
            input_dim (int): Number of bins in the input spectrum vector.
            num_classes (int): Number of ClassyFire superclasses.
            dropout (float): Dropout probability.
        """
        super(Spectra1DCNN, self).__init__()
        
        # Convolutional Block 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Convolutional Block 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Convolutional Block 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        )
        
        # Fully Connected Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of shape (batch_size, 1, input_dim)
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out
