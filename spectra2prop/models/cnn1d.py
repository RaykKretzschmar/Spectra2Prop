import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    def __init__(self, num_bins=2000, num_classes=10, embedding_dim=128):
        super(SpectralCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.embedding_dim = embedding_dim
        self.fc_embed = nn.Linear(64, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        embedding = F.relu(self.fc_embed(x))
        
        logits = self.fc_out(embedding)
        return logits, embedding

