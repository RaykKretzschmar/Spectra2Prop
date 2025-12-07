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

class SimpleTransformer(nn.Module):
    def __init__(self, num_bins=2000, num_classes=10, d_model=128, nhead=4, num_layers=2):
        super(SimpleTransformer, self).__init__()
        
        self.patch_size = 20
        self.num_patches = num_bins // self.patch_size
        self.embedding = nn.Linear(self.patch_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        b, c, l = x.shape
        x = x.squeeze(1).view(b, self.num_patches, self.patch_size)
        
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1) 
        
        logits = self.fc_out(x)
        return logits, x
