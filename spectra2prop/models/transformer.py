import torch
import torch.nn as nn

class SpectraTransformer(nn.Module):
    def __init__(self, num_bins=2000, num_classes=10, d_model=64, nhead=4, num_layers=2, patch_size=20):
        super(SpectraTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_bins // self.patch_size
        
        self.embedding = nn.Linear(self.patch_size, d_model)
        
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        if x.dim() == 3:
            b, c, l = x.shape
            pass
        else:
            b, l = x.shape
            x = x.unsqueeze(1)
            b, c, l = x.shape

        x = x.squeeze(1).view(b, self.num_patches, self.patch_size)
        
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1) 
        
        logits = self.fc_out(x)
        return logits
