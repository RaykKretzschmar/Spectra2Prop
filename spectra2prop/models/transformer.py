import torch
import torch.nn as nn

class SpectraTransformer(nn.Module):
    def __init__(self, input_dim=100, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)