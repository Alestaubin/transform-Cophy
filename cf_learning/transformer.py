import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=30):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. Project input (x,y,z) to hidden dimension d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. The Transformer Encoder Layer
        # batch_first=True is CRITICAL if your data is (Batch, Time, Feat)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        
        # 4. Stack multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (Batch, Time, 3)
        
        # Embed
        x = self.input_projection(x) # (B, T, d_model)
        
        # Add Position Info
        x = self.pos_encoder(x)      # (B, T, d_model)
        
        # Apply Transformer
        # out shape: (B, T, d_model)
        out = self.transformer_encoder(x)
        
        # Pooling: Aggregate over Time to get one vector per trajectory
        # Option A: Mean Pooling (Average all timesteps)
        confounders = out.mean(dim=1) # (B, d_model)
        
        # Option B: Max Pooling (Good for detecting single sharp events like collisions)
        # confounders = out.max(dim=1)[0]
        
        return confounders
