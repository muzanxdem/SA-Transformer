import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for feature sequences in tabular data"""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for tabular data classification.
    Treats each feature as a token in a sequence.
    
    Architecture:
    - Input features are projected to d_model dimensions
    - Positional encoding is added
    - Transformer encoder processes the sequence
    - Global pooling (mean) aggregates sequence
    - Classification head outputs class logits
    """
    def __init__(self, input_features, num_classes, d_model=128, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.1, max_seq_len=1000):
        super(TransformerModel, self).__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        
        # Input projection: map each feature to model dimension
        # Each feature becomes a token, so we project from 1 to d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.classifier[0].weight.data.uniform_(-initrange, initrange)
        self.classifier[3].weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape: (batch, features) -> (features, batch, 1)
        # Each feature becomes a token in the sequence
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = x.transpose(0, 1)  # (features, batch, 1)
        
        # Project to model dimension: (features, batch, 1) -> (features, batch, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder: (seq_len, batch, d_model) -> (seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        
        # Global pooling: average over sequence length
        # (seq_len, batch, d_model) -> (batch, d_model)
        x = x.mean(dim=0)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits

