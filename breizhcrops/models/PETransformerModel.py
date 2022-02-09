import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PETransformerModel']

class PETransformerModel(nn.Module):
    def __init__(self, input_dim=13, num_classes=9, d_model=64, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.017998950510888446, max_len=200):

        super(PETransformerModel, self).__init__()
        self.modelname = f"PeTransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.flatten = Flatten()
        self.outlinear = Linear(d_model, num_classes)
        self.pe = PositionalEncoding(d_model, max_len=max_len)

        """
        self.sequential = Sequential(
            ,
            ,
            ,
            ,
            ReLU(),

        )
        """

    def forward(self,x):
        x = self.inlinear(x)
        x = self.pe(x)
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D
        x = x.max(1)[0]
        x = self.relu(x)
        logits = self.outlinear(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)
