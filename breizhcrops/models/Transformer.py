from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import torch

class TranformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, sequencelength=13, d_model=512, n_head=8, n_layers=1,
                 d_inner=2048, activation="relu", dropout=0.2):
        super(TranformerModel, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.sequential = Sequential(
            Linear(input_dim, d_model),
            ReLU(),
            TransformerEncoder(encoder_layer, n_layers, encoder_norm),
            Flatten(),
            ReLU(),
            Linear(d_model*sequencelength, num_classes)
        )

    def forward(self,x):
        logits = self.sequential(x)
        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

if __name__=="__main__":
    model = TranformerModel(input_dim=8, num_classes=2).to(torch.device("cpu"))
    src = torch.rand(10, 32, 8)
    model(src)
    print()