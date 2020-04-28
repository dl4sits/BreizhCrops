import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os

__all__ = ['LSTM']

class LSTM(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, hidden_dims=128, num_layers=4, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.use_layernorm = use_layernorm

        self.d_model = num_layers * hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims * num_layers, num_classes, bias=True)


    def logits(self, x):

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        logits = self.linear_class.forward(h)

        return logits

    def forward(self, x):
        logprobabilities = F.log_softmax(self.logits(x), dim=-1)
        return logprobabilities

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot
