import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from models.ClassificationModel import ClassificationModel
from models.transformer.Models import Encoder
from models.AttentionModule import Attention
from torch.nn.modules.normalization import LayerNorm

SEQUENCE_PADDINGS_VALUE=-1

class TransformerEncoder(ClassificationModel):
    def __init__(self, in_channels=13, len_max_seq=100,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, nclasses=6):

        super(TransformerEncoder, self).__init__()

        self.d_model = 512

        self.inlayernorm = nn.LayerNorm(in_channels)
        self.convlayernorm = nn.LayerNorm(d_model)
        self.outlayernorm = nn.LayerNorm(d_model)

        self.inconv = torch.nn.Conv1d(in_channels, d_model, 1)

        self.encoder = Encoder(
            n_src_vocab=None, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.outlinear = nn.Linear(d_model, nclasses, bias=False)

    def _logits(self, x):
        # b,d,t - > b,t,d
        x = x.transpose(1,2)

        x = self.inlayernorm(x)

        # b,
        x = self.inconv(x.transpose(1,2)).transpose(1,2)

        x = self.convlayernorm(x)

        batchsize, seq, d = x.shape
        src_pos = torch.arange(1, seq + 1, dtype=torch.long).expand(batchsize, seq)

        if torch.cuda.is_available():
            src_pos = src_pos.cuda()

        enc_output, enc_slf_attn_list = self.encoder.forward(src_seq=x, src_pos=src_pos, return_attns=True)

        enc_output = self.outlayernorm(enc_output)

        logits = self.outlinear(enc_output)[:,-1,:]

        return logits, None, None, None

    def forward(self, x):

        logits, *_ = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=-1)

        return logprobabilities, None, None, None

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

