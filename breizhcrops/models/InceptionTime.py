import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

__all__ = ['InceptionTime']

class InceptionTime(nn.Module):

    def __init__(self,num_classes, input_dim=1,num_layers=6, hidden_dims=128,use_bias=False, use_residual= True, device=torch.device("cpu")):
        super(InceptionTime, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.inception_modules_list = [InceptionModule(input_dim = input_dim, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device)]
        for i in range(num_layers-1):
          self.inception_modules_list.append(InceptionModule(input_dim = hidden_dims, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device))
        self.shortcut_layer_list = [ShortcutLayer(input_dim,hidden_dims,stride = 1, bias = False)]
        for i in range(num_layers//3):
          self.shortcut_layer_list.append(ShortcutLayer(hidden_dims,hidden_dims,stride = 1, bias = False))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims,num_classes)

        self.to(device)

    def forward(self,x):
        # N x T x D -> N x D x T
        x = x.transpose(1,2)
        input_res = x
        
        for d in range(self.num_layers):
            x = self.inception_modules_list[d](x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer_list[d//3](input_res, x)
                input_res = x
        x = self.avgpool(x).squeeze(2)
        x = self.outlinear(x)
        logprobabilities = F.log_softmax(x, dim=-1)
        return logprobabilities

class InceptionModule(nn.Module):
    def __init__(self,input_dim=32, kernel_size=40, num_filters= 32, residual=False, use_bias=False, device=torch.device("cpu")):
        super(InceptionModule, self).__init__()

        self.residual = residual

        self.bottleneck = nn.Conv1d(input_dim, num_filters , kernel_size = 1, stride=1, padding= 0,bias=use_bias)
        
        # the for loop gives 40, 20, and 10 convolutions
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size+1, stride=1, bias= False, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s] #padding is 1 instead of kernel_size//2
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, num_filters,kernel_size=1, stride = 1,padding=0, bias=use_bias) 
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU()
        )

        self.to(device)


    def forward(self, input_tensor):
        # collapse feature dimension

        input_inception = self.bottleneck(input_tensor)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1) 
        features = self.bn_relu(features)

        return features

class ShortcutLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bias):
        super(ShortcutLayer, self).__init__()
        self.sc = nn.Sequential(nn.Conv1d(in_channels=in_planes,
                                          out_channels=out_planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=bias),
                                nn.BatchNorm1d(num_features=out_planes))
        self.relu = nn.ReLU()

    def forward(self, input_tensor, out_tensor):
        x = out_tensor + self.sc(input_tensor)
        x = self.relu(x)

        return x
        
if __name__=="__main__":
    model = InceptionTime(input_dim=13, hidden_dims=128, num_classes=2, num_layers=6).to(torch.device("cpu"))
    #                N   T   D
    src = torch.rand(16, 128, 13)
    model(src)
    print()
