import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

__all__ = ['InceptionTime']

class InceptionTime(nn.Module):

    def __init__(self, input_dim=13, num_classes=9, num_layers=4, hidden_dims=64, use_bias=False, device=torch.device("cpu")):
        super(InceptionTime, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        self.inlinear = nn.Linear(input_dim, hidden_dims*4)
        self.num_layers = num_layers
        self.inception_modules_list = [InceptionModule(kernel_size=32, num_filters=hidden_dims*4,
                                                       use_bias=use_bias, device=device) for _ in range(num_layers)]
        self.inception_modules = nn.Sequential(
            *self.inception_modules_list
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims*4,num_classes)

        self.to(device)

    def forward(self,x):
        # N x T x D -> N x D x T
        x = x.transpose(1,2)

        # expand dimensions
        x = self.inlinear(x.transpose(1, 2)).transpose(1, 2)
        for i in range(self.num_layers):
            x = self.inception_modules_list[i](x)

            #if self.use_residual and d % 3 == 2:
            #    x = self._shortcut_layer(input_res, x)
            #    input_res = x
        x = self.avgpool(x).squeeze(2)
        x = self.outlinear(x)
        logprobabilities = F.log_softmax(x, dim=-1)
        return logprobabilities

class InceptionModule(nn.Module):
    def __init__(self, kernel_size=32, num_filters=128, residual=True, use_bias=False, device=torch.device("cpu")):
        super(InceptionModule, self).__init__()

        self.residual = residual

        self.bottleneck = nn.Linear(num_filters, out_features=1, bias=use_bias)

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(1, num_filters//4, kernel_size=kernel_size+1, stride=1, bias=use_bias, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s]

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(num_filters, num_filters//4, kernel_size=1, padding=0, bias=use_bias)
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )

        if residual:
            self.residual_relu = nn.ReLU()

        self.to(device)


    def forward(self, input_tensor):
        # collapse feature dimension
        input_inception = self.bottleneck(input_tensor.transpose(1,2)).transpose(1,2)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1)
        features = self.bn_relu(features)
        if self.residual:
            features = features + input_tensor
            features = self.residual_relu(features)
        return features

if __name__=="__main__":
    model = InceptionTime(input_dim=28, hidden_dims=32, num_classes=2, num_layers=2).to(torch.device("cpu"))
    #                N   T   D
    src = torch.rand(16, 28, 13)
    model(src)
    print()