import torch.nn as nn
import torch.nn.functional as F


def compute_output_size(i, K, P, S):
    output_size = ((i - K + 2*P)/S) + 1
    return int(output_size)


class CNN_2D(nn.Module):
    def __init__(self, param):
        super(CNN_2D, self).__init__()

        self.embedding = nn.ModuleList()

        for i in range(param.n_conv):
            kernel = param.kernels[i]
            pooling = param.pooling[i]
            pad = int((kernel - 1) / 2)

            layers = [
                nn.Conv2d(
                    in_channels=param.in_channels[i],
                    out_channels=param.out_channels[i],
                    kernel_size=kernel,
                    stride=1,
                    padding=pad,
                    bias=False
                ),
                nn.BatchNorm2d(param.out_channels[i]),
                nn.ReLU(inplace=True)
            ]

            if pooling != 0:
                layers.append(
                    nn.MaxPool2d(kernel_size=pooling, stride=pooling)
                )

            self.embedding.append(nn.Sequential(*layers))

        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=param.dropout)

        self.f = nn.ModuleList()
        for i in range(len(param.fweights) - 1):
            self.f.append(
                nn.Linear(param.fweights[i], param.fweights[i + 1])
            )

    def forward(self, x, return_conv=False):
        out = self.embedding[0](x)

        if return_conv:
            all_layers = [out]

        for i in range(1, len(self.embedding)):
            out = self.embedding[i](out)

            if return_conv:
                all_layers.append(out)

        out = out.view(out.size(0), -1)

        for fc in self.f[:-1]:
            out = fc(out)
            out = self.ReLU(out)
            out = self.Dropout(out)

        out = self.f[-1](out)

        if return_conv:
            return F.softmax(out, dim=1), all_layers
        else:
            return F.softmax(out, dim=1)


class CNN_8CL_2D(object):
    def __init__(self, input_dim=128):
        self.input_dim = [input_dim, input_dim]

        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [1] + self.out_channels[:-1]
        self.n_conv = len(self.out_channels)

        self.kernels = [3] * self.n_conv
        self.pooling = [4, 0, 3, 0, 2, 0, 2, 0]

        for i in range(self.n_conv):
            if self.pooling[i] != 0:
                for d in range(2):
                    self.input_dim[d] = compute_output_size(
                        self.input_dim[d],
                        self.pooling[i],
                        0,
                        self.pooling[i]
                    )

        out = self.input_dim[0] * self.input_dim[1]

        self.fweights = [
            self.out_channels[-1] * out,
            3
        ]

        self.dropout = 0.0
