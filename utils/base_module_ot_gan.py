#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import glu, interpolate
import torch
import math
import numpy as np

def cReLU(x):
    # x.ndim should be 4, i.e. batch_size * nc * h * w
    # cReLU doubles the number channels
    m = nn.ReLU()
    return torch.cat((m(x), m(-x)), dim=1)

def normalize_conv_weight(conv):
    # conv weight is a tensor of size "out_channels * in_channels * h * w"
    # normalize the weight of conv along dim = [1, 2, 3], i.e. in_channels * h * w.
    for param in conv.parameters():
        x = x.view([-1, np.prod(x.shape[1:])])

        param /= torch.norm(param)
class Encoder(nn.Module):
    def __init__(self, image_size, nc, bias=False):
        super(Encoder, self).__init__()
        # out_channels is 128*2=256 since we will use crelu as activation
        self.conv2d_3_to_256_k5_s1 = weight_norm(nn.Conv2d(in_channels=nc,
                                               out_channels=128,
                                               kernel_size=5,
                                               stride=1,
                                               padding=2,
                                               bias=bias)
                                                 )
        # out_channels is 256*2=512 since we will use crelu as activation
        self.conv2d_256_to_512_k5_s2 = weight_norm(nn.Conv2d(in_channels=256,
                                                 out_channels=256,
                                                 kernel_size=5,
                                                 stride=2,
                                                 padding=2,
                                                 bias=bias)
                                                   )
        # out_channels is 512*2=1024 since we will use crelu as activation
        self.conv2d_512_to_1024_k5_s2 = weight_norm(nn.Conv2d(in_channels=512,
                                                  out_channels=512,
                                                  kernel_size=5,
                                                  stride=2,
                                                  padding=2,
                                                  bias=bias)
                                                    )
        # out_channels is 1024*2=2048 since we will use crelu as activation
        self.conv2d_1024_to_2048_k5_s2 = weight_norm(nn.Conv2d(in_channels=1024,
                                                   out_channels=1024,
                                                   kernel_size=5,
                                                   stride=2,
                                                   padding=2,
                                                   bias=bias)
                                                     )

    def forward(self, x):
        # x.ndim should be 4, i.e. batch_size * nc * h * w
        assert(x.ndim == 4)
        x = self.conv2d_3_to_256_k5_s1(x)
        x = cReLU(x)
        x = self.conv2d_256_to_512_k5_s2(x)
        x = cReLU(x)
        x = self.conv2d_512_to_1024_k5_s2(x)
        x = cReLU(x)
        x = self.conv2d_1024_to_2048_k5_s2(x)
        x = cReLU(x)
        # reshape x as batch_size * (2048*4*4)
        x = x.view([-1, np.prod(x.shape[1:])])
        # normalize x such that x[i, :] has norm 1
        x = x/torch.norm(x, dim=1).unsqueeze(dim=1)
        return x


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert math.log2(isize).is_integer() and isize > 4, "isize has to be a power of 2 & isize > 4"
        self.n_layers = int(math.log2(isize//4) + 1) # size: 4,4|8,8|...|isize,isize
        self.k = k
        self.n_channels_init = ngf * 2 ** (self.n_layers - 1)# n_channels: ngf|ngf*2|...|ngf*2^(n_layers-1)
        self.n_linear_featurs = 2*4*4*self.n_channels_init
        self.linear = nn.Linear(self.k, self.n_linear_featurs)
        self.conv2ds = nn.ModuleList()
        for layer in range(self.n_layers):
            n_conv2d_featues_layer = self.n_channels_init//2**layer
            self.conv2ds.append(nn.Conv2d(n_conv2d_featues_layer, n_conv2d_featues_layer, kernel_size=5, stride=1, padding=2))
        self.conv2d_ngf_to_nc = nn.Conv2d(ngf, nc, kernel_size=5, stride=1, padding=2)

        # self.conv2d_1024_to_512 = nn.Conv2d(512, 256*2, kernel_size=5, stride=1, padding=2)
        # self.conv2d_512_to_256 = nn.Conv2d(256, 128*2, kernel_size=5, stride=1, padding=2)
        # self.conv2d_256_to_128 = nn.Conv2d(128, 64*2, kernel_size=5, stride=1, padding=2)
        # self.conv2d_128_to_nc = nn.Conv2d(64, nc, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.linear(x.view([-1, self.k]))
        x = glu(x, dim=1)
        x = x.view([-1, self.n_channels_init, 4, 4])
        for layer, conv2d in zip(range(self.n_layers), self.conv2ds):
            isize = 4*2**layer
            x = interpolate(x, size = [isize, isize])
            x = conv2d(x)
            # last layer have no glu activation
            if layer + 1 < self.n_layers:
                x = glu(x, dim=1)
        x = self.conv2d_ngf_to_nc(x)
        x = torch.tanh(x)
        return x

        # x = x.view([-1, 512, 4, 4])
        # x = interpolate(x, size=[8, 8])
        # x = self.conv2d_1024_to_512(x)
        # x = glu(x, dim=1)
        # x = interpolate(x, size=[16, 16])
        # x = self.conv2d_512_to_256(x)
        # x = glu(x, dim=1)
        # x = interpolate(x, size=[32, 32])
        # x = self.conv2d_256_to_128(x)
        # x = glu(x, dim=1)
        # x = self.conv2d_128_to_nc(x)
        # x = torch.tanh(x)
        #
        # return x


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    for param in m.parameters():
        param.data.normal_(0, 0.01)


def _squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy


class GroundCost(nn.Module):
    def __init__(self, encoder, shape):
        super(GroundCost, self).__init__()
        self.φ = encoder
        self.shape = shape

    def forward(self, x, y):
        φ_x = self.φ(x.view(self.shape)).squeeze()
        φ_y = self.φ(y.view(self.shape)).squeeze()
        cost = _squared_distances(φ_x, φ_y)/2
        return cost
