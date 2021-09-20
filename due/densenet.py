'''
Pytorch impplementation of DenseNet.
Reference:
[1] Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks.
    arXiv preprint arXiv:1608.06993, 2016a.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from due.layers import spectral_norm_conv, spectral_norm_fc


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, inp_size, wrapped_conv, activation):
        super(Bottleneck, self).__init__()
        self.activation = activation
        self.wrapped_conv = wrapped_conv
        self.inp_size = inp_size
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = wrapped_conv(self.inp_size, in_planes, 4*growth_rate, kernel_size=1, stride=1)
        # self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = wrapped_conv(self.inp_size, 4*growth_rate, growth_rate, kernel_size=3, stride=1)
        # self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, inp_size, wrapped_conv, activation):
        super(Transition, self).__init__()
        self.activation = activation
        self.wrapped_conv = wrapped_conv
        self.inp_size = inp_size
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = wrapped_conv(self.inp_size, in_planes, out_planes, kernel_size=1, stride=1)
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        block,
        nblocks,
        growth_rate=12,
        reduction=0.5,
        num_classes=None,
        temp=1.0,
        spectral_normalization=True,
        mod=False,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.temp = temp

        self.mod = mod
        self.num_classes = num_classes

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(conv, coeff, shapes, n_power_iterations)

            return wrapped_conv

        self.wrapped_conv = wrapped_conv
        self.activation = F.leaky_relu if self.mod else F.relu

        num_planes = 2*growth_rate
        self.conv1 = wrapped_conv(32, 3, num_planes, kernel_size=3, stride=1)
        # self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], 32)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, 32, self.wrapped_conv, self.activation)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], 16)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, 16, self.wrapped_conv, self.activation)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], 8)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, 8, self.wrapped_conv, self.activation)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], 4)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

        if self.num_classes is not None:
            self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, inp_size):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, inp_size, self.wrapped_conv, self.activation))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.activation(self.bn(out)), 4)
        out = out.view(out.size(0), -1)

        if self.num_classes is not None:
            out = self.linear(out) / self.temp
        return out



def densenet121(spectral_normalization=True, mod=False, temp=1.0, mnist=False, **kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)


def densenet169(spectral_normalization=True, mod=False, temp=1.0, mnist=False, **kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)


def densenet201(spectral_normalization=True, mod=False, temp=1.0, mnist=False, **kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)


def densenet161(spectral_normalization=True, mod=False, temp=1.0, mnist=False, **kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, spectral_normalization=spectral_normalization, mod=mod, temp=temp, **kwargs)
