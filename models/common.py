# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync

from torch.nn import init
from torch.nn.parameter import Parameter

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


# -------------------   GSConv   -------------------
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__(c1, c2, k=1, s=1, g=1, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU(self.shuf(x2))


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)

# -------------------   GSConv   -------------------



# https://arxiv.org/pdf/2102.00240.pdf
class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out
# -------------shuffleAttention--------------------


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

# CBAM
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)

# CBAM
class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


#CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
#CA
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

#CA
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1,c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # print(y.shape,y.squeeze(-1).shape,y.squeeze(-1).transpose(-1, -2).shape)
        # Two different branches of ECA module
        # 50*C*1*1
        #50*C*1
        #50*1*C
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class SEBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.se=SE(c1,c2,ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        b, c, _, _ = x.size()
        y = self.avgpool(x1).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        out = x1 * y.expand_as(x1)

        # out=self.se(x1)*x1
        return x + out if self.add else out


class SE(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SE, self).__init__()
        #c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class ECABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16, k_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.eca=ECA(c1,c2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        # out=self.eca(x1)*x1
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x1 * y.expand_as(x1)

        return x + out if self.add else out


class C3ECA(C3):
    # C3 module with ECABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ECABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))




# C3CA
class CABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.ca=CoordAtt(c1,c2,ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // ratio)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x1)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x1 * a_w * a_h

        # out=self.ca(x1)*x1
        return x + out if self.add else out

# C3CA
class C3CA(C3):
    # C3 module with CABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,ratio=16,kernel_size=7):  # ch_in, ch_out, shortcut, groups, expansion
        super(CBAMBottleneck,self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        #self.cbam=CBAM(c1,c2,ratio,kernel_size)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        out = self.channel_attention(x1) * x1
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return x + out if self.add else out


class C3CBAM(C3):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3SE(C3):
    # C3 module with SEBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SEBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '8.0.0', verbose=True)  # version requirement
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.engine or self.onnx:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# C2f
class v8_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(v8_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



# S2-MLPv2 芒果哥
# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x



#  ------------------  迪导改版 ---------------------

# ---------------------------SE Begin---------------------------
# class SE(nn.Module):
#     def __init__(self, c1, ratio=16):
#         super(SE, self).__init__()
#         # c*1*1
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avgpool(x).view(b, c)
#         y = self.l1(y)
#         y = self.relu(y)
#         y = self.l2(y)
#         y = self.sig(y)
#         y = y.view(b, c, 1, 1)
#         return x * y.expand_as(x)


# ---------------------------SE End---------------------------
#
#
# # ---------------------------CBAM Begin---------------------------
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
#         max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
#         out = self.sigmoid(avg_out + max_out)
#         return out
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 1*h*w
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         # 2*h*w
#         x = self.conv(x)
#         # 1*h*w
#         return self.sigmoid(x)
#
#
# class CBAM(nn.Module):
#     def __init__(self, c1, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(c1, ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         # c*h*w
#         # c*h*w * 1*h*w
#         out = self.spatial_attention(out) * out
#         return out
#
#
# # ---------------------------CBAM End---------------------------
#
#
# # ---------------------------ECA Begin---------------------------
# class ECA(nn.Module):
#
#     def __init__(self, k_size=3):
#         super(ECA, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)
#
#
# # ---------------------------ECA End---------------------------
#
#
# # ---------------------------CA Begin---------------------------
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.size()
#         # c*1*W
#         x_h = self.pool_h(x)
#         # c*H*1
#         # C*1*h
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#         y = torch.cat([x_h, x_w], dim=2)
#         # C*1*(h+w)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         out = identity * a_w * a_h
#         return out
#
#
# # ---------------------------CA End---------------------------
#
#
# # ---------------------------SimAM Begin---------------------------
# class SimAM(torch.nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)
#
#
# # ---------------------------SimAM End---------------------------
#
#
# # ---------------------------S2-MLPv2 Begin---------------------------
# def spatial_shift1(x):
#     b, w, h, c = x.size()
#     x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
#     x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
#     x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
#     x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
#     return x
#
#
# def spatial_shift2(x):
#     b, w, h, c = x.size()
#     x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
#     x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
#     x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
#     x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
#     return x
#
#
# class SplitAttention(nn.Module):
#     def __init__(self, channel=512, k=3):
#         super().__init__()
#         self.channel = channel
#         self.k = k
#         self.mlp1 = nn.Linear(channel, channel, bias=False)
#         self.gelu = nn.GELU()
#         self.mlp2 = nn.Linear(channel, channel * k, bias=False)
#         self.softmax = nn.Softmax(1)
#
#     def forward(self, x_all):
#         b, k, h, w, c = x_all.shape
#         x_all = x_all.reshape(b, k, -1, c)
#         a = torch.sum(torch.sum(x_all, 1), 1)
#         hat_a = self.mlp2(self.gelu(self.mlp1(a)))
#         hat_a = hat_a.reshape(b, self.k, c)
#         bar_a = self.softmax(hat_a)
#         attention = bar_a.unsqueeze(-2)
#         out = attention * x_all
#         out = torch.sum(out, 1).reshape(b, h, w, c)
#         return out
#
#
# class S2Attention(nn.Module):
#
#     def __init__(self, channels=512):
#         super().__init__()
#         self.mlp1 = nn.Linear(channels, channels * 3)
#         self.mlp2 = nn.Linear(channels, channels)
#         self.split_attention = SplitAttention()
#
#     def forward(self, x):
#         b, c, w, h = x.size()
#         x = x.permute(0, 2, 3, 1)
#         x = self.mlp1(x)
#         x1 = spatial_shift1(x[:, :, :, :c])
#         x2 = spatial_shift2(x[:, :, :, c:c * 2])
#         x3 = x[:, :, :, c * 2:]
#         x_all = torch.stack([x1, x2, x3], 1)
#         a = self.split_attention(x_all)
#         x = self.mlp2(a)
#         x = x.permute(0, 3, 1, 2)
#         return x
#
#
# # ---------------------------S2-MLPv2 End---------------------------
#
#
# # ---------------------------NAMAttention Begin---------------------------
# class Channel_Att(nn.Module):
#     def __init__(self, channels):
#         super(Channel_Att, self).__init__()
#         self.channels = channels
#
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#
#     def forward(self, x):
#         residual = x
#
#         x = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#
#         x = torch.sigmoid(x) * residual  #
#
#         return x
#
#
# class NAMAttention(nn.Module):
#     def __init__(self, channels):
#         super(NAMAttention, self).__init__()
#         self.Channel_Att = Channel_Att(channels)
#
#     def forward(self, x):
#         x_out1 = self.Channel_Att(x)
#
#         return x_out1
#
#
# # ---------------------------NAMAttention End---------------------------
#
#
# # ---------------------------Criss-CrossAttention Begin---------------------------
# from torch.nn import Softmax
#
#
# def INF(B, H, W):
#     return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
#
#
# class CrissCrossAttention(nn.Module):
#     """ Criss-Cross Attention Module"""
#
#     def __init__(self, in_dim):
#         super(CrissCrossAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.softmax = Softmax(dim=3)
#         self.INF = INF
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         m_batchsize, _, height, width = x.size()
#         proj_query = self.query_conv(x)
#         proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
#                                                                                                                  1)
#         proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
#                                                                                                                  1)
#         proj_key = self.key_conv(x)
#         proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#         proj_value = self.value_conv(x)
#         proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#         energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
#                                                                                                      height,
#                                                                                                      height).permute(0,
#                                                                                                                      2,
#                                                                                                                      1,
#                                                                                                                      3)
#         energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
#         concate = self.softmax(torch.cat([energy_H, energy_W], 3))
#
#         att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
#         # print(concate)
#         # print(att_H)
#         att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
#         out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
#         out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
#         # print(out_H.size(),out_W.size())
#         return self.gamma * (out_H + out_W) + x
#
#
# # ---------------------------Criss-CrossAttention End---------------------------
#
#
# # ---------------------------GAMAttention Begin---------------------------
# class GAMAttention(nn.Module):
#
#     def __init__(self, c1, c2, group=True, rate=4):
#         super(GAMAttention, self).__init__()
#
#         self.channel_attention = nn.Sequential(
#             nn.Linear(c1, int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(c1 / rate), c1)
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(int(c1 / rate)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
#                                                                                                      kernel_size=7,
#                                                                                                      padding=3),
#             nn.BatchNorm2d(c2)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
#         x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
#         x_channel_att = x_att_permute.permute(0, 3, 1, 2)
#         x = x * x_channel_att
#
#         x_spatial_att = self.spatial_attention(x).sigmoid()
#         x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
#         out = x * x_spatial_att
#         return out
#
#
# def channel_shuffle(x, groups=2):  ##shuffle channel
#     # RESHAPE----->transpose------->Flatten
#     B, C, H, W = x.size()
#     out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
#     out = out.view(B, C, H, W)
#     return out
#
#
# # ---------------------------GAMAttention End---------------------------
#
#
# # ---------------------------Selective Kernel Attention Begin-------------------------
# class SKAttention(nn.Module):
#
#     def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
#         super().__init__()
#         self.d = max(L, channel // reduction)
#         self.convs = nn.ModuleList([])
#         for k in kernels:
#             self.convs.append(
#                 nn.Sequential(OrderedDict([
#                     ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
#                     ('bn', nn.BatchNorm2d(channel)),
#                     ('relu', nn.ReLU())
#                 ]))
#             )
#         self.fc = nn.Linear(channel, self.d)
#         self.fcs = nn.ModuleList([])
#         for i in range(len(kernels)):
#             self.fcs.append(nn.Linear(self.d, channel))
#         self.softmax = nn.Softmax(dim=0)
#
#     def forward(self, x):
#         bs, c, _, _ = x.size()
#         conv_outs = []
#         ### split
#         for conv in self.convs:
#             conv_outs.append(conv(x))
#         feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w
#
#         ### fuse
#         U = sum(conv_outs)  # bs,c,h,w
#
#         ### reduction channel
#         S = U.mean(-1).mean(-1)  # bs,c
#         Z = self.fc(S)  # bs,d
#
#         ### calculate attention weight
#         weights = []
#         for fc in self.fcs:
#             weight = fc(Z)
#             weights.append(weight.view(bs, c, 1, 1))  # bs,channel
#         attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
#         attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1
#
#         ### fuse
#         V = (attention_weughts * feats).sum(0)
#         return V
#
#
# # ---------------------------Selective Kernel Attention End---------------------------
#
#
# # ---------------------------ShuffleAttention Begin---------------------------
# from torch.nn.parameter import Parameter
#
#
# class ShuffleAttention(nn.Module):
#
#     def __init__(self, channel=512, G=8):
#         super().__init__()
#         self.G = G
#         self.channel = channel
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
#         self.sigmoid = nn.Sigmoid()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)
#
#         # flatten
#         x = x.reshape(b, -1, h, w)
#
#         return x
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # group into subfeatures
#         x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w
#
#         # channel_split
#         x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w
#
#         # channel attention
#         x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
#         x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
#         x_channel = x_0 * self.sigmoid(x_channel)
#
#         # spatial attention
#         x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
#         x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
#         x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w
#
#         # concatenate along channel axis
#         out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
#         out = out.contiguous().view(b, -1, h, w)
#
#         # channel shuffle
#         out = self.channel_shuffle(out, 2)
#         return out
#
#
# # ---------------------------ShuffleAttention End---------------------------
#
#
# # ---------------------------A2-Net  Begin---------------------------
# class DoubleAttention(nn.Module):
#
#     def __init__(self, in_channels, c_m, c_n, reconstruct=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.reconstruct = reconstruct
#         self.c_m = c_m
#         self.c_n = c_n
#         self.convA = nn.Conv2d(in_channels, c_m, 1)
#         self.convB = nn.Conv2d(in_channels, c_n, 1)
#         self.convV = nn.Conv2d(in_channels, c_n, 1)
#         if self.reconstruct:
#             self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         assert c == self.in_channels
#         A = self.convA(x)  # b,c_m,h,w
#         B = self.convB(x)  # b,c_n,h,w
#         V = self.convV(x)  # b,c_n,h,w
#         tmpA = A.view(b, self.c_m, -1)
#         attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=1)
#         attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=1)
#         # step 1: feature gating
#         global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
#         # step 2: feature distribution
#         tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
#         tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
#         if self.reconstruct:
#             tmpZ = self.conv_reconstruct(tmpZ)
#
#         return tmpZ
#
#
# # ---------------------------A2-Net  End---------------------------
#
#
# # ---------------------------RFB  Begin---------------------------
# class BasicConv(nn.Module):
#
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         if bn:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                   dilation=dilation, groups=groups, bias=False)
#             self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             self.relu = nn.ReLU(inplace=True) if relu else None
#         else:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                   dilation=dilation, groups=groups, bias=True)
#             self.bn = None
#             self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class BasicRFB(nn.Module):
#
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
#         super(BasicRFB, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
#                       dilation=vision + 1, relu=False, groups=groups)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
#                       dilation=vision + 2, relu=False, groups=groups)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
#                       groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
#                       dilation=vision + 4, relu=False, groups=groups)
#         )
#
#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out
#
#
# # ---------------------------RFB  End---------------------------
#
#
# # ---------------------------CoTAttention Begin---------------------------
# class CoTAttention(nn.Module):
#
#     def __init__(self, dim=512, kernel_size=3):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#
#         self.key_embed = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU()
#         )
#         self.value_embed = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#
#         factor = 4
#         self.attention_embed = nn.Sequential(
#             nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
#             nn.BatchNorm2d(2 * dim // factor),
#             nn.ReLU(),
#             nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
#         )
#
#     def forward(self, x):
#         bs, c, h, w = x.shape
#         k1 = self.key_embed(x)  # bs,c,h,w
#         v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w
#
#         y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
#         att = self.attention_embed(y)  # bs,c*k*k,h,w
#         att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
#         att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
#         k2 = F.softmax(att, dim=-1) * v
#         k2 = k2.view(bs, c, h, w)
#
#         return k1 + k2
#
#
# # ---------------------------CoTAttention End---------------------------
#
#
# # ---------------------------EffectiveSEModule Begin---------------------------
# from timm.models.layers.create_act import create_act_layer
#
#
# class EffectiveSEModule(nn.Module):
#     def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid'):
#         super(EffectiveSEModule, self).__init__()
#         self.add_maxpool = add_maxpool
#         self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.gate = create_act_layer(gate_layer)
#
#     def forward(self, x):
#         x_se = x.mean((2, 3), keepdim=True)
#         if self.add_maxpool:
#             # experimental codepath, may remove or change
#             x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
#         x_se = self.fc(x_se)
#         return x * self.gate(x_se)
#
#
# # ---------------------------EffectiveSEModule End---------------------------
#
#
# # ---------------------------GlobalContext Begin---------------------------
# from timm.models.layers.norm import LayerNorm2d
#
#
# class GlobalContext(nn.Module):
#
#     def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
#                  rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
#         super(GlobalContext, self).__init__()
#         act_layer = get_act_layer(act_layer)
#
#         self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
#
#         if rd_channels is None:
#             rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
#         if fuse_add:
#             self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
#         else:
#             self.mlp_add = None
#         if fuse_scale:
#             self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
#         else:
#             self.mlp_scale = None
#
#         self.gate = create_act_layer(gate_layer)
#         self.init_last_zero = init_last_zero
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.conv_attn is not None:
#             nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
#         if self.mlp_add is not None:
#             nn.init.zeros_(self.mlp_add.fc2.weight)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         if self.conv_attn is not None:
#             attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
#             attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
#             context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
#             context = context.view(B, C, 1, 1)
#         else:
#             context = x.mean(dim=(2, 3), keepdim=True)
#
#         if self.mlp_scale is not None:
#             mlp_x = self.mlp_scale(context)
#             x = x * self.gate(mlp_x)
#         if self.mlp_add is not None:
#             mlp_x = self.mlp_add(context)
#             x = x + mlp_x
#
#         return x
#
#
# # ---------------------------GlobalContext End---------------------------
#
# ---------------------------GatherExcite Begin---------------------------
# from timm.models.layers.create_act import create_act_layer, get_act_layer
# from timm.models.layers.create_conv2d import create_conv2d
# from timm.models.layers.helpers import make_divisible
# from timm.models.layers.mlp import ConvMlp
# timm版本更新后，脚本的位置发生了变化。上面导包改为以下格式

from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp
from torch.nn import functional as F


class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1. / 16, rd_channels=None, rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


# ---------------------------GatherExcite End---------------------------
#
#
# # ---------------------------MHSA Begin---------------------------
# class MHSA(nn.Module):
#     def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
#         super(MHSA, self).__init__()
#
#         self.heads = heads
#         self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
#         self.pos = pos_emb
#         if self.pos:
#             self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
#                                              requires_grad=True)
#             self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
#                                              requires_grad=True)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         n_batch, C, width, height = x.size()
#         q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
#         k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
#         v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
#         content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
#         c1, c2, c3, c4 = content_content.size()
#         if self.pos:
#             content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
#                 0, 1, 3, 2)  # 1,4,1024,64
#
#             content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
#             content_position = content_position if (
#                     content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
#             assert (content_content.shape == content_position.shape)
#             energy = content_content + content_position
#         else:
#             energy = content_content
#         attention = self.softmax(energy)
#         out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
#         out = out.view(n_batch, C, width, height)
#         return out
#
#
# # ---------------------------MHSA End---------------------------
#
#
# # ---------------------------ParNetAttention Begin---------------------------
# class ParNetAttention(nn.Module):
#
#     def __init__(self, channel=512):
#         super().__init__()
#         self.sse = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel, channel, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=1),
#             nn.BatchNorm2d(channel)
#         )
#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel)
#         )
#         self.silu = nn.SiLU()
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         x1 = self.conv1x1(x)
#         x2 = self.conv3x3(x)
#         x3 = self.sse(x) * x
#         y = self.silu(x1 + x2 + x3)
#         return y
#
#
# # ---------------------------ParNetAttention End---------------------------
#
#
# # ---------------------------ParallelPolarizedSelfAttention Begin---------------------------
# class ParallelPolarizedSelfAttention(nn.Module):
#
#     def __init__(self, channel=512):
#         super().__init__()
#         self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
#         self.softmax_channel = nn.Softmax(1)
#         self.softmax_spatial = nn.Softmax(-1)
#         self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
#         self.ln = nn.LayerNorm(channel)
#         self.sigmoid = nn.Sigmoid()
#         self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # Channel-only Self-Attention
#         channel_wv = self.ch_wv(x)  # bs,c//2,h,w
#         channel_wq = self.ch_wq(x)  # bs,1,h,w
#         channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#         channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
#         channel_wq = self.softmax_channel(channel_wq)
#         channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
#         channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
#                                                                                                                  1).reshape(
#             b, c, 1, 1)  # bs,c,1,1
#         channel_out = channel_weight * x
#
#         # Spatial-only Self-Attention
#         spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
#         spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
#         spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
#         spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#         spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
#         spatial_wq = self.softmax_spatial(spatial_wq)
#         spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
#         spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
#         spatial_out = spatial_weight * x
#         out = spatial_out + channel_out
#         return out
#
#
# # ---------------------------ParallelPolarizedSelfAttention End---------------------------
#
# # ---------------------------SpatialGroupEnhance Begin---------------------------
# class SpatialGroupEnhance(nn.Module):
#     def __init__(self, groups=8):
#         super().__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
#         self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
#         self.sig = nn.Sigmoid()
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
#         xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
#         xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
#         t = xn.view(b * self.groups, -1)  # bs*g,h*w
#
#         t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
#         std = t.std(dim=1, keepdim=True) + 1e-5
#         t = t / std  # bs*g,h*w
#         t = t.view(b, self.groups, h, w)  # bs,g,h*w
#
#         t = t * self.weight + self.bias  # bs,g,h*w
#         t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
#         x = x * self.sig(t)
#         x = x.view(b, c, h, w)
#         return x
#
#
# # ---------------------------SpatialGroupEnhance End---------------------------
#
#
# # ---------------------------SequentialPolarizedSelfAttention Begin---------------------------
# class SequentialPolarizedSelfAttention(nn.Module):
#
#     def __init__(self, channel=512):
#         super().__init__()
#         self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
#         self.softmax_channel = nn.Softmax(1)
#         self.softmax_spatial = nn.Softmax(-1)
#         self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
#         self.ln = nn.LayerNorm(channel)
#         self.sigmoid = nn.Sigmoid()
#         self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # Channel-only Self-Attention
#         channel_wv = self.ch_wv(x)  # bs,c//2,h,w
#         channel_wq = self.ch_wq(x)  # bs,1,h,w
#         channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#         channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
#         channel_wq = self.softmax_channel(channel_wq)
#         channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
#         channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
#                                                                                                                  1).reshape(
#             b, c, 1, 1)  # bs,c,1,1
#         channel_out = channel_weight * x
#
#         # Spatial-only Self-Attention
#         spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
#         spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
#         spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
#         spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#         spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
#         spatial_wq = self.softmax_spatial(spatial_wq)
#         spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
#         spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
#         spatial_out = spatial_weight * channel_out
#         return spatial_out
#
#
# # ---------------------------SequentialPolarizedSelfAttention End---------------------------
#
#
# # ---------------------------TripletAttention Begin---------------------------
# class BasicConv_T(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv_T, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#
#
# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv_T(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
#
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale
#
#
# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.hw = AttentionGate()
#
#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_out = self.hw(x)
#             x_out = 1 / 3 * (x_out + x_out11 + x_out21)
#         else:
#             x_out = 1 / 2 * (x_out11 + x_out21)
#         return x_out
#
# # ---------------------------TripletAttention End---------------------------









