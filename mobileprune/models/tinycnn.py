import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
import numpy as np
from torch.nn.modules.utils import _pair


class RoIFAN65(nn.Module): #
    def __init__(self, n_classes, alpha=1.0):
        """
        first stage:
        both stage:
        Args:
            num_coords : final prediction of landmarks [numLandmarks]
        """
        super(RoIFAN65, self).__init__()
        self.crop_size = 7

        self.get_back_bone = nn.Sequential(
            nn.Conv2d(3,  int(32 * alpha), kernel_size=3, stride=2, bias=False),  # 224 -> 112
            DepthwiseBlockPytorchFirst(int(32 * alpha), int(16 * alpha), 1),
            DepthwiseBlockPytorch(int(16 * alpha), int(24 * alpha), 2),  # 112 -> 56
            DepthwiseBlockPytorch(int(24 * alpha), int(24 * alpha)),
        )
        self.get_point_masker = nn.Sequential(
            DepthwiseBlockPytorch(int(24 * alpha), int(32 * alpha), 2),  # 56 -> 28
            DepthwiseBlockPytorch(int(32 * alpha), int(32 * alpha)),
            DepthwiseBlockPytorch(int(32 * alpha), int(32 * alpha)),
            DepthwiseBlockPytorch(int(32 * alpha), int(64 * alpha), 2),  # 28 -> 14
            DepthwiseBlockPytorch(int(64 * alpha), int(64 * alpha)),
            # DepthwiseBlockPytorch(int(64 * alpha), int(64 * alpha)),
            DepthwiseBlockPytorch(int(64 * alpha), int(64 * alpha)),
            DepthwiseBlockPytorch(int(64 * alpha), int(96 * alpha)),
            # layers.DepthwiseBlockPytorch(int(96 * alpha), int(96 * alpha)),
            DepthwiseBlockPytorch(int(96 * alpha), 65)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(65, n_classes)
        )

    def forward(self, x):
        x = self.get_back_bone(x)
        feats = self.get_point_masker(x)  # L, 14, 14
        feats = F.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)

        return preds


class DepthwiseBlockPytorch(nn.Module):
    """
        Depthwise conv + Pointwise conv
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()

        self.skip_connection = (in_channels == out_channels and stride == 1)

        self.conv_1 = nn.Conv2d(in_channels, in_channels * expansion, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(in_channels * expansion)

        self.conv_2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, 3,
                                stride=stride, padding=1, bias=False, groups=in_channels * expansion)
        self.bn_2 = nn.BatchNorm2d(in_channels * expansion)

        self.conv_3 = nn.Conv2d(in_channels * expansion, out_channels, 1, bias=False)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.bn_3(x)

        if self.skip_connection:
            x = x + input
        return x


class DepthwiseBlockPytorchFirst(nn.Module):
    """
        Depthwise conv + Pointwise conv
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3,
                                stride=stride, padding=1, bias=False, groups=out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        return x


class RoIFAN62(nn.Module): #
    def __init__(self, n_classes):
        """
        first stage:
        both stage:
        Args:
            num_coords : final prediction of landmarks [numLandmarks]
        """
        super(RoIFAN62, self).__init__()
        self.crop_size = 4
        self.n_classes = n_classes
        self.get_back_bone = nn.Sequential(
            Conv2dSamePadding(3, 8, kernel_size=3, stride=2, bias=False),  # 128 -> 64
            InvertedResidual(8, 8, expand_ratio=1),
            InvertedResidual(8, 8, down_sample=True),  # 64 -> 32
        )
        self.get_point_masker = nn.Sequential(
            InvertedResidual(8, 16, down_sample=True),  # 32 -> 16
            InvertedResidual(16, 32, down_sample=True),  # 16 -> 8
            InvertedResidual(32, 32),
            InvertedResidual(32, 62)
        )


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(62, n_classes)
        )

    def forward(self, x):
        x = self.get_back_bone(x)
        feats = self.get_point_masker(x)  # L, 14, 14
        feats = F.adaptive_avg_pool2d(feats, 1)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)

        return preds


class Conv2dSamePadding(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    """
    Do conv2d with padding same
    """
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    input_cols = input.size(3)
    filter_cols = weight.size(3)
    out_cols = (input_cols + stride[0] - 1) // stride[0]
    padding_cols = max(0, (out_cols - 1) * stride[0] + (filter_cols - 1) * dilation[0] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    # if rows_odd or cols_odd:
    #     input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    padding_rows = int(np.ceil(padding_rows / 2))
    padding_cols = int(np.ceil(padding_cols / 2))
    output = F.conv2d(input, weight, bias, stride, padding=(padding_rows, padding_cols),
                    dilation=dilation, groups=groups)
    return output


class InvertedResidual(nn.Module):
    # using expand_ratio = 6 as default as the MobileNetV2 described
    # using Relu instead of Relu6 cause we are not using float16
    def __init__(self, inp, oup, expand_ratio=5, down_sample=False, dilation_rate=1):
        super(InvertedResidual, self).__init__()
        stride = 2 if down_sample else 1

        self.use_connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            Conv2dSamePadding(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio,
                              bias=False, dilation=dilation_rate),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PredictBlock(nn.Module):
    """
        only the corresponding channels
        Depthwise conv + Pointwise conv
    """
    def __init__(self, inp, oup, group_size):
        super(PredictBlock, self).__init__()
        self.conv_dw = nn.Sequential(
                Conv2dSamePadding(inp, inp, kernel_size=3, stride=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                Conv2dSamePadding(inp, oup, 1, 1, groups=group_size, bias=False),
                nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        x = self.conv_dw(x)
        return x
