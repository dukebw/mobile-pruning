# Copyright 2018 Brendan Duke.
#
# This file is part of Mobile Prune.
#
# Mobile Prune is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Mobile Prune is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Mobile Prune. If not, see <http://www.gnu.org/licenses/>.

"""From
https://github.com/Randl/MobileNetV2-pytorch/blob/3518846c69971c10cae89b6b29497a502200da65/model.py
"""
from collections import OrderedDict

import torch
from torch.nn import init

from .ib_layers import InformationBottleneck


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(torch.nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 t=6,
                 activation=torch.nn.ReLU6,
                 grp_fact=1):
        torch.nn.Module.__init__(self)

        self.conv1 = torch.nn.Conv2d(inplanes,
                                     inplanes * t,
                                     kernel_size=1,
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(inplanes * t)

        groups = max(inplanes * t // grp_fact, 1)
        self.conv2 = torch.nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=1,
                                     bias=False,
                                     groups=groups)
        self.bn2 = torch.nn.BatchNorm2d(inplanes * t)

        self.conv3 = torch.nn.Conv2d(inplanes * t,
                                     outplanes,
                                     kernel_size=1,
                                     bias=False)
        self.bn3 = torch.nn.BatchNorm2d(outplanes)

        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNetV2(torch.nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self,
                 scale=1.0,
                 input_size=224,
                 t=6,
                 in_channels=3,
                 num_classes=1000,
                 activation=torch.nn.ReLU6,
                 grp_fact=1):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        torch.nn.Module.__init__(self)

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.grp_fact = grp_fact
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8)
                  for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     self.c[0],
                                     kernel_size=3,
                                     bias=False,
                                     stride=self.s[0],
                                     padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        if self.scale <= 1:
            self.last_conv_out_ch = 1280
        else:
            self.last_conv_out_ch = _make_divisible(1280 * self.scale, 8)
        self.conv_last = torch.nn.Conv2d(self.c[-1],
                                         self.last_conv_out_ch,
                                         kernel_size=1,
                                         bias=False)
        self.bn_last = torch.nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(p=0.2, inplace=True)
        self.fc = torch.nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes,
                                        outplanes=outplanes,
                                        stride=stride,
                                        t=t,
                                        activation=self.activation_type,
                                        grp_fact=self.grp_fact)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes,
                                      outplanes=outplanes,
                                      stride=1,
                                      t=6,
                                      activation=self.activation_type,
                                      grp_fact=self.grp_fact)
            modules[name] = module

        return torch.nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0],
                                       outplanes=self.c[1],
                                       n=self.n[1],
                                       stride=self.s[1],
                                       t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i],
                                      outplanes=self.c[i + 1],
                                      n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return torch.nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LinearBottleneckIB(torch.nn.Module):
    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 t=6,
                 activation=torch.nn.ReLU6):
        torch.nn.Module.__init__(self)

        self.conv1 = torch.nn.Conv2d(inplanes,
                                     inplanes * t,
                                     kernel_size=1,
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(inplanes * t)
        self.ib1 = InformationBottleneck(
            inplanes * t,
            mask_thresh=self.threshold,
            init_mag=self.init_mag,
            init_var=self.init_var,
            kl_mult=self.kl_mult_base,
            sample_in_training=self.sample_in_training,
            sample_in_testing=self.sample_in_testing)

        self.conv2 = torch.nn.Conv2d(inplanes * t,
                                     inplanes * t,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=1,
                                     bias=False,
                                     groups=inplanes * t)
        self.bn2 = torch.nn.BatchNorm2d(inplanes * t)
        self.ib2 = InformationBottleneck(
            inplanes * t,
            mask_thresh=self.threshold,
            init_mag=self.init_mag,
            init_var=self.init_var,
            kl_mult=self.kl_mult_base,
            sample_in_training=self.sample_in_training,
            sample_in_testing=self.sample_in_testing)

        self.conv3 = torch.nn.Conv2d(inplanes * t,
                                     outplanes,
                                     kernel_size=1,
                                     bias=False)
        self.bn3 = torch.nn.BatchNorm2d(outplanes)
        self.ib3 = InformationBottleneck(
            outplanes,
            mask_thresh=self.threshold,
            init_mag=self.init_mag,
            init_var=self.init_var,
            kl_mult=self.kl_mult_base,
            sample_in_training=self.sample_in_training,
            sample_in_testing=self.sample_in_testing)

        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kl_list = [self.ib1, self.ib2, self.ib3]

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.ib1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.ib2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ib3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNetV2IB(torch.nn.Module):

    def __init__(self,
                 scale=1.0,
                 input_size=224,
                 t=6,
                 in_channels=3,
                 num_classes=1000,
                 activation=torch.nn.ReLU6):
        torch.nn.Module.__init__(self)

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.init_var = 0.01
        self.threshold = 0
        self.sample_in_training = True
        self.sample_in_testing = False
        self.kl_mult_base = 1.0/32

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8)
                  for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     self.c[0],
                                     kernel_size=3,
                                     bias=False,
                                     stride=self.s[0],
                                     padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.c[0])
        self.ib1 = InformationBottleneck(
            self.c[0],
            mask_thresh=self.threshold,
            init_mag=self.init_mag,
            init_var=self.init_var,
            kl_mult=self.kl_mult_base,
            sample_in_training=self.sample_in_training,
            sample_in_testing=self.sample_in_testing)
        self.bottlenecks, bottleneck_kl_list = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        if self.scale <= 1:
            self.last_conv_out_ch = 1280
        else:
            self.last_conv_out_ch = _make_divisible(1280 * self.scale, 8)
        self.conv_last = torch.nn.Conv2d(self.c[-1],
                                         self.last_conv_out_ch,
                                         kernel_size=1,
                                         bias=False)
        self.bn_last = torch.nn.BatchNorm2d(self.last_conv_out_ch)
        self.ib_last = InformationBottleneck(
            self.last_conv_out_ch,
            mask_thresh=self.threshold,
            init_mag=self.init_mag,
            init_var=self.init_var,
            kl_mult=self.kl_mult_base,
            sample_in_training=self.sample_in_training,
            sample_in_testing=self.sample_in_testing)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(p=0.2, inplace=True)
        self.fc = torch.nn.Linear(self.last_conv_out_ch, self.num_classes)

        self.kl_list = [self.ib1] + bottleneck_kl_list + [self.ib_last]

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneckIB(inplanes=inplanes,
                                          outplanes=outplanes,
                                          stride=stride,
                                          t=t,
                                          activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneckIB depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneckIB(inplanes=outplanes,
                                        outplanes=outplanes,
                                        stride=1,
                                        t=6,
                                        activation=self.activation_type)
            modules[name] = module

        return torch.nn.Sequential(modules), [m.kl_list for m in modules]

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        kl_list = []
        # First module is the only one with t=1
        bottleneck1, kls = self._make_stage(inplanes=self.c[0],
                                            outplanes=self.c[1],
                                            n=self.n[1],
                                            stride=self.s[1],
                                            t=1,
                                            stage=0)
        modules[stage_name + "_0"] = bottleneck1
        kl_list.append(kls)

        # add more LinearBottleneckIB depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module, kls = self._make_stage(inplanes=self.c[i],
                                           outplanes=self.c[i + 1],
                                           n=self.n[i + 1],
                                           stride=self.s[i + 1],
                                           t=self.t, stage=i)
            modules[name] = module
            kl_list.append(kls)

        return torch.nn.Sequential(modules), kl_list

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.ib1(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)
        x = self.ib_last(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        ib_kld = sum(kl.kld for kl in self.kl_list)

        return x, ib_kld
