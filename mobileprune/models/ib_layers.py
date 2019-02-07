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

"""From https://github.com/zhuchen03/VIBNet/blob/0ae05cd4aa02d7aa615af6287c9655306616ab6f/ib_layers.py
"""

import torch
from torch.nn.parameter import Parameter
import numpy as np


def _reparameterize(mu, logvar, batch_size, cuda=False, sampling=True):
    # output dim: batch_size * dim
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(batch_size, std.size(0))
        eps = eps.cuda(mu.get_device()).normal_()

        return mu.view(1, -1) + eps * std.view(1, -1)

    return mu.view(1, -1)


class InformationBottleneck(torch.nn.Module):
    def __init__(self,
                 dim,
                 mask_thresh=0,
                 init_mag=9,
                 init_var=0.01,
                 kl_mult=1,
                 divide_w=False,
                 sample_in_training=True,
                 sample_in_testing=False):
        torch.nn.Module.__init__(self)

        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing

        # initialization
        self.post_z_mu.data.normal_(1, init_var)
        self.prior_z_logD.data.normal_(-init_mag, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        self.need_update_z = True # flag for updating z during testing
        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult
        self.divide_w = divide_w


    def adapt_shape(self, src_shape, x_shape):
        # to distinguish conv layers and fc layers
        # see if we need to expand the dimension of x
        new_shape = src_shape if len(src_shape) == 2 else (1, src_shape[0])
        if len(x_shape) > 2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape) - 2)]

        return new_shape

    def get_logalpha(self):
        return (self.post_z_logD.data -
                torch.log(self.post_z_mu.data.pow(2) + self.epsilon))

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)

        return alpha / (1 + alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()

        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float()*self.post_z_mu.data

        return mask

    def forward(self, x):
        # 4 modes: sampling, hard mask, weighted mask, use mean value
        bsize = x.size(0)
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            z_scale = _reparameterize(self.post_z_mu,
                                      self.post_z_logD,
                                      bsize,
                                      cuda=True,
                                      sampling=True)
            if not self.training:
                z_scale *= self.get_mask_hard(self.mask_thresh)
        else:
            z_scale = self.get_mask_weighted(self.mask_thresh)

        self.kld = self.kl_closed_form(x)
        new_shape = self.adapt_shape(z_scale.size(), x.size())

        return x * z_scale.view(new_shape)

    def kl_closed_form(self, x):
        new_shape = self.adapt_shape(self.post_z_mu.size(), x.size())

        h_D = torch.exp(self.post_z_logD.view(new_shape))
        h_mu = self.post_z_mu.view(new_shape)

        # TODO(brendan): Why is the KLD scaled by C_in / C_out?
        KLD = torch.sum(
            torch.log(1 + h_mu.pow(2)/(h_D + self.epsilon)))
        KLD *= x.size(1) / h_D.size(1)

        if x.dim() > 2:
            if self.divide_w:
                # divide by the width
                KLD *= x.size()[2]
            else:
                KLD *= np.prod(x.size()[2:])

        return KLD * 0.5 * self.kl_mult
