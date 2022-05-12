# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor


class MultiContextGating(nn.Module):
    '''
    Multi Context Gating: MCG - https://arxiv.org/pdf/2111.14973.pdf
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.stacked_num = config['stacked_num']
        self.context_gatings = nn.ModuleList([ContextGating(config) for _ in range(self.stacked_num)])

    def forward(self, s: Tensor, c: Tensor, s_mask=None):
        '''MCG model forward
        Args:
            s: [..., L, F]
            c: [..., 1, F]
            s_mask: [..., L]

        Return:
            s: [..., L, F']
            c: [..., 1, F']
        '''

        s_bar = s
        c_bar = c

        # for loop of stacked_num
        for idx in range(self.stacked_num):
            # [..., L, F], [..., 1, F], [..., L] -> [..., L, F'], [..., 1, F']
            s_new, c_new = self.context_gatings[idx](s_bar, c_bar, s_mask)

            # [..., L, F'], [..., L, F]  -> [..., L, F']
            s_bar = s_new + s
            # [..., 1, F'], [..., 1, F]  -> [..., 1, F']
            c_bar = c_new + c

        # [..., L, F'], [..., 1, F']
        return s_bar, c_bar


class ContextGating(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feature_size = config['feature_size']
        self.pooling = config['pooling']  # max or avg

        self.mlp_s = nn.Linear(self.feature_size, self.feature_size)
        self.mlp_c = nn.Linear(self.feature_size, self.feature_size)
        self.layer_norm = nn.LayerNorm(self.feature_size)

    def forward(self, s: Tensor, c: Tensor, s_mask=None):
        '''CG module forward'''
        # [..., L, F] -> [..., L, F']
        s_hidden = self.mlp_s(s)
        # [..., 1, F] -> [..., 1, F']
        c_hidden = self.mlp_c(c)

        # element-wise product: [..., L, F] * [..., 1, F] -> [..., L, F]
        s_new = torch.mul(s_hidden, c_hidden)

        if s_mask is not None:
            # [..., L, F'], [..., L, 1] -> [..., L, F'] with mask feature
            s_new = s_new.masked_fill((s_mask == 0).unsqueeze(-1), 0.)

        # [..., L, F'] -> [..., L, F']
        s_new = self.layer_norm(s_new)

        if self.pooling == 'max':
            # max pooling: [..., L, F'] -> [..., 1, F']
            c_new = torch.max(s_new, dim=-2, keepdim=True)[0]
        elif self.pooling == 'avg':
            # max pooling: [..., L, F'] -> [..., 1, F']
            c_new = torch.mean(s_new, dim=-2, keepdim=True)
        else:
            raise ValueError('pooling method not supported')

        # [..., L, F'], [..., 1, F']
        return s_new, c_new
