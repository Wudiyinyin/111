# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn


class ResMLPLayer(nn.Module):
    '''
    Residual MLP Module: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add = True if 'add' in self.config and self.config['add'] else False

        self.mlp = nn.Sequential(
            nn.Linear(config['in_size'], config['hidden_size']),
            nn.LayerNorm(config['hidden_size']),
            nn.ReLU(inplace=True),
        )

        if not self.add:
            self.linear = nn.Linear(config['linear_hidden_size'], config['out_size'])

    def forward(self, x, mask=None):
        '''
            x:      [..., F]
            mask:   [...]
            return: [..., F''], [..., F'], F''=out_size F'=hidden_size
        '''
        # [..., F] -> [..., F']
        y = self.mlp(x)
        # [..., F']
        hidden_state = y
        if mask is not None:
            # Fills elements of self tensor with value where mask is True
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            # mask: [...] 0,1 -> [...] True,False -> [..., 1] True,False
            # masked_fill: [..., F'] -> [..., F'], mlp has relu, so min value of y is 0
            hidden_state = y.masked_fill((mask == 0).unsqueeze(-1), 0)

        if self.add:
            out = x + hidden_state
            return out
        # [..., F + F']
        res_hidden_state = torch.cat([x, hidden_state], dim=-1)
        # [..., F + F'] -> [..., F'']  F + F'=linear_hidden_size  F''=out_size
        out = self.linear(res_hidden_state)

        # [..., F''], [..., F']
        return out, hidden_state
