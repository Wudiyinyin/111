# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F


class LinearResidualBlock(nn.Module):
    '''
    Residual block with two linear layers
    '''

    def __init__(self, in_size, out_size, hidden_size=None, norm_type='BN', hidden_norm=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = out_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        self.out_size = out_size

        residual_seq = []
        residual_seq.append(nn.Linear(self.in_size, self.hidden_size))

        if hidden_norm:
            if norm_type == 'BN':
                residual_seq.append(nn.BatchNorm1d(self.hidden_size))
            elif norm_type == 'LN':
                residual_seq.append(nn.LayerNorm(self.hidden_size))
            else:
                raise Exception('Not implemented')

        residual_seq.append(nn.ReLU(inplace=True))

        residual_seq.append(nn.Linear(self.hidden_size, self.out_size))

        if norm_type == 'BN':
            residual_seq.append(nn.BatchNorm1d(self.out_size))
        elif norm_type == 'LN':
            residual_seq.append(nn.LayerNorm(self.out_size))
        else:
            raise Exception('Not implemented')

        self.residual_seq = nn.Sequential(*residual_seq)

    def forward(self, x):
        '''
         x: shape=[batch_size, feature_size]
         return: [batch_size, out_size]
        '''
        # [batch_size, feature_size] -> [batch_size, out_size]
        residual = self.residual_seq(x)

        # output shape maybe different with input, so interpolate is needed
        # x: [batch_size, feature_size] -> [batch_size, 1, feature_size]
        # -> [batch_size, 1, out_size] -> [batch_size, out_size]
        # Currently temporal, spatial and volumetric sampling are supported,
        # i.e. expected inputs are 3-D, 4-D or 5-D in shape.
        bypass = F.interpolate(x.unsqueeze(1), size=self.out_size, mode='nearest').view(x.shape[0], -1)
        # [batch_size, out_size] + [batch_size, out_size] -> [batch_size, out_size]
        out = F.relu(bypass + residual)
        # [batch_size, out_size]
        return out
