# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    '''
    MLP Ops: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        in_size = config['in_size']
        hidden_size = config['hidden_size']

        op_seq = []
        op_seq.append(nn.Linear(in_size, hidden_size))

        norm_type = config['norm_type']
        if norm_type == 'BN':
            op_seq.append(nn.BatchNorm1d(hidden_size))
        elif norm_type == 'LN':
            op_seq.append(nn.LayerNorm(hidden_size))
        elif norm_type == 'none':
            pass
        else:
            raise Exception(f'{norm_type} Not implemented')

        active_func = config['active_func']
        if active_func == 'ReLU':
            op_seq.append(nn.ReLU(inplace=True))
        elif active_func == 'Tanh':
            op_seq.append(nn.Tanh())
        elif active_func == 'sigmoid':
            op_seq.append(nn.Sigmoid())
        elif active_func == 'softmax':
            op_seq.append(nn.Softmax(dim=-1))                                      
        elif active_func == 'none':
            pass
        else:
            raise Exception(f'{active_func} Not implemented')

        self.mlp = nn.Sequential(*op_seq)

    def forward(self, x, mask=None):
        '''
            x:      [..., F]
            mask:   [...]
            return: [..., F']
        '''

        # [..., F] -> [..., F']
        # for compatible with batch_norm and layer_norm reshape to [N, F]
        # [..., F]
        raw_shape = x.shape
        # [N, F]
        tmp_shape = (-1, raw_shape[-1])
        # [...] + [-1] -> [..., F']
        new_shape = raw_shape[:-1] + (-1,)

        # [..., F] -> [N, F] -> [N, F'] -> [..., F']
        y = self.mlp(x.view(tmp_shape)).view(*new_shape)

        # [..., F']
        out = y
        if mask is not None:
            # Fills elements of self tensor with value where mask is True
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            # mask: [...] 0,1 -> [...] True,False -> [..., 1] True,False
            # masked_fill: [..., F'] -> [..., F'], mlp has relu, so min value of y is 0
            out = y.masked_fill((mask == 0).unsqueeze(-1), 0)

        # [..., F']
        return out
