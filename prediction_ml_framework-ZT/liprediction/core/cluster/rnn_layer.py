# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn


class LSTMLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.use_bias = config['bias'] if 'bias' in config else True
        self.is_reduce = config['is_reduce'] if 'is_reduce' in config else True

        self.lstm = nn.LSTM(config['in_size'],
                            config['hidden_size'],
                            num_layers=config['layer_num'],
                            batch_first=True,
                            bias=self.use_bias)
        self.layer_norm = nn.LayerNorm(config['hidden_size'])

    def forward(self, x):
        '''
            x:      [B, L, F] must be dim3
            return: [B, F'] (reduce length) is_reduce is True(normal)
                    [B, L, F'] is_reduce is False
        '''
        output, (h, c) = self.lstm(x)
        if self.is_reduce:
            output = output[..., -1, :]
        output = self.layer_norm(output)
        return output
