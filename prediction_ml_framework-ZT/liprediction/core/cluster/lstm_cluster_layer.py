# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.cluster.mlp_layer import MLPLayer
from core.cluster.rnn_layer import LSTMLayer


class LSTMClusterLayer(nn.Module):
    ''' LSTM Cluster Layer'''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = MLPLayer(config['embedding'])
        self.lstm = LSTMLayer(config['lstm'])

    def forward(self, x, mask=None):
        """LSTM cluster forward

        Args:
            x: shape [..., L, F]
            mask: shape [..., L]
        """
        assert x.dim() >= 3 and mask.dim() >= 2

        origin_shape = x.shape
        L = x.shape[-2]
        F = x.shape[-1]

        # [..., L, F] -> [-1, L, F]
        x = x.view(-1, L, F)

        if mask is not None:
            # [..., L] -> [-1, L]
            mask = mask.view(-1, L)

        # embedding: [-1, L, F], [-1, L] -> [-1, L, F]
        x_embed = self.embedding(x, mask)

        # LSTM: [-1, L, F] -> [-1, F']
        lstm_out = self.lstm(x_embed)

        # Reshape
        # new_shape: [...] + [F'] = [..., F']
        new_shape = origin_shape[:-2] + lstm_out.shape[-1:]
        # [-1, F'] -> [..., F']
        cluster_out = lstm_out.view(*new_shape)

        return cluster_out
