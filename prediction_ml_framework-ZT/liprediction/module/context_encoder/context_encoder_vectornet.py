# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch.nn as nn
from core.attention.encoder_layer_tensor import SelfAttentionEncoderLayer
from core.cluster.pointnet_tensor import ClusterLayer

from .context_encoder_base import ContextEncoderBase


class ContextEncoderVectorNet(ContextEncoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # encode vector into clusters
        self.cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['cluster_layers']])

        # context transformer encoders
        self.encoder_layers = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['encoder_layers']])

    def forward(self, vector, vector_mask, cluster_mask):
        '''
            encoding the vector into clusters
            vector: shape=[B, C, L, F_v]
            vector_mask: shape=[B, C, L]
            cluster_mask: shape=[B, C]
        '''

        # enconding vector into cluster feature using pointnet
        for cluster_layer in self.cluster_layers:
            # [B, C, L, F_v], [B, C, L] -> [B, C, L, F_v+F_c], [B, C, F_c]
            vector, cluster_feature = cluster_layer(vector, vector_mask)

        # enconding cluster feature using self attention
        for encoder_layer in self.encoder_layers:
            # [B, C, F_c], [B, C] -> [B, C, F_c']
            cluster_feature = encoder_layer(cluster_feature, cluster_mask)
        # [B, C, F_c']
        return cluster_feature
