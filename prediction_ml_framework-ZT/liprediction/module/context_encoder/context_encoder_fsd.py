# Copyright (c) 2022 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer
from core.attention.multi_context_gating import MultiContextGating
from core.cluster.pointnet_tensor import ClusterLayer

from .context_encoder_base import ContextEncoderBase


class ContextEncoderFSD(ContextEncoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # encode vector into clusters
        self.obs_cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['obs_cluster_layers']])
        self.lane_cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['lane_cluster_layers']])

        if self.config['interaction_type'] == 'attention':
            # interaction with attention
            self.a2o_cross_attention_layer = CrossAttentionEncoderLayer(config['a2o_cross_attention'])
            self.a2l_cross_attention_layer = CrossAttentionEncoderLayer(config['a2l_cross_attention'])
        elif self.config['interaction_type'] == 'mcg':
            # interaction with mcg
            self.a2o_cross_mcg_layer = MultiContextGating(config['a2o_cross_mcg'])
            self.a2l_cross_mcg_layer = MultiContextGating(config['a2l_cross_mcg'])
            self.affine_layer = nn.Linear(config['affine']['in_size'], config['affine']['out_size'])
        else:
            raise Exception('Not implemented')


    def forward(self, vectors, vector_masks, cluster_masks):
        '''
        encoding the vectors into clusters
        vectors: list shape [B, C, L, F_v]
        vector_masks: list shape [B, C, L]
        cluster_masks: list shape [B, C]
        '''
        obs_vector = vectors[0]
        lane_vector = vectors[1]

        obs_vector_mask = vector_masks[0]
        lane_vector_mask = vector_masks[1]

        obs_cluster_mask = cluster_masks[0]
        lane_cluster_mask = cluster_masks[1]

        for layer in self.obs_cluster_layers:
            # [B, C, L, F_v], [B, C, L] -> [B, C, L, F_v+F_c], [B, C, F_c]
            obs_vector, obs_cluster_feature = layer(obs_vector, obs_vector_mask)
        for layer in self.lane_cluster_layers:
            # [B, C, L, F_v], [B, C, L] -> [B, C, L, F_v+F_c], [B, C, F_c]
            lane_vector, lane_cluster_feature = layer(lane_vector, lane_vector_mask)

        # get agent to predict
        agent_feature = obs_cluster_feature[:, 0:1]
        # agent mask set to invalid
        obs_cluster_mask[:, 0] = 0.0        
        if self.config['interaction_type'] == 'attention':
            a2o_feature = self.a2o_cross_attention_layer(obs_cluster_feature, agent_feature, x_k_mask=obs_cluster_mask)
            a2o_feature = torch.cat([agent_feature, a2o_feature], dim=-1)
            a2l_feature = self.a2l_cross_attention_layer(lane_cluster_feature, a2o_feature, x_k_mask=lane_cluster_mask)
            agent_total_feature = torch.cat([a2o_feature, a2l_feature], dim=-1)            
        elif self.config['interaction_type'] == 'mcg':
            _, a2o_feature = self.a2o_cross_mcg_layer(obs_cluster_feature, agent_feature, obs_cluster_mask)
            a2o_feature = torch.cat([agent_feature, a2o_feature], dim=-1)
            a2o_feature = self.affine_layer(a2o_feature)
            _, a2l_feature = self.a2l_cross_mcg_layer(lane_cluster_feature, a2o_feature, lane_cluster_mask)            
            agent_total_feature = torch.cat([agent_feature, a2o_feature, a2l_feature], dim=-1)

        return agent_total_feature
