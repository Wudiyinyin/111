# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer, SelfAttentionEncoderLayer
from core.cluster.pointnet_tensor import ClusterLayer

from .context_encoder_base import ContextEncoderBase


class ContextEncoderDenseTNT(ContextEncoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # encode vector into clusters
        self.obs_cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['obs_cluster_layers']])
        self.lane_cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['lane_cluster_layers']])

        # context cross attention encoders
        self.cross_attention_encoder_layers = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['cross_attention_encoder_layers']])

        # context transformer encoders
        self.a2a_self_attention_encoder_layers = nn.ModuleList(
            [SelfAttentionEncoderLayer(cfg) for cfg in config['a2a_self_attention_encoder_layers']])
        self.l2l_self_attention_encoder_layers = nn.ModuleList(
            [SelfAttentionEncoderLayer(cfg) for cfg in config['l2l_self_attention_encoder_layers']])
        self.all_self_attention_encoder_layers = nn.ModuleList(
            [SelfAttentionEncoderLayer(cfg) for cfg in config['all_self_attention_encoder_layers']])

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

        # [B, 1, F_c], extract adc cluster feature from obstacle features
        adc_cluster_feature = obs_cluster_feature[:, 0:1, :].expand(-1, lane_cluster_feature.shape[1], -1)
        # [B, C, F_c + F_c], combine adc feature and lane cluster feature
        lane_key_feature = torch.cat([lane_cluster_feature, adc_cluster_feature], dim=-1)
        lane_key_mask = lane_cluster_mask
        lane_cluster_feature_l2a = lane_cluster_feature
        for layer in self.cross_attention_encoder_layers:
            # [B, C, F_c + F_c], [B, C, F_c], [B, C] -> [B, C, F_c]
            lane_cluster_feature_l2a = layer(lane_key_feature, lane_cluster_feature_l2a, lane_key_mask)

        # sub polyline shape : [B, C, F_c], [B, C', F_c] -> [B, C + C', F_c]
        sub_polyline_feature = torch.cat([obs_cluster_feature, lane_cluster_feature_l2a], dim=1)

        # [B, C, F_c]
        obs_cluster_feature_a2a = obs_cluster_feature
        for layer in self.a2a_self_attention_encoder_layers:
            # [B, C, F_c], [B, C] -> [B, C, F_c']
            obs_cluster_feature_a2a = layer(obs_cluster_feature_a2a, obs_cluster_mask)

        # [B, C, F_c]
        lane_cluster_feature_l2l = lane_cluster_feature_l2a
        for layer in self.l2l_self_attention_encoder_layers:
            # [B, C, F_c], [B, C] -> [B, C, F_c']
            lane_cluster_feature_l2l = layer(lane_cluster_feature_l2l, lane_cluster_mask)

        # global polyline shape : [B, C, F_c'], [B, C', F_c'] -> [B, C + C', F_c']
        global_polyline_feature = torch.cat([obs_cluster_feature_a2a, lane_cluster_feature_l2l], dim=1)
        # [B, C + C']
        global_polyline_masks = torch.cat([obs_cluster_mask, lane_cluster_mask], dim=-1)
        for layer in self.all_self_attention_encoder_layers:
            # [B, C + C', F_c] -> [B, C + C', F_c']
            global_polyline_feature = layer(global_polyline_feature, global_polyline_masks)

        # [ [B, C + C', F_c'], [B, C + C', F_c'] ]
        polyline_features = [sub_polyline_feature, global_polyline_feature]
        # [ [B, C + C'], [B, C + C'] ]
        polyline_masks = [
            torch.cat([obs_cluster_mask, lane_cluster_mask], dim=-1),
            torch.cat([obs_cluster_mask, lane_cluster_mask], dim=-1)
        ]
        # [ [B, C + C', F_c'], [B, C + C', F_c'] ], [ [B, C + C'], [B, C + C'] ]
        return polyline_features, polyline_masks
