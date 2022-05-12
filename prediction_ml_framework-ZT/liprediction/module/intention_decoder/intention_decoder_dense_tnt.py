# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer
from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .intention_decoder_base import IntentionDecoderBase


class IntentionDecoderDenseTNT(IntentionDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.intention_subgraph_encoder = nn.ModuleList([MLPLayer(cfg) for cfg in config['mlp_layers']])

        self.cross_attention_encoder_layers = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['cross_attention_layers']])

        self.mlp_decoder_3s_layers = ResMLPLayer(config['3s_res_mlp_layer'])
        self.mlp_decoder_5s_layers = ResMLPLayer(config['5s_res_mlp_layer'])
        self.mlp_decoder_8s_layers = ResMLPLayer(config['8s_res_mlp_layer'])

    def forward(self, cand_target_points, polyline_features, polyline_masks):
        '''
        cand_target_points: [B, S, F], S is the number of goals
        polyline_features: list[ [B, C, F_c] ]
            [0] sub_encoder_features: [B, C, F_c], combined with obstacle and lane features
            [1] global_encoder_feature: [B, C, F_c], combined with obstacle and lane features
        polyline_masks: list[ [B, C] ]
            [0] sub_encoder_features mask
            [1] global_encoder_feature mask
        '''

        sub_encoder_features = polyline_features[0]
        global_encoder_feature = polyline_features[1]

        sub_encoder_features_mask = polyline_masks[0]

        # [B, C, F_c] -> [B, 1, F_c] -> [B, S, F_c]
        adc_global_feature = global_encoder_feature[:, 0:1, :].expand(-1, cand_target_points.shape[1], -1)
        for idx, layer in enumerate(self.intention_subgraph_encoder):
            if idx == 0:
                # [B, S, 2] -> [B, S, F_c]
                goals_2d_feature = layer(cand_target_points)
                continue

            # [B, S, F_c + F_c]
            goals_2d_feature = torch.cat([adc_global_feature, goals_2d_feature], dim=-1)
            # [B, S, F_c + F_c] -> [B, S, F_c]
            goals_2d_feature = layer(goals_2d_feature)

        # [B, S, F_c]
        attention_goals_2d = goals_2d_feature
        for layer in self.cross_attention_encoder_layers:
            # [B, C, F_c], [B, S, F_c], [B, C] -> [B, S, F_c]
            attention_goals_2d = layer(sub_encoder_features, attention_goals_2d, sub_encoder_features_mask)

        # [B, S, F_c + F_c' + F_c'']
        goals_2d_features = torch.cat([adc_global_feature, goals_2d_feature, attention_goals_2d], dim=-1)
        # [B, S, F_c + F_c' + F_c''] -> [B, S, 1], [B, S, F_c]
        goals_score_3s_raw, _ = self.mlp_decoder_3s_layers(goals_2d_features)
        # [B, S, F_c + F_c' + F_c''] -> [B, S, 1], [B, S, F_c]
        goals_score_5s_raw, _ = self.mlp_decoder_5s_layers(goals_2d_features)
        # [B, S, F_c + F_c' + F_c''] -> [B, S, 1], [B, S, F_c]
        goals_score_8s_raw, _ = self.mlp_decoder_8s_layers(goals_2d_features)

        # [ [B, S, 1], [B, S, 1], [B, S, 1] ]
        return [goals_score_3s_raw, goals_score_5s_raw, goals_score_8s_raw]
