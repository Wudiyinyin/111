# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer
from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderDenseTNT(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.target_goals_encoder = nn.ModuleList([MLPLayer(cfg) for cfg in config['mlp_layers']])

        self.cross_attention_encoder_layers = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['cross_attention_layers']])

        self.mlp_decoder_layers = ResMLPLayer(config['res_mlp_layer'])

    def forward(self, teacher_points, polyline_features, polyline_masks):
        '''
        teacher_points: [B, I, 2], I is the number of goals

        polyline_features: list
        [0] sub_encoder_features [B, C, F_c], combined with obstacle and lane features
        [1] global_encoder_feature: [B, C, F_c'], combined with obstacle and lane features

        polyline_masks: list
        [0] sub_encoder_features_mask [B, C]
        [1] global_encoder_feature_mask [B, C]

        return [B, I, T*2] generate T traj points for each target point I
        '''

        sub_encoder_features = polyline_features[0]
        global_encoder_feature = polyline_features[1]

        sub_encoder_features_mask = polyline_masks[0]

        # [B, I, 2]
        target_goals_feature = teacher_points
        for layer in self.target_goals_encoder:
            # [B, I, 2] -> [B, I, F]
            target_goals_feature = layer(target_goals_feature)

        # [B, I, F]
        target_attention_feature = target_goals_feature
        for layer in self.cross_attention_encoder_layers:
            # [B, C, F_c], [B, I, F], [B, C] -> [B, I, F]
            target_attention_feature = layer(sub_encoder_features, target_attention_feature, sub_encoder_features_mask)

        # [B, 1, F] -> [B, I, F]
        adc_global_feature = global_encoder_feature[:, 0:1, :].expand(-1, teacher_points.shape[1], -1)
        # [B, I, F + F + F]
        target_goals_features = torch.cat([adc_global_feature, target_goals_feature, target_attention_feature], dim=-1)
        # [B, I , F + F + F] -> [B, I, T*2]
        predict_traj, _ = self.mlp_decoder_layers(target_goals_features)

        # [B, I, T*2]
        return predict_traj
