# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.multi_context_gating import MultiContextGating
from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderMultiPathPro(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.agent_feature_size = config['agent_feature_size']

        self.target_goals_encoder = nn.ModuleList([MLPLayer(cfg) for cfg in config['mlp_layers']])

        self.target_mcg_encoder = MultiContextGating(config['target_mcg_encoder'])
        self.mlp_decoder_layers = ResMLPLayer(config['res_mlp_layer'])

    def forward(self, teacher_points, global_mcg_ctx):
        '''Trajectory decoder forward

        Args:
            teacher_points: [B, I, 2], I is the number of goals
            global_mcg_ctx: [B, 1, F_c]

        Returns:
            predict_traj: [B, I, T*2] generate T traj points for each target point I
        '''

        # [B, I, 2]
        target_goals_feature = teacher_points
        for layer in self.target_goals_encoder:
            # [B, I, 2] -> [B, I, F]
            target_goals_feature = layer(target_goals_feature)

        # [B, I, F], [B, 1, F] -> [B, I, F]
        target_attention_feature, _ = self.target_mcg_encoder(target_goals_feature, global_mcg_ctx)

        # [B, 1, F_c] -> [B, 1, F_a] -> [B, I, F_a]
        adc_global_feature = global_mcg_ctx[:, :, :self.agent_feature_size].expand(-1, teacher_points.shape[1], -1)

        # [B, I, F_a + F + F]
        target_goals_features = torch.cat([adc_global_feature, target_goals_feature, target_attention_feature], dim=-1)
        # [B, I , F_a + F + F] -> [B, I, T*2]
        predict_traj, _ = self.mlp_decoder_layers(target_goals_features)

        # [B, I, T*2]
        return predict_traj
