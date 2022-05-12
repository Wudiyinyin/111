# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.multi_context_gating import MultiContextGating
from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .intention_decoder_base import IntentionDecoderBase


class IntentionDecoderMultiPathPro(IntentionDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.agent_feature_size = config['agent_feature_size']

        self.intention_subgraph_encoder = nn.ModuleList([MLPLayer(cfg) for cfg in config['mlp_layers']])

        self.candi_target_mcg_encoder = MultiContextGating(config['candi_target_mcg_encoder'])

        self.mlp_decoder_3s_layers = ResMLPLayer(config['3s_res_mlp_layer'])
        self.mlp_decoder_5s_layers = ResMLPLayer(config['5s_res_mlp_layer'])
        self.mlp_decoder_8s_layers = ResMLPLayer(config['8s_res_mlp_layer'])

    def forward(self, cand_target_points, global_mcg_ctx):
        '''Intention decoder forward

        Args:
            cand_target_points: [B, S, 2], S is the number of goals
            global_mcg_ctx: [B, 1, F_c]
        '''

        # [B, 1, F_c] -> [B, 1, F_a] -> [B, S, F_a]
        adc_global_feature = global_mcg_ctx[:, :, :self.agent_feature_size].expand(-1, cand_target_points.shape[1], -1)

        for idx, layer in enumerate(self.intention_subgraph_encoder):
            if idx == 0:
                # [B, S, 2] -> [B, S, F_c]
                goals_2d_feature = layer(cand_target_points)
                continue

            # [B, S, F_c + F_a]
            goals_2d_feature = torch.cat([adc_global_feature, goals_2d_feature], dim=-1)
            # [B, S, F_c + F_c] -> [B, S, F_c]
            goals_2d_feature = layer(goals_2d_feature)

        # [B, S, F_c], [B, 1, F_c] -> [B, S, F_c]
        attention_goals_2d, _ = self.candi_target_mcg_encoder(goals_2d_feature, global_mcg_ctx)

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
