# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attention.multi_context_gating import MultiContextGating
from core.cluster.mlp_layer import MLPLayer

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderMultiPathDecoder(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # create learned anchor embedding
        self.query_vector = nn.Embedding(self.config['num_of_query'], config['query_vector_dim'])

        self.predictor_mcg_decoder = MultiContextGating(config['predictor_mcg_decoder'])
        self.predictor_mlp_decoder = MLPLayer(config['predictor_mlp_decoder'])

        self.hidden_mlp_layer = MLPLayer(config['hidden_mlp_layer'])
        self.prob_mlp_layer = MLPLayer(config['prob_mlp_layer'])
        self.traj_pt_mlp_layer = MLPLayer(config['traj_pt_mlp_layer'])
        self.traj_sigma_mlp_layer = MLPLayer(config['traj_sigma_mlp_layer'])
        self.traj_rho_mlp_layer = MLPLayer(config['traj_rho_mlp_layer'])

    def forward(self, global_mcg_ctx):
        '''Trajectory decoder forward

        Args:
            global_mcg_ctx: [B, 1, F]

        Returns:
            predict_traj: [B, M, 1], [B, M, T*2] generate T traj points for each modality
        '''
        # [B, M, F]
        query_embeds = self.query_vector.weight
        # [B, M, F], [B, 1, F] -> [B, M, F]
        predictor_mcg_feature, _ = self.predictor_mcg_decoder(query_embeds, global_mcg_ctx)

        # [B, M, F] -> [B, M, F']
        predictor_mlp_feature = self.predictor_mlp_decoder(predictor_mcg_feature)

        # [B, M, F'] -> [B, M, F']
        hidden_state = self.hidden_mlp_layer(predictor_mlp_feature)
        # [B, M, F'], [B, M, F'] -> [B, M, 1]
        traj_prob = self.prob_mlp_layer(torch.cat([predictor_mlp_feature, hidden_state], dim=-1))
        # [B, M, F'], [B, M, F'] -> [B, M, T*2]
        traj_pt = self.traj_pt_mlp_layer(torch.cat([predictor_mlp_feature, hidden_state], dim=-1))
        # [B, M, F'], [B, M, F'] -> [B, M, T*2]
        traj_sigma = self.traj_sigma_mlp_layer(torch.cat([predictor_mlp_feature, hidden_state], dim=-1))
        # [B, M, F'], [B, M, F'] -> [B, M, T*1]
        traj_rho = self.traj_rho_mlp_layer(torch.cat([predictor_mlp_feature, hidden_state], dim=-1))

        # T
        time_steps = traj_pt.shape[-1] // 2
        # [B, M, T, 2]
        traj_pt = traj_pt.reshape(traj_pt.shape[0], traj_pt.shape[1], time_steps, 2)
        # [B, M, T, 2]
        traj_sigma = traj_sigma.reshape(traj_sigma.shape[0], traj_sigma.shape[1], time_steps, 2)
        # [B, M, T, 1]
        traj_rho = traj_rho.reshape(traj_rho.shape[0], traj_rho.shape[1], time_steps, 1)

        # [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1]
        return traj_prob, traj_pt, traj_sigma, traj_rho
