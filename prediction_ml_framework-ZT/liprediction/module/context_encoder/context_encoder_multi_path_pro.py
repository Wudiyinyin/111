# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.multi_context_gating import MultiContextGating
from core.cluster.lstm_cluster_layer import LSTMClusterLayer
from core.cluster.pointnet_tensor import ClusterLayer
from module.context_encoder.context_encoder_base import ContextEncoderBase


class ContextEncoderMultiPathPro(ContextEncoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.obs_lstm_encoder = LSTMClusterLayer(config['obs_encoder'])
        self.obs_diff_encoder = LSTMClusterLayer(config['obs_diff_encoder'])

        self.lane_encoder = nn.ModuleList([ClusterLayer(cfg) for cfg in config['lane_encoder']])

        self.agent_history_encoder = MultiContextGating(config['agent_history_encoder'])
        self.interaction_encoder = MultiContextGating(config['interaction_encoder'])
        self.road_graph_encoder = MultiContextGating(config['road_graph_encoder'])

    def forward(self, vectors, vector_masks, cluster_masks):
        ''' Context encoder enable vectors into clusters

        Args:
            vectors: obstacle and polyline list shape [B, C, L, F]
            vector_masks: obstacle and polyline list shape [B, C, L]
            cluster_masks: obstacle and polyline list shape [B, C]
        '''

        # split vector list, [B, C, L, F]  前后2帧做diff之后的向量
        obs_vector = vectors[0]
        obs_diff_vector = vectors[1]
        lane_vector = vectors[2]

        # split mask list, [B, C, L]
        obs_vector_mask = vector_masks[0]
        obs_diff_vector_mask = vector_masks[1]
        lane_vector_mask = vector_masks[2]

        # split cluster mask list, [B, C]
        obs_cluster_mask = cluster_masks[0]
        # obs_diff_cluster_mask = cluster_masks[1]
        lane_cluster_mask = cluster_masks[2]

        # obstacle vector cluster : [B, C, L, F], [B, C, L] -> [B, C, F']  L
        obs_lstm_feature = self.obs_lstm_encoder(obs_vector, obs_vector_mask)
        # [B, C, L-1, F], [B, C, L-1] -> [B, C, F'] L-1
        obs_diff_feature = self.obs_diff_encoder(obs_diff_vector, obs_diff_vector_mask)

        # [B, C, F'], [B, C, F''], [B, C, F''] -> [B, C, F' + F'' + F'']
        obs_cluster_feature = torch.cat([obs_lstm_feature, obs_diff_feature], dim=-1)

        # lane vector cluster : [B, C, L, F], [B, C, L] -> [B, C, L, F + F'], [B, C, F']
        for layer in self.lane_encoder:
            lane_vector, lane_cluster_feature = layer(lane_vector, lane_vector_mask)

        # init ones vector for init MCG context
        B, _, F = obs_cluster_feature.shape
        agent_init_ctx = torch.ones((B, 1, F), dtype=torch.float32).type_as(obs_vector)

        # agent mcg history encoder
        agent_cluster_feature = obs_cluster_feature[:, 0:1, :]
        agent_cluster_mask = obs_cluster_mask[:, 0:1]
        # [B, 1, F], [B, 1, F], [B, 1] -> [B, 1, F'], [B, 1, F']
        _, agent_mcg_ctx = self.agent_history_encoder(agent_cluster_feature, agent_init_ctx, agent_cluster_mask)

        # interaction obstacle (without predicted agent) mcg encoder
        # [B, C, F'] -> [B, C-1, F']
        interact_obs_feature = obs_cluster_feature[:, 1:, :]
        interact_obs_mask = obs_cluster_mask[:, 1:]
        # [B, C-1, F'], [B, C-1, F'], [B, C-1] -> [B, C-1, F'], [B, 1, F']
        _, interact_obs_mcg_ctx = self.interaction_encoder(interact_obs_feature, agent_mcg_ctx, interact_obs_mask)

        # [B, 1, F'], [B, 1, F'] -> [B, 1, F' + F']
        obs_mcg_ctx = torch.cat([agent_mcg_ctx, interact_obs_mcg_ctx], dim=-1)

        # road graph MCG encoder: [B, C, F_l], [B, 1, F' + F'], [B, C] -> [B, C, F_l], [B, 1, F_l]
        _, road_graph_mcg_ctx = self.road_graph_encoder(lane_cluster_feature, obs_mcg_ctx, lane_cluster_mask)

        # global MCG context: [B, 1, F' + F'], [B, 1 ,F_l] -> [B, 1, F' + F' + F_l]
        global_mcg_ctx = torch.cat([obs_mcg_ctx, road_graph_mcg_ctx], dim=-1)

        # [B, 1, F' + F' + F_l]
        return global_mcg_ctx
