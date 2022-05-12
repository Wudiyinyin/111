# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderMlp(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # traj prediction
        self.traj_prediction = nn.Sequential(
            nn.Linear(config['traj_prediction']['in_size'], config['traj_prediction']['hidden_size']),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config['traj_prediction']['hidden_size']),
            nn.Linear(config['traj_prediction']['hidden_size'], config['traj_prediction']['out_size']),
        )
        # TODO intention encoding

    def forward(self, cluster_feature, cluster_mask, intention):
        '''
        intention.shape : [B, K, 2]
        traj.shape : [B, K, 30, 2]
        '''
        # batch_size, topk, 2
        B, K, L = intention.shape[0], intention.shape[1], self.config['pred_horizon']
        # [B, C, F] -> [B, 1, F] -> [B, K, F]
        agent_cluster_feature = cluster_feature[:, [0], :].expand(-1, K, -1)
        # [B, K, F], [B, K, 2] -> [B, K, F+2]
        traj_feature = torch.cat([agent_cluster_feature, intention], dim=-1)
        # [B, K, F+2] -> [B, K, L*2] -> [B, K, L, 2]
        traj = self.traj_prediction(traj_feature).view(B, K, L, 2)
        # [B, K, L, 2]
        return traj
