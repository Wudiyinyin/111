# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn

from core.cluster.res_mlp_layer import ResMLPLayer
from core.cluster.mlp_layer import MLPLayer
from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderFSD(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.point_num = config['point_num']

        self.res_affine = ResMLPLayer(config['res_affine'])
        self.mlp_decoder = MLPLayer(config['mlp_decoder'])


    def forward(self, agent_all_feature):
        '''
        traj.shape : [B, K, N, 2]
        '''
        B, _, _ = agent_all_feature.shape
        affine_embedding = self.res_affine(agent_all_feature)
        traj = self.mlp_decoder(affine_embedding)
        traj = traj.view(B, self.point_num, 2)
        return traj
