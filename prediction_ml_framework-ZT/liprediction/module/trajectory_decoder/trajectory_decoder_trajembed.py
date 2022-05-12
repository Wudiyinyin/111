# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderTrajEmbeding(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # traj embeding
        self.traj_embeding = nn.Sequential(
            nn.Linear(config['traj_embeding']['in_size'],
                      config['traj_embeding']['hidden_size']), nn.ReLU(inplace=True),
            nn.Linear(config['traj_embeding']['hidden_size'], config['traj_embeding']['out_size']),
            nn.LayerNorm(config['traj_embeding']['out_size']), nn.ReLU(inplace=True))

        # decoder layers
        self.decoder_layers = nn.ModuleList([CrossAttentionEncoderLayer(cfg) for cfg in config['traj_decoder_layers']])

        self.dropout = nn.Dropout(p=config['traj_dropout'])

        # traj prediction
        self.traj_prediction = nn.Sequential(
            nn.Linear(config['traj_prediction']['in_size'], config['traj_prediction']['hidden_size']),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config['traj_prediction']['hidden_size']),
            nn.Linear(config['traj_prediction']['hidden_size'], 2),
        )

    def forward(self, cluster_feature, cluster_mask, intention):
        '''
        intention.shape : [B, K, 2]
        '''

        (B, K), L, C = intention.shape[:2], self.config['pred_horizon'], cluster_feature.shape[1]
        # [B, C, F] -> [B, 1, C, F] -> [B, K, C, F] -> [B*K, C, F]
        cluster_feature = cluster_feature.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, C, -1)
        # [B, C] -> [B, 1, C] -> [B, K, C] -> [B*K, C]
        cluster_mask = cluster_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, C)
        # [B, K, 2] -> [B, K, 1, 2] -> [B, K, L, 2] -> [B*K, L, 2]
        intention = intention.unsqueeze(2).expand(-1, -1, L, -1).reshape(B * K, L, -1)

        # construct pix feature
        # [B*K, L, 2]
        traj_embeding = self.genTrajEmbeding(B, K, L, cluster_feature.device, cluster_feature.dtype)
        # [B*K, L, 2] + [B*K, L, 2] -> [B*K, L, 4] -> [B*K, L, 32]
        traj_feature = self.traj_embeding(torch.cat([traj_embeding, intention], dim=-1))

        # decoding traj feature using bipartite attention
        # [B*K, C, F], [B*K, L, 32], [B*K, C] -> [B*K, L, 32]
        for decoder_layer in self.decoder_layers:
            traj_feature = decoder_layer(cluster_feature, traj_feature, cluster_mask)

        # do dropout
        # [B*K, L, 32]
        traj_feature = self.dropout(traj_feature)

        # decode the pix score
        # [B*K, L, 32] -> [B*K, L, 2] -> [B, K, L, 2]
        traj = self.traj_prediction(traj_feature).view(B, K, L, -1)

        return traj

    def genTrajEmbeding(self, B, K, L, device='cpu', dtype=torch.float32):
        # sample L points from [0,1]
        a = torch.arange(L, device=device, dtype=dtype) / (L - 1)
        # [L] -> [L, 2] -> [1, L, 2] -> [1, 1, L, 2] -> [B, K, L, 2] -> [B*K, L, 2]
        traj_embeding = torch.stack([torch.sin(a), torch.cos(a)],
                                    dim=-1).unsqueeze(0).unsqueeze(0).expand(B, K, L, 2).view(B * K, L, 2)
        # [B*K, L, 2]
        return traj_embeding
