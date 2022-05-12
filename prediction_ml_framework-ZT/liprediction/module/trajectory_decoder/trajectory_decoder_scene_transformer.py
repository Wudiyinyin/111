# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer, SelfAttentionEncoderLayer
from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .trajectory_decoder_base import TrajectoryDecoderBase


class TrajDecoderSceneTransformer(TrajectoryDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.F = config['future_num']

        self.t_layers_mlp = nn.ModuleList([MLPLayer(cfg) for cfg in config['t_layers_mlp']])

        self.u_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['u_layers_time']])
        self.v_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['v_layers_agent']])
        self.w_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['w_layers_time']])
        self.x_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['x_layers_agent']])

        self.y_layer_norm = nn.LayerNorm(config['y_layer_norm']['feature_dim'])

        self.z1_layers_mlp = nn.ModuleList([MLPLayer(cfg) for cfg in config['z1_layers_mlp']])
        self.z2_layers_mlp = nn.ModuleList([MLPLayer(cfg) for cfg in config['z2_layers_mlp']])

    def forward(self, actor_vector_a2l, actor_vector_a2l_mask):
        '''
        Args:
            actor_vector_a2l: [B, A, T, D]
            actor_vector_a2l_mask: [B, A, T]

        Return:
            traj:[B, F, A, T, 2]
            score:[B, F, A, 1]
        '''

        B, A, T, D = actor_vector_a2l.shape

        # Layer R: [B, A, T, D] -> [B, 1, A, T, D] -> [B, F, A, T, D]
        actor_vector_r = actor_vector_a2l.unsqueeze(1).expand(-1, self.F, -1, -1, -1)
        # assert actor_vector_r.shape == (B, self.F, A, T, D)
        # [B, A, T] -> [B, F, A, T]
        actor_vector_mask_time = actor_vector_a2l_mask.unsqueeze(1).expand(B, self.F, A, T)
        # assert actor_vector_mask_time.shape == (B, self.F, A, T)
        # [B, F, A, T] -> [B, F, T, A]
        actor_vector_mask_agent = actor_vector_mask_time.transpose(-1, -2)
        # assert actor_vector_mask_agent.shape == (B, self.F, T, A)

        # Layer S: [B, F, A, T, D] + [B, F, A, T, F] -> [F, A, T, D+F]
        # [F] (0,1,2,...) -> [F, F]
        one_hot = F.one_hot(torch.arange(self.F, device=actor_vector_r.device), num_classes=self.F)
        # assert one_hot.shape == (self.F, self.F)
        # [F, F] -> [1, F, 1, 1, F] -> [B, F, A, T, F]
        one_hot = one_hot.view(1, self.F, 1, 1, self.F).expand(B, -1, A, T, -1)
        # assert one_hot.shape == (B, self.F, A, T, self.F)
        # [B, F, A, T, D] + [B, F, A, T, F] -> [B, F, A, T, D+F]
        actor_vector_s = torch.cat([actor_vector_r, one_hot], dim=-1)
        # assert actor_vector_s.shape == (B, self.F, A, T, D + self.F)

        # Layer T: [B, F, A, T, D+F] -> [B, F, A, T, D]
        actor_vector_t = actor_vector_s
        for layer in self.t_layers_mlp:
            # [B, F, A, T, D+F], [B, F, A, T] -> [B, F, A, T, D]
            actor_vector_t = layer(actor_vector_t, actor_vector_mask_time)
            # assert actor_vector_t.shape == (B, self.F, A, T, D)

        # Layer U: self attention across time
        # [B, F, A, T, D]
        actor_vector_u = actor_vector_t
        for layer in self.u_layers_time:
            # [B, F, A, T, D], [B, F, A, T] -> [B, F, A, T, D]
            actor_vector_u = layer(actor_vector_u, actor_vector_mask_time)
            # assert actor_vector_u.shape == (B, self.F, A, T, D)

        # Layer V: self attention across agent
        # [B, F, A, T, D] -> [B, F, T, A, D]
        actor_vector_v = actor_vector_u.transpose(-2, -3).contiguous()
        for layer in self.v_layers_agent:
            # [B, F, T, A, D], [B, F, T, A] -> [B, F, T, A, D]
            actor_vector_v = layer(actor_vector_v, actor_vector_mask_agent)
            # assert actor_vector_v.shape == (B, self.F, T, A, D)
        # [B, F, T, A, D] -> [B, F, A, T, D]
        actor_vector_v = actor_vector_v.transpose(-2, -3).contiguous()

        # Layer W: self attention across time
        # [B, F, A, T, D]
        actor_vector_w = actor_vector_v
        for layer in self.w_layers_time:
            # [B, F, A, T, D], [B, F, A, T] -> [B, F, A, T, D]
            actor_vector_w = layer(actor_vector_w, actor_vector_mask_time)
            # assert actor_vector_w.shape == (B, self.F, A, T, D)

        # Layer X: self attention across agent
        # [B, F, A, T, D] -> [B, F, T, A, D]
        actor_vector_x = actor_vector_w.transpose(-2, -3).contiguous()
        for layer in self.x_layers_agent:
            # [B, F, T, A, D], [B, F, T, A] -> [B, F, T, A, D]
            actor_vector_x = layer(actor_vector_x, actor_vector_mask_agent)
            # assert actor_vector_x.shape == (B, self.F, T, A, D)
        # [B, F, T, A, D] -> [B, F, A, T, D]
        actor_vector_x = actor_vector_x.transpose(-2, -3).contiguous()

        # [B, F, A, T, D] -> [B, F, A, T, D]
        actor_vector_y = self.y_layer_norm(actor_vector_x)
        # assert actor_vector_y.shape == (B, self.F, A, T, D)

        # Layer Z1: [B, F, A, T, D] -> [B, F, A-1, 1]
        # [B, F, A, T, D] -> [B, F, A-1, D]
        actor_vector_tmp = actor_vector_y[:, :, :-1, -1, :]
        # [B, F, A, T, D] -> [B, F, 1, D] -> [B, F, A-1, D]
        actor_vector_tmp1 = actor_vector_y[:, :, -2:-1, -1, :].expand(-1, -1, actor_vector_tmp.shape[2], -1)
        # [B, F, A-1, D] + [B, F, A-1, D] -> [B, F, A-1, 2*D]
        actor_vector_z1 = torch.cat([actor_vector_tmp, actor_vector_tmp1], dim=-1)
        assert actor_vector_z1.shape == (B, self.F, A - 1, 2 * D)
        # [B, F, A, T] -> [B, F, A] -> [B, F, A-1]
        actor_vector_z1_mask = torch.where(torch.sum(actor_vector_mask_time, dim=-1, keepdim=False) > 0, 1,
                                           0)[:, :, :-1]
        assert actor_vector_z1_mask.shape == (B, self.F, A - 1)
        for layer in self.z1_layers_mlp:
            # [B, F, A-1, 2*D], [B, F, A-1] -> [B, F, A-1, 1]
            actor_vector_z1 = layer(actor_vector_z1, actor_vector_z1_mask)
        assert actor_vector_z1.shape == (B, self.F, A - 1, 1)

        # Layer Z2: [B, F, A, T, D] -> [B, F, A-1, T-1, 2]
        # [B, F, A, T, D] -> [B, F, A-1, T-1, D]
        actor_vector_z2 = actor_vector_y[:, :, :-1, :-1, :].contiguous()
        # [B, F, A, T] -> [B, F, A-1, T-1]
        actor_vector_z2_mask = actor_vector_mask_time[:, :, :-1, :-1]
        for layer in self.z2_layers_mlp:
            # [B, F, A-1, T-1, D], [B, F, A-1, T-1] -> [B, F, A-1, T-1, 2]
            actor_vector_z2 = layer(actor_vector_z2, actor_vector_z2_mask)
        assert actor_vector_z2.shape == (B, self.F, A - 1, T - 1, 2)

        # [B, F, A-1, T-1, 2], [B, F, A-1, 1]
        return actor_vector_z2, actor_vector_z1
