# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer, SelfAttentionEncoderLayer
from core.cluster.mlp_layer import MLPLayer
from core.cluster.pointnet_tensor import ClusterLayer

from .context_encoder_base import ContextEncoderBase


class ContextEncoderSceneTransformer(ContextEncoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # encode actor and lane
        self.actor_encoder_layers = nn.ModuleList([MLPLayer(cfg) for cfg in config['actor_encoder_layers']])
        self.lane_cluster_layers = nn.ModuleList([ClusterLayer(cfg) for cfg in config['lane_cluster_layers']])

        # context transformer encoders
        self.d_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['d_layers_time']])
        self.e_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['e_layers_agent']])
        self.f_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['f_layers_time']])
        self.g_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['g_layers_agent']])
        self.h_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['h_layers_time']])
        self.i_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['i_layers_agent']])

        self.j_layers_agent2staticrg_time = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['j_layers_agent2staticrg_time']])
        # self.k_layers_agent2dynamicrg_time = nn.ModuleList(
        #     [CrossAttentionEncoderLayer(cfg) for cfg in config['k_layers_agent2dynamicrg_time']])

        self.l_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['l_layers_time']])
        self.m_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['m_layers_agent']])

        self.n_layers_agent2staticrg_time = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['n_layers_agent2staticrg_time']])
        # self.o_layers_agent2dynamicrg_time = nn.ModuleList(
        #     [CrossAttentionEncoderLayer(cfg) for cfg in config['o_layers_agent2dynamicrg_time']])

        self.p_layers_time = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['p_layers_time']])
        self.q_layers_agent = nn.ModuleList([SelfAttentionEncoderLayer(cfg) for cfg in config['q_layers_agent']])

    def forward(self, vectors, vectors_padding_mask, vectors_hidden_mask=None):
        '''

        Args:
            vectors: list
                [0] actor vector [B, A, T, Da], where A=Actor T=Time
                [1] lane  vector [B, G, L, Dg], where G=laneGraph, L=Length
            vector_masks: list
                [0] actor vector mask [B, A, T]
                [1] lane vector mask [B, G, L]
        Return [B, A, T, D]
        '''

        # [B, A, T, Da]
        actor_vector = vectors[0]
        # [B, G, L, Dg]
        lane_vector = vectors[1]
        # [B, A, T]
        actor_vector_mask = vectors_padding_mask[0]
        # [B, G, L]
        lane_vector_mask = vectors_padding_mask[1]
        # [B, A, T]
        actor_vector_hidden_mask = vectors_hidden_mask[0]
        # [B, G, L]
        # lane_vector_hidden_mask = vectors_hidden_mask[1]
        B, A, T, Da = actor_vector.shape
        B, G, L, Dg = lane_vector.shape
        # assert actor_vector_mask.shape == (B, A, T)
        # assert lane_vector_mask.shape == (B, G, L)
        # assert actor_vector_hidden_mask.shape == (B, A, T)
        # assert lane_vector_hidden_mask.shape == (B, G, L)

        # # handle hidden mask
        # # [B, A, T] 0/1 -> [B, A, T, 1] 0/1 -> [B, A, T, Da-2] 0/1  1:valid 0:mask
        # mask1 = actor_vector_hidden_mask.unsqueeze(-1).expand(-1, -1, -1, Da - 2)
        # # [B, A, T, 2] 1, the last two feature should not be hidden
        # mask2 = torch.ones((B, A, T, 2), dtype=torch.int32).type_as(mask1)
        # # [B, A, T, Da-2] + [B, A, T, 2] -> [B, A, T, Da]
        # hidden_mask = torch.cat((mask1, mask2), dim=-1)
        # assert hidden_mask.shape == (B, A, T, Da)
        # # [B, A, T, Da], [B, A, T, Da] -> [B, A, T, Da], set hidden element to 0
        # actor_vector = actor_vector.masked_fill((hidden_mask == 0), 0.0)

        # Layer A: encode Actor [B, A, T, Da] -> [B, A, T, D]
        for layer in self.actor_encoder_layers:
            # [B, A, T, Da], [B, A, T] -> [B, A, T, D]
            actor_vector = layer(actor_vector, actor_vector_mask)
        D = actor_vector.shape[-1]
        # assert actor_vector.shape == (B, A, T, D)

        # Layer C: encode Lane Graph  [B, G, L, Dg] -> [B, G, D] -> [B, G, T, D]
        for layer in self.lane_cluster_layers:
            # [B, G, L, Dg], [B, G, L] -> [B, G, L, D+D], [B, G, D]
            lane_vector, lane_cluster_vector = layer(lane_vector, lane_vector_mask)
            # assert lane_vector.shape == (B, G, L, D + D)
            # assert lane_cluster_vector.shape == (B, G, D)

        # [B, A, T]
        actor_vector_mask_time = actor_vector_mask
        assert actor_vector_mask_time.shape == (B, A, T)
        # [B, A, T] -> [B, T, A]
        actor_vector_mask_agent = actor_vector_mask.transpose(-1, -2)
        assert actor_vector_mask_agent.shape == (B, T, A)

        # Layer D: self attention across time
        # [B, A, T, D]
        actor_vector_d = actor_vector
        for layer in self.d_layers_time:
            # [B, A, T, D], [B, A, T] -> [B, A, T, D]
            actor_vector_d = layer(actor_vector_d, actor_vector_mask_time)
            # assert actor_vector_d.shape == (B, A, T, D)

        # Layer E: self attention across agent
        # [B, A, T, D] -> [B, T, A, D]
        actor_vector_e = actor_vector_d.transpose(-2, -3).contiguous()
        for layer in self.e_layers_agent:
            # [B, T, A, D], [B, T, A] -> [B, T, A, D]
            actor_vector_e = layer(actor_vector_e, actor_vector_mask_agent)
            # assert actor_vector_e.shape == (B, T, A, D)
        # [B, T, A, D] -> [B, A, T, D]
        actor_vector_e = actor_vector_e.transpose(-2, -3).contiguous()

        # Layer F: self attention across time
        # [B, A, T, D]
        actor_vector_f = actor_vector_e
        for layer in self.f_layers_time:
            # [B, A, T, D], [B, A, T] -> [B, A, T, D]
            actor_vector_f = layer(actor_vector_f, actor_vector_mask_time)
            # assert actor_vector_f.shape == (B, A, T, D)

        # Layer G: self attention across agent
        # [B, A, T, D] -> [B, T, A, D]
        actor_vector_g = actor_vector_f.transpose(-2, -3).contiguous()
        for layer in self.g_layers_agent:
            # [B, T, A, D], [B, T, A] -> [B, T, A, D]
            actor_vector_g = layer(actor_vector_g, actor_vector_mask_agent)
            # assert actor_vector_g.shape == (B, T, A, D)
        # [B, T, A, D] -> [B, A, T, D]
        actor_vector_g = actor_vector_g.transpose(-2, -3).contiguous()

        # Layer H: self attention across time
        # [B, A, T, D]
        actor_vector_h = actor_vector_g
        for layer in self.h_layers_time:
            # [B, A, T, D], [B, A, T] -> [B, A, T, D]
            actor_vector_h = layer(actor_vector_h, actor_vector_mask_time)
            # assert actor_vector_h.shape == (B, A, T, D)

        # Layer I: self attention across agent
        # [B, A, T, D] -> [B, T, A, D]
        actor_vector_i = actor_vector_h.transpose(-2, -3).contiguous()
        for layer in self.i_layers_agent:
            # [B, T, A, D], [B, T, A] -> [B, T, A, D]
            actor_vector_i = layer(actor_vector_i, actor_vector_mask_agent)
            # assert actor_vector_i.shape == (B, T, A, D)
        # [B, T, A, D] -> [B, A, T, D]
        actor_vector_i = actor_vector_i.transpose(-2, -3).contiguous()

        # Add extra agent/time dimention
        # [B, A, T, D] -> [B, A+1, T+1, D]
        # [B, A, T, D] -> [B, A, 1, D]
        actor_vector_time_mean = torch.mean(actor_vector_i, dim=-2, keepdim=True)
        # [B, A, T, D] + [B, A, 1, D] -> [B, A, T+1, D]
        actor_vector_extend = torch.cat([actor_vector_i, actor_vector_time_mean], dim=-2)
        # [B, A, T+1, D] -> [B, 1, T+1, D]
        actor_vector_agent_mean = torch.mean(actor_vector_extend, dim=-3, keepdim=True)
        # [B, A, T+1, D] + [B, 1, T+1, D] -> [B, A+1, T+1, D]
        actor_vector_extend = torch.cat([actor_vector_extend, actor_vector_agent_mean], dim=-3)
        assert actor_vector_extend.shape == (B, A + 1, T + 1, D)

        # [B, A, 1]
        extra_mask_for_time = torch.ones((B, A, 1), dtype=torch.int32).type_as(actor_vector_mask)
        # [B, 1, T+1]
        extra_mask_for_agent = torch.ones((B, 1, T + 1), dtype=torch.int32).type_as(actor_vector_mask)
        # [B, A, T] + [B, A, 1] -> [B, A, T+1]
        actor_vector_mask_extend = torch.cat([actor_vector_mask, extra_mask_for_time], dim=-1)
        # [B, A, T+1] + [B, 1, T+1] -> [B, A+1, T+1]
        actor_vector_mask_extend = torch.cat([actor_vector_mask_extend, extra_mask_for_agent], dim=-2)
        # [B, A+1, T+1]
        actor_vector_mask_time = actor_vector_mask_extend
        assert actor_vector_mask_time.shape == (B, A + 1, T + 1)
        # [B, A+1, T+1] -> [B, T+1, A+1]
        actor_vector_mask_agent = actor_vector_mask_extend.transpose(-1, -2)
        assert actor_vector_mask_agent.shape == (B, T + 1, A + 1)

        # [B, G, D] -> [B, 1, G, D] -> [B, T+1, G, D]
        lane_cluster_vector = lane_cluster_vector.unsqueeze(-3).expand(-1, T + 1, -1, -1)
        assert lane_cluster_vector.shape == (B, T + 1, G, D)
        # [B, G, L] -> [B, G] n/0 -> [B, G] True/False -> [B, G] 1/0 1:valid 0:mask
        # -> [B, 1, G] -> [B, T+1, G]
        lane_cluster_vector_mask = (torch.sum(lane_vector_mask, dim=-1, keepdim=False) != 0).to(
            torch.int32).unsqueeze(-2).expand(-1, T + 1, -1)
        assert lane_cluster_vector_mask.shape == (B, T + 1, G)

        # Layer J: cross attention between lane(static RG) and actor across time
        # [B, A+1, T+1, D] -> [B, T+1, A+1, D]
        actor_vector_j = actor_vector_extend.transpose(-2, -3).contiguous()
        for layer in self.j_layers_agent2staticrg_time:
            # [B, T+1, G, D] [B, T+1, A+1, D],
            # [B, T+1, G]    [B, T+1, A+1]
            # -> [B, T+1, A+1, D]
            actor_vector_j = layer(lane_cluster_vector, actor_vector_j, lane_cluster_vector_mask,
                                   actor_vector_mask_agent)
            # assert actor_vector_j.shape == (B, T, A, D)
        # [B, T+1, A+1, D] -> [B, A+1, T+1, D]
        actor_vector_j = actor_vector_j.transpose(-2, -3).contiguous()

        # Layer K: cross attention between lane(dynamic RG) and actor across time
        # [B, A+1, T+1, D]
        actor_vector_k = actor_vector_j
        # for layer in self.k_layers_agent2staticrg_time:
        #     # [B, G, T, D] [B, A, T, D], [B, G, T] [B, A, T]
        #     actor_vector_k = layer(dynamic_lane_cluster_vector, actor_vector_k, dynamic_lane_cluster_vector_mask,
        #                            actor_vector_mask_time)

        # Layer L: self attention across time
        # [B, A+1, T+1, D]
        actor_vector_l = actor_vector_k
        for layer in self.l_layers_time:
            # [B, A+1, T+1, D], [B, A+1, T] -> [B, A+1, T+1, D]
            actor_vector_l = layer(actor_vector_l, actor_vector_mask_time)
            # assert actor_vector_l.shape == (B, A, T, D)

        # Layer M: self attention across agent
        # [B, A+1, T+1, D] -> [B, T+1, A+1, D]
        actor_vector_m = actor_vector_l.transpose(-2, -3).contiguous()
        for layer in self.m_layers_agent:
            # [B, T+1, A+1, D], [B, T+1, A+1] -> [B, T+1, A+1, D]
            actor_vector_m = layer(actor_vector_m, actor_vector_mask_agent)
            # assert actor_vector_m.shape == (B, T+1, A+1, D)
        # [B, T+1, A+1, D] -> [B, A+1, T+1, D]
        actor_vector_m = actor_vector_m.transpose(-2, -3).contiguous()

        # Layer N: cross attention between lane(static RG) and actor across time
        # [B, A+1, T+1, D] -> [B, T+1, A+1, D]
        actor_vector_n = actor_vector_m.transpose(-2, -3).contiguous()
        for layer in self.n_layers_agent2staticrg_time:
            # [B, T+1, G, D] [B, T+1, A+1, D],
            # [B, T+1, G]    [B, T+1, A+1]
            # -> [B, T+1, A+1, D]
            actor_vector_n = layer(lane_cluster_vector, actor_vector_n, lane_cluster_vector_mask,
                                   actor_vector_mask_agent)
            # assert actor_vector_n.shape == (B, T+1, A+1, D)
        # [B, T+1, A+1, D] -> [B, A+1, T+1, D]
        actor_vector_n = actor_vector_n.transpose(-2, -3).contiguous()

        # Layer O: cross attention between lane(dynamic RG) and actor across time
        # [B, A+1, T+1, D]
        actor_vector_o = actor_vector_n
        # for layer in self.o_layers_agent2staticrg_time:
        #     # [B, G, T, D] [B, A, T, D], [B, G, T] [B, A, T]
        #     actor_vector_o = layer(dynamic_lane_cluster_vector, actor_vector_o, dynamic_lane_cluster_vector_mask,
        #                            actor_vector_mask_time)

        # Layer P: self attention across time
        # [B, A, T, D]
        actor_vector_p = actor_vector_o
        for layer in self.p_layers_time:
            # [B, A+1, T+1, D], [B, A+1, T+1] -> [B, A+1, T+1, D]
            actor_vector_p = layer(actor_vector_p, actor_vector_mask_time)
            # assert actor_vector_p.shape == (B, A+1, T+1, D)

        # Layer Q: self attention across agent
        # [B, A+1, T+1, D] -> [B, T+1, A+1, D]
        actor_vector_q = actor_vector_p.transpose(-2, -3).contiguous()
        for layer in self.q_layers_agent:
            # [B, T+1, A+1, D], [B, T+1, A+1] -> [B, T+1, A+1, D]
            actor_vector_q = layer(actor_vector_q, actor_vector_mask_agent)
            # assert actor_vector_q.shape == (B, T+1, A+1, D)
        # [B, T+1, A+1, D] -> [B, A+1, T+1, D]
        actor_vector_q = actor_vector_q.transpose(-2, -3).contiguous()

        # [B, A+1, T+1, D], [B, A+1, T+1]
        assert actor_vector_q.shape == (B, A + 1, T + 1, D)
        assert actor_vector_mask_time.shape == (B, A + 1, T + 1)
        return actor_vector_q, actor_vector_mask_time
