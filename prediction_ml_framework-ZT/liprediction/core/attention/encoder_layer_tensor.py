# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch.nn as nn

from .cross_attention_tensor import BipartiteAttention
from .linear_residual_block import LinearResidualBlock
from .self_attention_tensor import SelfAttention


class SelfAttentionEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention = SelfAttention.Create(config['self_atten'])

        self.out_ff = LinearResidualBlock(config['ff']['in_size'],
                                          config['ff']['out_size'],
                                          config['ff']['hidden_size'],
                                          norm_type='LN')

    def forward(self, x, mask=None):
        '''
        Args:
            x: shape=[..., L, F]
            mask: shape=[..., L]

        Return:
            out: [..., L, out_size]
        '''

        # [..., L, F], [..., L] -> [..., L, val_size], [..., head_num, L(q), L(k)]
        att_out, _ = self.attention(x, mask)
        # [..., L, val_size] -> [...*L, val_size] -> [...*L, out_size]
        ff_out = self.out_ff(att_out.view(-1, att_out.shape[-1]))

        # [..., L] + [-1] -> [..., L, -1]
        new_shape = x.shape[:-1] + (-1,)
        # [...*L, out_size] -> [..., L, out_size]
        out = ff_out.view(*new_shape)
        assert out.shape == (x.shape[:-1] + (ff_out.shape[-1],))

        return out


class CrossAttentionEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.do_self_atten = True if 'self_atten' in config else False

        if self.do_self_atten:
            self.self_atten = SelfAttention.Create(config['self_atten'])

        self.cross_attention = BipartiteAttention.Create(config['cross_atten'])

        self.out_ff = LinearResidualBlock(config['ff']['in_size'],
                                          config['ff']['out_size'],
                                          config['ff']['hidden_size'],
                                          norm_type='LN')

    def forward(self, x_k, x_q, x_k_mask=None, x_q_mask=None):
        '''
        Args:
            x_k: shape=[..., L_kv, F_kv]
            x_q: shape=[..., L_q, F_q]
            x_k_mask: shape=[..., L_kv]
            x_q_mask: shape=[..., L_q]

        Return:
            out: [..., L_q, out_size]
        '''

        if self.do_self_atten:
            # [..., L_q, F_q] -> [..., L_q, F_q]
            x_q, _ = self.self_atten(x_q, x_q_mask)

        # [..., L_kv, F_kv], [..., L_q, F_q], [..., L_kv] [..., L_q]
        # -> [..., L_q, val_size], [..., head_num, L_q, L_kv]
        att_out, _ = self.cross_attention(x_k, x_q, x_k_mask, x_q_mask)
        # [..., L_q, val_size] -> [...*L_q, val_size] -> [...*L_q, out_size]
        ff_out = self.out_ff(att_out.view(-1, att_out.shape[-1]))

        # [..., L_q] + [-1] -> [..., L_q, -1]
        shape = x_q.shape[:-1] + (-1,)
        # [...*L_q, out_size] -> [..., L_q, out_size]
        out = ff_out.view(*shape)
        assert out.shape == x_q.shape[:-1] + (ff_out.shape[-1],)

        return out
