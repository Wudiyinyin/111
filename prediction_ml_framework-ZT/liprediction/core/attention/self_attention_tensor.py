# Copyright (c) 2021 Li Auto Company. All rights reserved.

import math
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module, metaclass=ABCMeta):
    '''
    SelfAttention: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'DotProdMultiHead':
                return SelfAttentionDotProdMultiHead(config)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')

    @abstractmethod
    def forward(self, x, mask=None):
        raise Exception('Not implemented')


class SelfAttentionDotProdMultiHead(SelfAttention):
    '''
    SelfAttention: Tensor version
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.merged_by_mlp = False
        self.in_size = config['in_size']
        self.out_size = config['out_size']
        self.head_num = config['head_num']
        assert self.out_size % self.head_num == 0
        self.key_size = self.out_size // self.head_num
        self.val_size = self.out_size // self.head_num

        self.mask_wo_fill = True if 'mask_wo_fill' in self.config and self.config['mask_wo_fill'] else False 
        self.min_value = torch.finfo(torch.float32).min / 2.0
        self.wo_reize = True if 'wo_resize' in self.config and self.config['wo_resize'] else False
        if self.wo_reize:
            self.bypass_linear = nn.Linear(self.in_size, self.out_size)         

        self.lin_k = nn.ModuleList()
        self.lin_v = nn.ModuleList()
        self.lin_q = nn.ModuleList()
        for i in range(self.head_num):
            self.lin_k.append(nn.Linear(self.in_size, self.key_size))
            self.lin_v.append(nn.Linear(self.in_size, self.val_size))
            self.lin_q.append(nn.Linear(self.in_size, self.key_size))
        if self.merged_by_mlp:
            # merged by mlp
            self.lin_o = nn.ModuleList()
            for i in range(self.head_num):
                self.lin_o.append(nn.Linear(self.val_size, self.out_size))
        else:
            # merged by concat
            self.lin_o = nn.Linear(self.out_size, self.out_size)

        self.layer_norm = nn.LayerNorm(self.out_size)

    def forward(self, x, mask=None):
        '''
        Args:
          x: shape=[..., L, F]
          mask: shape=[..., L]

        Return:
          out: [..., L, out_size]
          atten: [..., head_num, L, L]
        '''

        head_values = []
        head_attens = []
        for i in range(self.head_num):
            # L = L_kv = L_q
            # [..., L_kv, F] -> [..., L_kv, key_size]
            k = self.lin_k[i](x)
            # [..., L_kv, F] -> [..., L_kv, val_size]
            v = self.lin_v[i](x)
            # [..., L_q, F] -> [..., L_q, key_size]
            q = self.lin_q[i](x)

            # k:[..., L_kv, key_size] -> [..., key_size, L_kv]
            # [..., L_q, key_size] * [..., key_size, L_kv] -> [..., L_q, L_kv]
            score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.key_size)
            # score = torch.matmul(q/math.sqrt(self.key_size), k.transpose(-1, -2))
            # TODO(xlm) excpet for rescale by key_size, we can also rescale by parameter as follow:
            # (ref: <<scene transformer>> Table 5)
            # self.scale = nn.Parameter(torch.FloatTensor([key_size]).fill_(init_value)), self.scale * q
            if mask is not None:
                # set mask to inf (along L_kv axis), decide which key is ignore(not interact)
                # mask: [..., L_kv] 0/1 -> [..., 1, L_kv] True/False
                # masked_fill: [..., L_q, L_kv], [..., 1, L_kv] -> [..., L_q, L_kv], 0->-inf, softmax(-inf)->0
                if self.mask_wo_fill:
                    inverse_mask = (1.0 - mask) * self.min_value
                    score = score + inverse_mask                
                else:                
                    score = score.masked_fill((mask == 0).unsqueeze(-2), self.min_value)

            # [..., L_q, L_kv] -> [..., L_q, L_kv]
            atten = F.softmax(score, dim=-1)

            # [..., L_q, L_kv] * [..., L_kv, val_size] -> [..., L_q, val_size]
            value = torch.matmul(atten, v)
            if self.merged_by_mlp:
                # [..., L_q, val_size] -> [..., L_q, out_size]
                value = self.lin_o[i](value)

            # [head_num, [..., L_q, out_size]]
            head_values.append(value)
            # [head_num, [..., L_q, L_kv]]
            head_attens.append(atten)

        # atten: [head_num, [..., L_q, L_kv]] -> [..., head_num, L_q, L_kv]
        atten = torch.stack(head_attens, dim=-3)
        # value: [..., L_q, out_size]
        if self.merged_by_mlp:
            # [head_num, [..., L_q, out_size]] -> [..., L_q, out_size, head_num] -> [..., L_q, out_size]
            value = torch.sum(torch.stack(head_values, dim=-1), dim=-1, keepdim=False)
        else:
            # [head_num, [..., L_q, val_size]] -> [..., L_q, out_size] -> [..., L_q, out_size]
            value = self.lin_o(torch.cat(head_values, dim=-1))

        # flatten to do skip connect
        prev_sizes = x.shape[:-2]
        L_q, feat_size = x.shape[-2:]
        prev_sizes_ = value.shape[:-2]
        L_q_, out_size = value.shape[-2:]
        assert L_q == L_q_
        assert prev_sizes == prev_sizes_

        x_tmp = x.view(-1, L_q, feat_size)
        # output shape maybe different with input, so interpolate is needed
        # x: [..., L_q, F] -> [N, L_q, F] -> [N, L_q, out_size] -> [N, L_q, out_size]
        # Currently temporal, spatial and volumetric sampling are supported,
        # i.e. expected inputs are 3-D, 4-D or 5-D in shape.
        if self.wo_reize:
            bypass = self.bypass_linear(x_tmp)
        else:        
            bypass = F.interpolate(x_tmp, size=self.out_size, mode='nearest')
        # [..., L_q, out_size] -> [N, L_q, out_size]
        value = value.view(-1, L_q, out_size)
        # [N, L_q, out_size] + [N, L_q, out_size] -> [N, L_q, out_size]
        out = self.layer_norm(bypass + value)
        # [N, L_q, out_size] -> [..., L_q, out_size]
        out = out.view(*(prev_sizes + (L_q, out_size)))

        # [..., L_q, out_size], [..., head_num, L_q, L_kv]
        return out, atten
