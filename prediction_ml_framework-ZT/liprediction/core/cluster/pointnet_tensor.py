# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn


class ClusterLayer(nn.Module):
    '''
    PointNet: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mask_wo_fill = True if 'mask_wo_fill' in self.config and self.config['mask_wo_fill'] else False 
        self.min_value = torch.finfo(torch.float32).min / 2.0

        self.mlp = nn.Sequential(
            nn.Linear(config['in_size'], config['hidden_size']),
            nn.LayerNorm(config['hidden_size']),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mask=None):
        '''
            x: shape=[B, C, L, F] or [N, L, F]
            mask: shape=[B, C, L] or [N, L]

            return: [B, C, L, 2*F'], [B, C, F']
        '''
        # [B, C, L, F] -> [B, C, L, F']
        y = self.mlp(x)
        # [B, C, L, F']
        y_masked = y
        if mask is not None:
            # Fills elements of self tensor with value where mask is True
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            # mask: [B, C, L] 0,1 -> [B, C, L] True,False -> [B, C, L, 1] True,False
            if self.mask_wo_fill:
                inverse_mask = (1.0 - mask) * self.min_value
                y_masked = y + inverse_mask 
            else:
                # masked_fill: [B, C, L, F'] -> [B, C, L, F'], mlp has relu, so min value of y is 0
                y_masked = y.masked_fill((mask == 0).unsqueeze(-1), 0)
        # max: [B, C, L, F'] -> [B, C, F'], maxpool along L
        g = torch.max(y_masked, dim=-2)[0]
        # g: [B, C, F'] -> [B, C, 1, F'] -> [B, C, L, F']
        # cat: [B, C, L, F'], [B, C, L, F'] -> [B, C, L, 2*F']
        out = torch.cat([y, g.unsqueeze(-2).expand(y.shape)], dim=-1)
        # [B, C, L, 2*F'], [B, C, F']
        return out, g
