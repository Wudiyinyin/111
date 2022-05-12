# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .intention_sampler_base import IntentionSamplerBase


class IntentionSamplerImageCandi(IntentionSamplerBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.maxpool2d = nn.MaxPool2d(kernel_size=config['kernel_size'],
                                      padding=int(config['kernel_size'] / 2),
                                      stride=1)

    def forward(self, pix_score, pix_offset):
        '''
        input pix_score is normlized to sum to 1.0
        pix_score.shape: [B, D, D]
        pix_offset.shape: [B, 2, D, D]
        '''

        B, D, dev = pix_score.shape[0], pix_score.shape[1], pix_score.device

        # normlize score
        # [B, D, D] -> [B, D*D] -> [B, D, D]
        pix_score = F.softmax(pix_score.view(B, -1), dim=-1).view(B, D, D)

        # smooth score
        # [B, D, D] -> [B, 1, D, D] -> [B, 1, D, D] -> [B, D, D]
        pix_score = F.conv2d(pix_score.unsqueeze(1),
                             torch.ones((1, 1, self.config['kernel_size'], self.config['kernel_size']),
                                        dtype=pix_score.dtype,
                                        device=dev),
                             padding=int(self.config['kernel_size'] / 2)).view(B, D, D)
        # [B, D, D] -> [B, 1, D, D] -> [B, 1, D, D] -> [B, D, D]
        pooled_score = self.maxpool2d(pix_score.unsqueeze(1)).view(B, D, D)

        # boost the local maximum with 1.0, so that the local maximum will always be selected
        # [B, D, D]
        pix_score = pix_score + \
            ((pooled_score == pix_score) & (pix_score > self.config['thresh']))
        # [B, D, D] -> [B, D*D] -> [B, candi_num], [B, candi_num]
        selected_pix_score, selected_pix_index = torch.topk(pix_score.view(B, -1), self.config['candi_num'], dim=-1)
        # [B] -> [B, 1]
        batch_idx = torch.arange(B).unsqueeze(-1)
        # [B, 2, D, D] -> [B, 2, D*D] -> [B, 2, candi_num]
        pix_offset = pix_offset.view(B, 2, -1)[batch_idx, :, selected_pix_index]
        # [B, 2, candi_num]
        selected_pix_intention = self.intentionFromIndex(selected_pix_index, pix_offset)
        # [B, 2, candi_num], [B, candi_num]
        return selected_pix_intention, selected_pix_score

    def intentionFromIndex(self, pix_idx, pix_offset):
        # pix_idx: [B, candi_num]
        # pix_offset: [B, 2, candi_num]
        # [B, candi_num]
        cord_x = torch.remainder(pix_idx, self.config['dim'])
        # [B, candi_num]
        cord_y = torch.floor_divide(pix_idx, self.config['dim'])
        # -> [B, candi_num, 2], corner_cord
        pix_cord = torch.stack([cord_x, cord_y], dim=-1)
        # -> [B, candi_num, 2], corner_cord -> center_cord
        pix_center = (pix_cord - (self.config['dim'] - 1) / 2) * self.config['reso']
        # [B, candi_num, 2] + [B, 2, candi_num] todo bug???
        return pix_center + pix_offset
