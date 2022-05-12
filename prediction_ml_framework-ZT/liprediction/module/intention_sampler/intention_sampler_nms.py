# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
import torch.nn.functional as F
from module.intention_sampler.intention_sampler_base import IntentionSamplerBase
from torch import linalg as LA


class IntentionSamplerNMS(IntentionSamplerBase):

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config

    def forward(self, cand_target_points_score, cand_target_points):
        '''
        cand_target_points_score is raw target score
        cand_target_points_score.shape: [B, S, 1]
        cand_target_points.shape: [B, S, 2]
        '''
        # candidate target number must be larger than the modality number
        assert self.config['modality_num'] <= cand_target_points_score.shape[1]

        # [B, S, 1] -> [B, S] -> normlize score
        cand_target_points_prob = F.softmax(cand_target_points_score.squeeze(-1), dim=-1)

        # [B, S] -> [B, S], descending order
        _, sort_idx = torch.sort(cand_target_points_prob, dim=-1, descending=True)

        # allocate zeros tensor [B, modality_num, 2]
        select_points = torch.zeros(
            (cand_target_points.shape[0], self.config['modality_num'], cand_target_points.shape[-1]),
            dtype=torch.float32)
        # allocate zeros tensor [B, modality_num, 1]
        select_points_prob = torch.zeros(
            (cand_target_points.shape[0], self.config['modality_num'], cand_target_points_score.shape[-1]),
            dtype=torch.float32)
        # init tensor use type_as in lightning framework, which can scale to any arbotrary number of GPUs
        select_points = select_points.type_as(cand_target_points_score)
        select_points_prob = select_points_prob.type_as(cand_target_points_score)

        # batch NMS for each sample, sample_sort_idx.shape: [S]
        for sample_i, sample_sort_idx in enumerate(sort_idx):
            select_points_num = 0

            # using the range function can speed up for-loop by a lot
            for i in range(len(sample_sort_idx)):
                idx = sample_sort_idx[i]
                # select top i candidate target points xy, select_target_point.shape : [2], which [0] is x, [1] is y
                select_target_point = cand_target_points[sample_i, idx, :]
                # select target points probability, select_target_prob.shape : [1]
                select_target_prob = cand_target_points_prob[sample_i, idx]

                if select_points_num > 0:
                    # [select_points_num, 2] - [2] -> [select_points_num, 2]
                    dist_select_target = select_points[sample_i, :select_points_num, :] - select_target_point
                    # [select_points_num, 2] -> [select_points_num], calculate euclidean distance for each target point
                    dist_select_target = LA.vector_norm(dist_select_target, ord=2, dim=-1)
                    # if any dist < threashold, skip
                    if torch.any(dist_select_target.lt(self.config['nms_threshold'])):
                        continue

                # [B, modality_num, 2] append target point -> [B, modality_num, 2]
                select_points[sample_i, select_points_num, :] = select_target_point
                # [B, modality_num, 1] append target prob -> [B, modality_num, 1]
                select_points_prob[sample_i, select_points_num, :] = select_target_prob

                select_points_num += 1
                if select_points_num >= self.config['modality_num']:
                    break

        # [B, modality_num, 2], [B, modality_num, 1]
        return select_points, select_points_prob


if __name__ == '__main__':
    config = {'modality_num': 6, 'nms_threshold': 0.1}
    sampler = IntentionSamplerNMS(config)

    cand_target_points_score = torch.rand(size=(2, 100, 1), dtype=torch.float32)
    cand_target_points = torch.rand(size=(2, 100, 2), dtype=torch.float32)

    select_points, select_points_prob = sampler.forward(cand_target_points_score, cand_target_points)

    print(select_points.shape)
    print(select_points_prob.shape)
