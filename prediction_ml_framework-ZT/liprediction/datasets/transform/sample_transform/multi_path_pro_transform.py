# Copyright (c) 2022 Li Auto Company. All rights reserved.

import torch
from datasets.processer.feature_processer import (ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y, ACTOR_FEATURE_START_X,
                                                  ACTOR_FEATURE_START_Y, ACTOR_FEATURE_VECTOR_HEADING,
                                                  ACTOR_FEATURE_VELOCITY_HDEADING, ACTOR_FEATURE_VELOCITY_X,
                                                  ACTOR_FEATURE_VELOCITY_Y)
from datasets.transform.agent_closure import AgentClosure


class MultiPathProTransform():

    def __init__(self, config):
        self.config = config

    def __call__(self, sample: AgentClosure):
        # [1, C, L, F]
        actor_vector = sample.actor_vector
        # [1, C, L]
        actor_vector_mask = sample.actor_vector_mask
        # [1, C]
        actor_cluster_mask = sample.actor_cluster_mask

        B, C, _, F = actor_vector.shape
        # [1, C, L, F] -> [1, C, L+1, F], add zeros tensor in the front of actor vector
        padding_zeros = torch.zeros((B, C, 1, F)).type_as(actor_vector)
        actor_vector_padding = torch.cat((padding_zeros, actor_vector), dim=-2)
        # select specific feature to get differential feature
        diff_fea_idx = [
            ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y, ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
            ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y,
            ACTOR_FEATURE_VELOCITY_HDEADING
        ]
        # [1, C, L, F_diff] - [1, C, L, F_diff] -> [1, C, L, F_diff](differential feature)
        actor_diff_vector_idx = actor_vector[:, :, :, diff_fea_idx] - actor_vector_padding[:, :, :-1, diff_fea_idx]

        # copy from actor vector and assign specific feature
        actor_diff_vector = actor_vector
        actor_diff_vector[:, :, :, diff_fea_idx] = actor_diff_vector_idx

        # remove the first unuseful diff tensor
        # TODO: zhankun@lixiang.com, this is not the correct dim to remove,
        # because the unusefule dim is the last agent sequence feature, such as -11
        # [1, C, L-1, F]
        sample.actor_diff_vector = actor_diff_vector[:, :, 1:, :]
        # [1, C, L-1]
        sample.actor_diff_vector_mask = actor_vector_mask[:, :, 1:]
        # [1, C]
        sample.actor_diff_cluster_mask = actor_cluster_mask

        if self.config['mask_lt_8s_labels']:
            # if any ground truth label is less than 8s, then set the corresponding label to 0
            if torch.any(sample.gt_target_point_mask == 0):
                sample.gt_target_point_mask = torch.zeros_like(sample.gt_target_point_mask)
                sample.gt_traj_mask = torch.zeros_like(sample.gt_traj_mask)

        return sample
