# Copyright (c) 2022 Li Auto Company. All rights reserved.

import torch
from datasets.transform.agent_closure import AgentClosure


class DenseTNTTransform():

    def __init__(self, config):
        self.config = config

    def __call__(self, sample: AgentClosure):

        if self.config['mask_lt_8s_labels']:
            if torch.any(sample.gt_target_point_mask == 0):
                sample.gt_target_point_mask = torch.zeros_like(sample.gt_target_point_mask)
                sample.gt_traj_mask = torch.zeros_like(sample.gt_traj_mask)

        return sample
