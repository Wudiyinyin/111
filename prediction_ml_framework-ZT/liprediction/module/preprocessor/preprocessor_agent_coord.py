# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
from core.utils.utils import pose2rt

from .preprocessor_base import PreprocessorBase


class PreprocessorAgentCoord(PreprocessorBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.rot_present = False
        if 'rot_present' in self.config and self.config['rot_present']:
            self.rot_present = True

    def forward(self, vector, agent_pose):
        '''
            vector: [B, C, L, F]
            agent_pose: [B, 4]
            return: [B, C, L, F]
        '''
        agent_origin, agent_rotm = pose2rt(agent_pose)

        # Handle list input vectors
        if isinstance(vector, list):
            normlized_vector = []
            for vector_i in vector:
                # normlize vector
                normlized_vector.append(
                    self.normlizeVector(vector_i, agent_origin, agent_rotm, rot_present=self.rot_present))
        else:
            # normlize vector
            normlized_vector = self.normlizeVector(vector, agent_origin, agent_rotm, rot_present=self.rot_present)

        return normlized_vector

    def normlizeVector(self, vector, origin, rotm, rot_present=False):
        '''
            vector: [B, C, L, F]
            origin: [B, 2]
            rotm: [B, 2, 2]
            return: [B, C, L, F]
        '''
        # [B, 1, 1, 2]
        origin = origin.unsqueeze(1).unsqueeze(1)
        # [B, 1, 2, 2]
        rotm = rotm.unsqueeze(1)
        # [B, Cluter?, Vector?, 2] - [B, 1, 1, 2] * [B, 1, 2, 2]  ???
        # 1) global->local: rotm^T  2) col_vector->row_vector: rotm^T^T = rotm
        vector_start = torch.matmul(vector[..., [0, 1]] - origin, rotm)
        vector_end = torch.matmul(vector[..., [2, 3]] - origin, rotm)
        # 1) R_map_agent^T * R_map_other = R_agent_other         2) col_vector -> row_vector
        # [a -b] ^T     * [a' -b']  => col_vector = [a -b]^T * [a'] => row_vector [a' b'] * [a -b]
        # [b  a]          [b'  a']                  [b  a]     [b']                         [b  a]
        if rot_present:
            vector_rot = torch.matmul(vector[..., [4, 5]], rotm)

        if rot_present:
            vector_remain = vector[..., 6:]
            normlized_vector = torch.cat([vector_start, vector_end, vector_rot, vector_remain], dim=-1)
        else:
            vector_remain = vector[..., 4:]
            normlized_vector = torch.cat([vector_start, vector_end, vector_remain], dim=-1)
        # [B, C, L, F]
        return normlized_vector
