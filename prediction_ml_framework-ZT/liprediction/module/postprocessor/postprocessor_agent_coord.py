# Copyright (c) 2021 Li Auto Company. All rights reserved.

from core.utils.utils import pose2rt, local2global

from .postprocessor_base import PostprocessorBase


class PostprocessorAgentCoord(PostprocessorBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, pred_traj, agent_pose):
        '''
            pred_traj: [B, K, L, 2] = [B, candi_num, predict_traj_num, 2]
            agent_pose: [B, 4]
            return: [B, C, L, 2]
        '''
        # normlize vector
        agent_origin, agent_rotm = pose2rt(agent_pose)
        # convert trajectory to global frame
        normlized_pred_traj = local2global(pred_traj, agent_origin, agent_rotm)
        return normlized_pred_traj
