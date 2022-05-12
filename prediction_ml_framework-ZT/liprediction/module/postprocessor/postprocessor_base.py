# Copyright (c) 2021 Li Auto Company. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod

import torch.nn as nn


class PostprocessorBase(nn.Module, metaclass=ABCMeta):
    '''
        PreprocessorBase: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, pred_traj, agent_pose):
        '''
            pred_traj: [B, K, L, 2] = [B, candi_num, predict_traj_num, 2]
            agent_pose: [B, 4]
            return: [B, C, L, 2]
        '''
        raise Exception('Not implemented')
