# Copyright (c) 2021 Li Auto Company. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod

import torch.nn as nn


class PreprocessorBase(nn.Module, metaclass=ABCMeta):
    '''
        PreprocessorBase: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, vector, agent_pose):
        '''
            vector: [B, C, L, F]
            agent_pose: [B, 4]
            return: [B, C, L, F]
        '''
        raise Exception('Not implemented')
