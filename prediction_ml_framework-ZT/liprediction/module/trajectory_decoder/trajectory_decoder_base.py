# Copyright (c) 2021 Li Auto Company. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod

import torch.nn as nn


class TrajectoryDecoderBase(nn.Module, metaclass=ABCMeta):
    '''
        TrajectoryDecoderBase: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, cluster_feature, cluster_mask, intention):
        raise Exception('Not implemented')
