# Copyright (c) 2021 Li Auto Company. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod

import torch.nn as nn


class ContextEncoderBase(nn.Module, metaclass=ABCMeta):
    '''
        ContextEncoderBase: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, vector, vector_mask, cluster_mask):
        raise Exception('Not implemented')
