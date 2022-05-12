# Copyright (c) 2021 Li Auto Company. All rights reserved.
from abc import ABCMeta
from abc import abstractmethod

import torch.nn as nn


class IntentionSamplerBase(nn.Module, metaclass=ABCMeta):
    '''
        IntentionSamplerBase: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, pix_score, pix_offset):
        raise Exception('Not implemented')
