# Copyright (c) 2021 Li Auto Company. All rights reserved.

from .intention_sampler_imagecandi import IntentionSamplerImageCandi
from .intention_sampler_nms import IntentionSamplerNMS


class IntentionSampler():
    '''
        IntentionSampler: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'ImageCandi':
                return IntentionSamplerImageCandi(config)
            elif config['type'] == 'NMS':
                return IntentionSamplerNMS(config)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
