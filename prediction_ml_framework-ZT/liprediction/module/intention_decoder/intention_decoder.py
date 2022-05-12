# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch.nn as nn

from .intention_decoder_dense_tnt import IntentionDecoderDenseTNT
from .intention_decoder_image_upsampler import IntentionDecoderImageUpsampler
from .intention_decoder_multi_path_pro import IntentionDecoderMultiPathPro
from .intention_decoder_fsd_cutin import IntentionDecoderFSDCutin


class IntentionDecoder(nn.Module):
    '''
        IntentionDecoder: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'IntentionDecoderImageUpsampler':
                return IntentionDecoderImageUpsampler(config)
            elif config['type'] == 'IntentionDecoderDenseTNT':
                return IntentionDecoderDenseTNT(config)
            elif config['type'] == 'IntentionDecoderMultiPathPro':
                return IntentionDecoderMultiPathPro(config)
            elif config['type'] == 'IntentionDecoderFSDCutin':
                return IntentionDecoderFSDCutin(config)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
