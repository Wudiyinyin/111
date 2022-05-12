# Copyright (c) 2021 Li Auto Company. All rights reserved.

from .context_encoder_dense_tnt import ContextEncoderDenseTNT
from .context_encoder_multi_path_pro import ContextEncoderMultiPathPro
from .context_encoder_scene_transformer import ContextEncoderSceneTransformer
from .context_encoder_vectornet import ContextEncoderVectorNet
from .context_encoder_fsd import ContextEncoderFSD


class ContextEncoder():
    '''
        ContextEncoder: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'VectorNet':
                return ContextEncoderVectorNet(config)
            elif config['type'] == 'ContextEncoderDenseTNT':
                return ContextEncoderDenseTNT(config)
            elif config['type'] == 'ContextEncoderSceneTransformer':
                return ContextEncoderSceneTransformer(config)
            elif config['type'] == 'ContextEncoderMultiPathPro':
                return ContextEncoderMultiPathPro(config)
            elif config['type'] == 'ContextEncoderFSD':
                return ContextEncoderFSD(config)
            else:                
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
