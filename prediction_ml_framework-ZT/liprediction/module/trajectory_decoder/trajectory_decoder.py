# Copyright (c) 2021 Li Auto Company. All rights reserved.

from .trajectory_decoder_dense_tnt import TrajDecoderDenseTNT
from .trajectory_decoder_mlp import TrajDecoderMlp
from .trajectory_decoder_multi_path_decoder import TrajDecoderMultiPathDecoder
from .trajectory_decoder_multi_path_pro import TrajDecoderMultiPathPro
from .trajectory_decoder_scene_transformer import TrajDecoderSceneTransformer
from .trajectory_decoder_trajembed import TrajDecoderTrajEmbeding
from .trajectory_decoder_fsd import TrajDecoderFSD


class TrajectoryDecoder():
    '''
        TrajectoryDecoder: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'TrajDecoderTrajEmbeding':
                return TrajDecoderTrajEmbeding(config)
            elif config['type'] == 'TrajDecoderMlp':
                return TrajDecoderMlp(config)
            elif config['type'] == 'TrajDecoderDenseTNT':
                return TrajDecoderDenseTNT(config)
            elif config['type'] == 'TrajDecoderSceneTransformer':
                return TrajDecoderSceneTransformer(config)
            elif config['type'] == 'TrajDecoderMultiPathPro':
                return TrajDecoderMultiPathPro(config)
            elif config['type'] == 'TrajDecoderMultiPathDecoder':
                return TrajDecoderMultiPathDecoder(config)
            elif config['type'] == 'TrajDecoderFSD':
                return TrajDecoderFSD(config)                
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
