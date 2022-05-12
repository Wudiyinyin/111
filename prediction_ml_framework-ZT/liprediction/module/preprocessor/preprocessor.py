# Copyright (c) 2021 Li Auto Company. All rights reserved.
from .preprocessor_agent_coord import PreprocessorAgentCoord


class Preprocessor():
    '''
        Preprocessor: Tensor version
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def Create(config):
        if 'type' in config:
            if config['type'] == 'AgentCoord':
                return PreprocessorAgentCoord(config)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
