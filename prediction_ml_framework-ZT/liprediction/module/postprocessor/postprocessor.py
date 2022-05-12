# Copyright (c) 2021 Li Auto Company. All rights reserved.

from .postprocessor_agent_coord import PostprocessorAgentCoord


class Postprocessor():
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
                return PostprocessorAgentCoord(config)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
