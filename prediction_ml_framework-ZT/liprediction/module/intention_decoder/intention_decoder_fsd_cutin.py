# Copyright (c) 2022 Li Auto Company. All rights reserved.

from core.cluster.mlp_layer import MLPLayer
from core.cluster.res_mlp_layer import ResMLPLayer

from .intention_decoder_base import IntentionDecoderBase


class IntentionDecoderFSDCutin(IntentionDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.cutin_cls_res_layer = ResMLPLayer(config['cutin_cls_res_layer'])
        self.is_cutin_res_layer = ResMLPLayer(config['is_cutin_res_layer'])       

        self.cutin_cls_1s_layer = MLPLayer(config['cutin_cls_1s_layer'])
        self.cutin_cls_2s_layer = MLPLayer(config['cutin_cls_2s_layer'])
        self.cutin_cls_3s_layer = MLPLayer(config['cutin_cls_3s_layer'])

        self.is_cutin_1s_layer = MLPLayer(config['is_cutin_1s_layer'])
        self.is_cutin_2s_layer = MLPLayer(config['is_cutin_2s_layer'])
        self.is_cutin_3s_layer = MLPLayer(config['is_cutin_3s_layer'])        

    def forward(self, agent_all_feature):
        '''Intention decoder forward

        Args:
            agent_all_feature: [B, 1, F_c]
        '''

        # [[B, 1, F_c] -> [B, 1, F_c']
        cutin_cls_embedding = self.cutin_cls_res_layer(agent_all_feature)
        is_cutin_embedding = self.is_cutin_1s_layer(agent_all_feature)

        cutin_1s_score = self.cutin_cls_1s_layer(cutin_cls_embedding)
        cutin_2s_score = self.cutin_cls_2s_layer(cutin_cls_embedding)
        cutin_3s_score = self.cutin_cls_3s_layer(cutin_cls_embedding)  

        is_cutin_1s_score = self.is_cutin_1s_layer(is_cutin_embedding)
        is_cutin_2s_score = self.is_cutin_2s_layer(is_cutin_embedding)
        is_cutin_3s_score = self.is_cutin_3s_layer(is_cutin_embedding)                       

        cutin_cls_scores = [cutin_1s_score, cutin_2s_score, cutin_3s_score]
        is_cutin_scores = [is_cutin_1s_score, is_cutin_2s_score, is_cutin_3s_score]
        return cutin_cls_scores, is_cutin_scores
