# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn
from core.attention.encoder_layer_tensor import CrossAttentionEncoderLayer
from core.image.image_upsample import ImageUpsampleLayer

from .intention_decoder_base import IntentionDecoderBase


class IntentionDecoderImageUpsampler(IntentionDecoderBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # pix embeding
        self.pix_embeding = nn.Sequential(
            nn.Linear(config['pix_embeding']['in_size'], config['pix_embeding']['hidden_size']), nn.ReLU(inplace=True),
            nn.Linear(config['pix_embeding']['hidden_size'], config['pix_embeding']['out_size']),
            nn.LayerNorm(config['pix_embeding']['out_size']), nn.ReLU(inplace=True))

        # decoder layers
        self.decoder_layers = nn.ModuleList(
            [CrossAttentionEncoderLayer(cfg) for cfg in config['intention_decoder_layers']])

        # dropout
        self.dropout = nn.Dropout2d(p=config['intention_dropout'])

        # upsample to high reso
        self.upsample_layers = nn.ModuleList([ImageUpsampleLayer(cfg) for cfg in config['upsample_layers']])

        # score prediction
        self.intention_prediction = nn.Sequential(
            nn.Conv2d(config['intention_prediction']['in_size'],
                      config['intention_prediction']['hidden_size'],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['intention_prediction']['hidden_size'],
                      config['intention_prediction']['out_size'],
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )

    def forward(self, cluster_feature, cluster_mask):
        '''
            cluster_feature: [B, C, F_e]
            cluster_mask: [B, C]
            construct pix feature
        '''
        # step1: encoding target point(position embedding vs encode with agent)
        # [B, low_dim*low_dim, 2]
        pix_embeding = self.genPixEmbeding(cluster_feature.shape[0], self.config['low_dim'], cluster_feature.device)
        # [B, low_dim*low_dim, 2] -> [B, low_dim*low_dim, F_pix]
        pix_feature = self.pix_embeding(pix_embeding)

        # step2: decoding pix feature using bipartite attention
        for decoder_layer in self.decoder_layers:
            # cluster_feature: [B, C, F_e]
            # pix_feature: [B, low_dim*low_dim, F_pix]
            # cluster_mask: [B, C]
            # [B, low_dim*low_dim, F_pix] -> [B, low_dim*low_dim, F_pix_atten_env]
            pix_feature = decoder_layer(cluster_feature, pix_feature, cluster_mask)

        # permute pix_feature to image
        # [B, low_dim*low_dim, F_pix_atten_env] -> [B, low_dim, low_dim, F_pix_atten_env]
        # -> [B, F_pix_atten_env, low_dim, low_dim]
        pix_feature = pix_feature.view(cluster_feature.shape[0], self.config['low_dim'], self.config['low_dim'],
                                       -1).permute(0, 3, 1, 2)

        # do dropout
        # [B, F_pix_atten_env, low_dim, low_dim]
        pix_feature = self.dropout(pix_feature)

        # upsample to high resolution
        # [B, F_pix_atten_env, low_dim, low_dim] -> [B, out_c, high_dim, high_dim]
        for layer in self.upsample_layers:
            pix_feature = layer(pix_feature)

        # step3: decode the pix score (Conv2d vs MLP)
        # TODO cat with agent_feature?
        # [B, out_c, high_dim, high_dim] -> [B, 3, high_dim, high_dim]
        pix_feature = self.intention_prediction(pix_feature)
        # -> [B, high_dim, high_dim]
        pix_score = pix_feature[:, 0, ...]
        # -> [B, 2, high_dim, high_dim]
        pix_offset = pix_feature[:, [1, 2], ...]
        # [B, high_dim, high_dim], [B, 2, high_dim, high_dim]
        return pix_score, pix_offset

    def genPixEmbeding(self, b, d, device='cpu'):
        # corner_coord -> center_coord:
        #   [0,1,2,...,d-1] - ((d-1)/2) -> [-(d-1)/2, ..., +(d-1)/2]
        # center_coord -> normalized_coord:
        #   [-(d-1)/2, ..., +(d-1)/2] ->"/(d/2)"-> [-1, ..., +1]
        cord = 2 * (torch.arange(d, device=device) - (d - 1) / 2) / d
        # [d] -> [1, d] -> [d, d]  ([-, ..., +])
        x_embeding = cord.expand(d, d)  # cord.unsqueeze(0).expand(d, d)
        # [d] -> [d, 1] -> [d, d] ([-, ..., +]^T)
        y_embeding = cord.unsqueeze(1).expand(d, d)
        # [[d,d],[d,d]] -> [d,d,2] -> [b,d,d,2] -> [b, d*d, 2]
        pix_embeding = torch.stack([x_embeding, y_embeding], dim=-1).expand(b, d, d, 2).reshape(b, -1, 2)
        return pix_embeding
