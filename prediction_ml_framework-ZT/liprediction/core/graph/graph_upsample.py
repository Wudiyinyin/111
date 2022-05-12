"""
# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch.nn as nn
#from torch_scatter import scatter_min, scatter_softmax, scatter_max, scatter_log_softmax

class GraphUpsampleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, down_x, edge_down2up):
        '''
            down_x: [N, F], edge_down2up: [M, 2]
            e.g. 1->3
        '''
        # [N, F] -> [N', F]
        down_x = down_x[edge_down2up[0]]
        # [N', F], [N'] -> [M, F'] ??
        up_x = scatter_mean(down_x, edge_down2up[1], dim=0)
        return up_x
"""
