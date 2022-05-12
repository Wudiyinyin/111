"""
# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch.nn as nn
# from torch_scatter import scatter_min, scatter_softmax, scatter_max, scatter_log_softmax

#from core.cluster.pointnet_tensor import ClusterLayer

class GraphDownsampleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.cluster_layers = nn.ModuleList(
            [ClusterNet(cfg) for cfg in config['cluster_layers']])

    def forward(self, up_x, edge_up2down, out_size=None):
        '''
            up_x: [N, F],  edge_up2down: [M, 2]
            e.g. 3->1
        '''
        # [N, F] -> [N', F]
        up_x = up_x[edge_up2down[0]]
        for cluster_layer in self.cluster_layers:
            # [N', F], [N'] -> [N', 2*out_size], [C, out_size]
            up_x, dw_x = cluster_layer(up_x, edge_up2down[1], out_size)
        # [C, out_size]
        return dw_x
"""
