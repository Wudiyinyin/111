# Copyright (c) 2021 Li Auto Company. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageUpsampleLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if 'cordconv' in config and config['cordconv']:
            self.input_aug = 3
        else:
            self.input_aug = 0

        self.deconv = nn.ConvTranspose2d(config['in_channels'] + self.input_aug, config['out_channels'],
                                         config['kernel_size'], config['stride'], config['padding'],
                                         config['out_padding'])

        self.conv = nn.Conv2d(config['out_channels'], config['out_channels'], 3, 1, padding=1)

        self.norm = nn.BatchNorm2d(config['out_channels'])

        self.dropout = nn.Dropout2d(p=config['dropout'], inplace=True)

    def forward(self, x):
        # add cord encoding
        if self.input_aug != 0:
            b, c, h, w = x.shape
            x_cord = 2 * (torch.arange(w, device=x.device) - (w - 1) / 2) / w
            y_cord = 2 * (torch.arange(h, device=x.device) - (h - 1) / 2) / h
            x_embeding = x_cord.expand(h, w)
            y_embeding = y_cord.unsqueeze(1).expand(h, w)
            r_embeding = torch.sqrt(x_embeding * x_embeding + y_embeding * y_embeding)
            cord_embeding = torch.stack([x_embeding, y_embeding, r_embeding], dim=0).expand(b, 3, h, w)
            x = torch.cat([x, cord_embeding], dim=1)
        # [b, in_c, h, w] -> [b, out_c, out_size, out_size]
        res = self.deconv(x)
        res = F.relu(res)
        # [b, out_c, out_size, out_size]
        res = self.conv(res)
        # [b, out_c, out_size, out_size]
        res = self.norm(res)
        # [b, in_c, h, w] -> [b, 1, in_c, h, w] (make it 5D)
        # -> [b, 1, out_c, out_size, out_size] -> [b, out_c, out_size, out_size]
        # reduce(*1/2) the channel, increase(*2) the h, w(out_size)
        bypass = F.interpolate(x.unsqueeze(1),
                               size=(self.config['out_channels'], self.config['out_size'], self.config['out_size']),
                               mode='nearest').view(-1, self.config['out_channels'], self.config['out_size'],
                                                    self.config['out_size'])
        # [b, out_c, out_size, out_size] + [b, out_c, out_size, out_size] -> [b, out_c, out_size, out_size]
        y = F.relu(res + bypass)
        # [b, out_c, out_size, out_size]
        return self.dropout(y)
