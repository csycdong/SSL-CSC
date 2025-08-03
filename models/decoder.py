import torch
import torch.nn as nn

import models.common as common


class Decoder(nn.Module):
    def __init__(self, in_chs=32, out_chs=3, n_feats=[32, 64], 
             n_resblocks=18, kernel_size=3, act='leakyrelu', conv=common.default_conv):
        super(Decoder, self).__init__()
        self.head_r = nn.Sequential(
                nn.Conv2d(in_chs, n_feats[0], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1))
        self.head_g = nn.Sequential(
                nn.Conv2d(in_chs, n_feats[0], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1))
        self.head_b = nn.Sequential(
                nn.Conv2d(in_chs, n_feats[0], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1))
        self.fuse = conv(n_feats[0]*3, n_feats[1], kernel_size)
        self.body = nn.Sequential(*[
            common.ResBlock(n_feats[1], kernel_size, act_type=act, bias=True, res_scale=1) 
                for _ in range(n_resblocks)])
        self.tail = nn.Sequential(
                nn.Conv2d(n_feats[1], out_chs, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.1))

    def forward(self, r_feats, g_feats, b_feats):
        r_feats = self.head_r(r_feats)
        g_feats = self.head_g(g_feats)
        b_feats = self.head_b(b_feats)
        x = self.fuse(torch.cat([r_feats, g_feats, b_feats], axis=1))
        res = self.body(x) + x
        x = self.tail(res)
        return x
