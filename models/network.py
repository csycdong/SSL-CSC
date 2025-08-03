import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class Net(nn.Module): # base model
    def __init__(self, n=41, n_feats=[64, 32, 32, 64], n_blocks=5, 
                 ksize=(13, 7), bsize=(0, 5), n_resblocks=4):
        super(Net, self).__init__()
        self.encoder = Encoder(n=n, out_chs=n_feats[0], n_feats=n_feats[1],
                                n_blocks=n_blocks, ksize=ksize, bsize=bsize)
        self.decoder = Decoder(in_chs=n_feats[0]*n_blocks, n_feats=n_feats[2:], n_resblocks=n_resblocks)

    def forward(self, raw_spk, mask):
        r_blocks, g_blocks, b_blocks = self.encoder(raw_spk, mask)
        r_feats = torch.cat(r_blocks, axis=1)
        g_feats = torch.cat(g_blocks, axis=1)
        b_feats = torch.cat(b_blocks, axis=1)
        rgb = self.decoder(r_feats, g_feats, b_feats)
        return rgb
