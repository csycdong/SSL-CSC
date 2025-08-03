import torch
import torch.nn as nn


class EncoderV2(nn.Module):
    def __init__(self, n=41, out_chs=64, n_feats=32, n_blocks=5, ksize=(13, 7), bsize=(1,5)):
        super(EncoderV2, self).__init__()
        self.encoder_r = BSNEncoder(n, out_chs, n_feats, n_blocks, ksize, bsize)
        self.encoder_g = BSNEncoder(n, out_chs, n_feats, n_blocks, ksize, bsize)
        self.encoder_b = BSNEncoder(n, out_chs, n_feats, n_blocks, ksize, bsize)

    def forward(self, raw_spk, mask):
        r_blocks = self.encoder_r(raw_spk, mask[:,0:1,:,:])
        g_blocks = self.encoder_g(raw_spk, mask[:,1:2,:,:]+mask[:,2:3,:,:])
        b_blocks = self.encoder_b(raw_spk, mask[:,3:4,:,:])
        return r_blocks, g_blocks, b_blocks

class BSNEncoder(nn.Module): 
    def __init__(self, n=41, out_chs=64, n_feats=32, n_blocks=5, ksize=(13, 7), bsize=(1,5)):
        super(BSNEncoder, self).__init__()
        self.n_blocks = n_blocks
        stride = (n - ksize[0]) // (self.n_blocks - 1)
        padding = (ksize[1] - 1) // 2
        self.tem_head_1 = PartialConv3d(1, n_feats//2, (3, ksize[1], ksize[1]), (1, 1, 1), (1, padding, padding))
        self.tem_encoder_1 = nn.Conv3d(n_feats//2, out_chs, (ksize[0], 3, 3), (stride, 1, 1), (0, 1, 1))
        self.tem_head_2 = PartialConv3d(1, n_feats//2, (3, ksize[1], ksize[1]), (1, 1, 1), (1, padding, padding))
        self.tem_encoder_2 = TMConv3d(n_feats//2, out_chs, (ksize[0], 3, 3), (stride, 1, 1), (0, 1, 1), size=bsize[1])
        self.fusion = nn.Conv2d(out_chs*2, out_chs, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.n_blocks = n_blocks
        self.out_chs = out_chs
        self.n_feats = n_feats

    def forward(self, raw_spk, mask):
        tem_feats_1 = self.act(self.tem_head_1(raw_spk.unsqueeze(1), mask.unsqueeze(1).repeat(1, 1, raw_spk.shape[1], 1, 1)))
        tem_feats_1 = self.act(self.tem_encoder_1(tem_feats_1))
        tem_feats_2 = self.act(self.tem_head_2(raw_spk.unsqueeze(1), mask.unsqueeze(1).repeat(1, 1, raw_spk.shape[1], 1, 1)))
        tem_feats_2 = self.act(self.tem_encoder_2(tem_feats_2))
        blocks = [self.act(self.fusion(torch.cat([tem_feats_1[:,:,i,:,:],
                                                  tem_feats_2[:,:,i,:,:]], dim=1))) for i in range(self.n_blocks)]
        return blocks
    

class TMConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, size=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kD, kH, kW = self.weight.size()
        self.mask.fill_(1)
        half_win = (size - 1)//2
        self.mask[:,:,kD//2-half_win:kD//2+half_win+1,kH//2,kW//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    

class PartialConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as( # 感觉是要加维度，就增加了一维
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        return output