import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.voxel_modules.DLPDecoder.decoder_utils import ResBlock3D, UpBlock3D

class ObjectDecoderCNN3D(nn.Module):
    """
    Takes per-particle bottleneck (B*N, F) and outputs a 3D patch (alpha + payload channels).
    """
    def __init__(self,
                 patch_size=(16,16,16),
                 num_chans_out=1,           # payload channels (excluding alpha)
                 bottleneck_size=16,
                 base_ch=32, ch_mult=(1,2,3), num_res_blocks=2,
                 res_from_fc=4,
                 use_resblock=True,
                 final_cnn_ch=32,
                 init_zero_bias=True, init_conv_layers=True, init_conv_fg_std=0.02):
        super().__init__()
        self.ps = tuple(patch_size)  # (d,h,w)
        self.res0 = int(res_from_fc)

        c0 = base_ch * ch_mult[0]
        self.fc = nn.Linear(bottleneck_size, c0 * self.res0 * self.res0 * self.res0)

        ups = []
        ch = c0
        for i, m in enumerate(ch_mult):
            ch_next = base_ch * m
            # first stage just formats to (res0,res0,res0)
            if i == 0:
                ups.append(nn.Sequential(
                    nn.Conv3d(ch, ch_next, 3, padding=1),
                    ResBlock3D(ch_next, use_resblock),
                    *[ResBlock3D(ch_next, use_resblock) for _ in range(num_res_blocks-1)],
                ))
            else:
                ups.append(UpBlock3D(ch, ch_next, use_resblock))
                for _ in range(num_res_blocks-1):
                    ups.append(ResBlock3D(ch_next, use_resblock))
            ch = ch_next

        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv3d(ch, 1 + num_chans_out, 3, padding=1)  # alpha + payload

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if init_conv_layers: nn.init.normal_(m.weight, 0.0, init_conv_fg_std)
                if init_zero_bias and m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if init_zero_bias and m.bias is not None: nn.init.constant_(m.bias, 0)

    @property
    def out_channels(self):
        return self.head.out_channels

    def forward(self, z_feat):  # z_feat: [B*N, F]
        Bn, Fdim = z_feat.shape
        x = self.fc(z_feat).view(Bn, -1, self.res0, self.res0, self.res0)
        x = self.ups(x)

        # if spatial size is smaller than patch size, upsample (nearest)
        _, _, d, h, w = x.shape
        D, H, W = self.ps
        if (d, h, w) != (D, H, W):
            x = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=False)
        return x  # [B*N, 1+num_chans_out, D, H, W]