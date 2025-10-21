import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, ch, use_resblock=True):
        super().__init__()
        self.use = use_resblock
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        if not self.use:  # identity if disabled
            return x
        return x + self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, use_resblock=True):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock3D(ch_out, use_resblock=use_resblock)
    def forward(self, x):
        x = self.deconv(x)
        x = self.res(x)
        return x
