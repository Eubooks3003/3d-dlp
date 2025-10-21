import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.voxel_modules.DLPDecoder.decoder_utils import ResBlock3D, UpBlock3D


class BgDecoder3D(nn.Module):
    """
    Maps z_bg_features [B,F] to a full volume [B, C_out, D, H, W].
    """
    def __init__(self,
                 grid_dhw=(48,48,48),
                 cdim_out=1,                  # output channels for the background volume
                 learned_bg_feature_dim=16,
                 base_ch=32, ch_mult=(1,2,3), num_res_blocks=2,
                 res_from_fc=4,
                 use_resblock=True,
                 final_cnn_ch=32,
                 init_zero_bias=True, init_conv_layers=True, init_conv_bg_std=0.005):
        super().__init__()
        self.D, self.H, self.W = map(int, grid_dhw)
        self.res0 = int(res_from_fc)

        c0 = base_ch * ch_mult[0]
        self.fc = nn.Linear(learned_bg_feature_dim, c0 * self.res0 * self.res0 * self.res0)

        ups = []
        ch = c0
        for i, m in enumerate(ch_mult):
            ch_next = base_ch * m
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
        self.head = nn.Conv3d(ch, cdim_out, 3, padding=1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if init_conv_layers: nn.init.normal_(m.weight, 0.0, init_conv_bg_std)
                if init_zero_bias and m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if init_zero_bias and m.bias is not None: nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """
        Light-touch initializer:
        - Prefer any submodule-specific init if it exists.
        - Otherwise: Conv/ConvT weights ~ N(0, init_conv_bg_std), biases -> 0,
          Norm scales -> 1, Norm biases -> 0, Linear biases -> 0.
        """
        # give any purpose-built inits a chance first
        for name in ["proj", "up", "upsampler", "to_rgb", "to_rgbd", "to_out"]:
            m = getattr(self, name, None)
            if m is not None and hasattr(m, "init_weights"):
                m.init_weights()

        # generic init pass
        std = getattr(self, "init_conv_bg_std", 0.005)
        zero_bias = getattr(self, "init_zero_bias", True)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight, 0.0, std)
                if zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def info(self) -> str:
        """
        Human-readable summary; resilient to missing attributes.
        """
        D, H, W = getattr(self, "grid_dhw", ("?", "?", "?"))
        res_from_fc = getattr(self, "res_from_fc", getattr(self, "bg_res_from_fc", "?"))
        return (
            "BgDecoder3D: "
            f"target_grid=({D},{H},{W}), "
            f"out_ch={getattr(self, 'cdim', getattr(self, 'cdim_out', '?'))}, "
            f"res_from_fc={res_from_fc}, "
            f"base_ch={getattr(self, 'base_ch', '?')}, "
            f"ch_mult={getattr(self, 'ch_mult', '?')}, "
            f"final_cnn_ch={getattr(self, 'final_cnn_ch', '?')}, "
            f"num_res_blocks={getattr(self, 'num_res_blocks', '?')}"
        )


    def forward(self, z_bg):   # [B,F]
        B, Fdim = z_bg.shape
        x = self.fc(z_bg).view(B, -1, self.res0, self.res0, self.res0)
        x = self.ups(x)
        # upsample to target grid
        x = F.interpolate(x, size=(self.D, self.H, self.W), mode='trilinear', align_corners=False)
        x = self.head(x)  # [B, cdim_out, D, H, W]
        return x