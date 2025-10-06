# ── imports ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# If you already have these somewhere (e.g., modules/layers_3d.py), do:
# from modules.layers_3d import ResnetBlock3D, ConvBlock3D, AttnBlock3D, norm_layer_3d

# ── fallbacks (remove if you have your own implementations) ─────────────────────
def norm_layer_3d(ch: int) -> nn.Module:
    # group norm is stable for small batch sizes
    num_groups = min(32, ch)
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch)

class ConvBlock3D(nn.Module):
    """Conv3d -> Norm -> SiLU"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = norm_layer_3d(out_ch)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x

class ResnetBlock3D(nn.Module):
    """(Conv-Norm-SiLU) x2 with skip (1x1) if channels change."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = norm_layer_3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = norm_layer_3d(out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch
                      else nn.Conv3d(in_ch, out_ch, kernel_size=1))
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.silu(y)
        return y + self.skip(x)

class AttnBlock3D(nn.Module):
    """
    Lightweight 3D self-attention: (B,C,D,H,W) -> (B,C,D,H,W).
    Uses channels as embedding dim; flattens DHW.
    """
    def __init__(self, ch):
        super().__init__()
        self.q = nn.Conv3d(ch, ch, 1)
        self.k = nn.Conv3d(ch, ch, 1)
        self.v = nn.Conv3d(ch, ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        q = self.q(x).view(B, C, -1).transpose(1, 2)        # [B, DHW, C]
        k = self.k(x).view(B, C, -1)                        # [B, C, DHW]
        v = self.v(x).view(B, C, -1).transpose(1, 2)        # [B, DHW, C]
        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)    # [B, DHW, DHW]
        y = attn @ v                                        # [B, DHW, C]
        y = y.transpose(1, 2).view(B, C, D, H, W)
        return self.proj(y) + x

# keep your class as-is; the imports above make it work
class Encoder3D(nn.Module):
    def __init__(self, *, ch, ch_mult=(1,2,4,8), num_res_blocks,
                 residual=True, dropout=0.0, in_channels=1,
                 use_attention=False, mid_blocks=True,
                 out_channels=1):                

        super().__init__()
        self.ch = ch
        self.out_channels = out_channels  
        self.residual = residual
        self.use_attention = use_attention
        self.mid_blocks = mid_blocks
        block_nn = ResnetBlock3D if residual else ConvBlock3D

        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        c_in = ch
        for i_level, m in enumerate(ch_mult):
            c_out = ch * m
            blocks = nn.ModuleList()
            attns  = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(block_nn(c_in, c_out, dropout=dropout))
                attns.append(AttnBlock3D(c_out) if use_attention else nn.Identity())
                c_in = c_out
            down = nn.Module()
            down.block = blocks
            down.attn  = attns
            if i_level != len(ch_mult) - 1:
                down.downsample = nn.MaxPool3d(2)
            self.down.append(down)

        self.mid = nn.Module()
        if mid_blocks:
            self.mid.block_1 = block_nn(c_in, c_in, dropout=dropout)
            self.mid.attn_1  = AttnBlock3D(c_in) if use_attention else nn.Identity()
            self.mid.block_2 = block_nn(c_in, c_in, dropout=dropout)
        else:
            self.mid.block_1 = self.mid.attn_1 = self.mid.block_2 = nn.Identity()

        self.norm_out = norm_layer_3d(c_in) if residual else nn.Identity()
        self.conv_out = nn.Conv3d(c_in, out_channels, kernel_size=1)   

    def forward(self, x):  # [B*T, Cg, D, H, W]
        h = self.conv_in(x)
        for lvl in self.down:
            for b, a in zip(lvl.block, lvl.attn):
                h = b(h); h = a(h)
            if hasattr(lvl, "downsample"):
                h = lvl.downsample(h)
        h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = F.silu(h) if isinstance(self.norm_out, nn.Module) else h
        return self.conv_out(h)  # [B*T, 1, D', H', W']
