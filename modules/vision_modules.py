"""
based on: https://github.com/CompVis/taming-transformers/blob/master/taming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision import models
from collections import namedtuple
import os
import hashlib
import requests
from PIL import Image
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

"""
Functions
"""


def calc_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # estimate model size on disk: https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return {'n_params': num_trainable_params, 'size_mb': size_all_mb}


def nonlinearity(x):
    # lrelu
    # return F.leaky_relu(x, negative_slope=0.01)
    # relu
    # return F.relu(x)
    # gelu
    return F.gelu(x)
    # swish
    # return x * torch.sigmoid(x)


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def norm_layer(in_channels, num_groups=4, eps=1e-5):
    # base_groups = num_groups
    # if in_channels <= 32:
    #     num_groups = base_groups
    # elif in_channels == 64:
    #     num_groups = base_groups * 2  # 8
    # elif num_groups == 128:
    #     num_groups = base_groups * 4  # 16
    # else:
    #     num_groups = base_groups * 8  # 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)


def rgb_to_minusoneone(x):
    x = 2. * x - 1.
    return x


def minusoneone_to_rgb(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helpers ----
def _norm3d(C, num_groups=4, eps=1e-5):

    return nn.GroupNorm(num_groups=num_groups, num_channels=C, eps=eps, affine=True)


# ============================ 3D VERSIONS ============================

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, padding_mode='zeros', mode='nearest'):
        super().__init__()
        self.with_conv = with_conv
        self.mode = mode  # 'nearest' or 'trilinear'
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                  padding_mode=padding_mode)

    def forward(self, x):
        if self.mode == 'trilinear':
            x = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, use_conv_block=False, padding_mode='constant'):
        super().__init__()
        self.with_conv = with_conv
        self.use_conv_block = use_conv_block
        # F.pad for 5D supports 'constant' reliably; map others to 'constant'
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        if self.with_conv:
            if self.use_conv_block:
                self.conv = ConvBlock(in_channels=in_channels, out_channels=in_channels,
                                      dropout=0.0, padding=0, stride=2, kernel_size=3)
            else:
                self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.avg = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.with_conv:
            # pad (W0,W1,H0,H1,D0,D1)
            x = F.pad(x, (0,1, 0,1, 0,1), mode=self.padding_mode, value=0)
            x = self.conv(x)
        else:
            x = self.avg(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=0, padding_mode='zeros', padding=1, stride=1, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, padding_mode=padding_mode)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm = _norm3d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temb=None):
        h = self.conv(x)
        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        h = self.norm(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=0, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = _norm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode=padding_mode)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = _norm3d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode=padding_mode)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                               padding_mode=padding_mode)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _norm3d(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, d, h, w = q.shape
        n = d * h * w
        q = q.reshape(b, c, n).permute(0, 2, 1)  # [B, N, C]
        k = k.reshape(b, c, n)                   # [B, C, N]
        attn = torch.bmm(q, k) * (c ** -0.5)     # [B, N, N]
        attn = F.softmax(attn, dim=2)

        v = v.reshape(b, c, n)                   # [B, C, N]
        attn_t = attn.permute(0, 2, 1)           # [B, N, N]
        h_ = torch.bmm(v, attn_t)                # [B, C, N]
        h_ = h_.reshape(b, c, d, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks, residual=True,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, padding_mode='zeros', attention=False,
                 mid_blocks=True, in_conv_kernel_size=3, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution  # assume cubic D=H=W=resolution
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.use_attention = attention
        self.residual = residual
        self.mid_blocks = mid_blocks

        block_nn = ResnetBlock if self.residual else ConvBlock

        first_conv_pad = in_conv_kernel_size // 2
        self.conv_in = nn.Conv3d(in_channels, self.ch, kernel_size=in_conv_kernel_size, stride=1,
                                 padding=first_conv_pad,
                                 padding_mode=self.padding_mode)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(block_nn(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout, padding_mode=self.padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in) if attention else nn.Identity())
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, padding_mode=self.padding_mode)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        if self.mid_blocks:
            self.mid.block_1 = block_nn(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout, padding_mode=self.padding_mode)
            self.mid.attn_1 = AttnBlock(block_in) if attention else nn.Identity()
            self.mid.block_2 = block_nn(in_channels=block_in, out_channels=block_in,
                                        temb_channels=self.temb_ch, dropout=dropout, padding_mode=self.padding_mode)
        else:
            self.mid.block_1 = nn.Identity()
            self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = nn.Identity()

        # end
        self.norm_out = _norm3d(block_in) if self.residual else nn.Identity()
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1,
                                  padding_mode=self.padding_mode)
        self.conv_output_size = self.calc_conv_output_size()

    def calc_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.resolution, self.resolution, self.resolution)
            dummy_out = self.forward(dummy_input)
        # returns [C_out, D_out, H_out, W_out]
        return dummy_out.shape[1:]

    def forward(self, x):
        # x: [B, C, D, H, W]
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        if self.mid_blocks:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        if self.residual:
            h = self.norm_out(h)
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    """
    3D-aware version of your Decoder.

    Args:
      ch, out_ch, ch_mult, num_res_blocks, attn_resolutions, dropout, resamp_with_conv, residual,
      resolution: int or (D,H,W)  -> full output spatial size
      z_channels: channels of latent tensor at lowest resolution
      give_pre_end, padding_mode, attention, mid_blocks, upsample_method ('nearest'|'trilinear')
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, residual=True,
                 resolution, z_channels, give_pre_end=False, padding_mode='zeros', attention=False,
                 mid_blocks=True, upsample_method='nearest', **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.residual = residual
        self.give_pre_end = give_pre_end
        self.padding_mode = padding_mode
        self.use_attention = attention
        self.mid_blocks = mid_blocks
        self.upsample_method = upsample_method

        # resolution handling (3D)
        D = H = W = resolution

        # lowest resolution (after downsampling n-1 times)
        d0 = D // 2 ** (self.num_resolutions - 1)
        h0 = H // 2 ** (self.num_resolutions - 1)
        w0 = W // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, d0, h0, w0)

        # choose block type
        block_nn = ResnetBlock if self.residual else ConvBlock

        # compute in_ch_mult, block_in and curr_res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res_d, curr_res_h, curr_res_w = d0, h0, w0

        # z to block_in
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

        # middle
        self.mid = nn.Module()
        if self.mid_blocks:
            self.mid.block_1 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
            # 3D attention block can be added here; identity for now
            self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
        else:
            self.mid.block_1 = nn.Identity()
            self.mid.attn_1  = nn.Identity()
            self.mid.block_2 = nn.Identity()

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn  = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(block_nn(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout, padding_mode=self.padding_mode))
                block_in = block_out
                # attention resolutions (use D resolution to decide)
                if curr_res_d in attn_resolutions:
                    attn.append(nn.Identity())  # placeholder for 3D attention if you add it
            up = nn.Module()
            up.block = block
            up.attn  = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=resamp_with_conv,
                                         padding_mode=self.padding_mode, mode=self.upsample_method)
                curr_res_d *= 2
                curr_res_h *= 2
                curr_res_w *= 2
            self.up.insert(0, up)

        # end
        self.norm_out = norm_layer(block_in) if self.residual else nn.Identity()
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

    def forward(self, z):
        # z: [B, C, d0, h0, w0]  expected to match self.z_shape[1:]
        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)

        if self.mid_blocks:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        if self.residual:
            h = self.norm_out(h)
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "eval/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LossLPIPS(nn.Module):
    def __init__(self, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        loss = torch.mean(nll_loss)

        log = {"total_loss".format(split): loss.clone().detach().mean(),
               "nll_loss".format(split): nll_loss.detach().mean(),
               "rec_loss".format(split): rec_loss.detach().mean(),
               "p_loss".format(split): p_loss.detach().mean(),
               }
        return loss, log

