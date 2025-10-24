import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.point_cloud_modules.vision_modules_3d import Encoder3D


def spatial_transform_3d(x, z_pos, z_scale, out_dims, padding_mode='border'):
    """
    3D analog of your 2D spatial_transform using affine_grid + grid_sample.

    Args:
      x:        [B*K, C, D, H, W]   (source volume per kp)
      z_pos:    [B*K, 3]            (x,y,z) in [-1,1] w.r.t. full volume
      z_scale:  [B*K, 3]            (sx,sy,sz) in (0,1], fraction of full field-of-view per axis
      out_dims: (B*K, C, d, h, w)   target crop size

    Returns:
      crops:    [B*K, C, d, h, w]
    """
    BK, C, D, H, W = x.shape
    _, _, d, h, w = out_dims

    # theta maps normalized output coords -> normalized input coords
    # So to sample a *local* region, set scale = z_scale and translate = z_pos
    sx, sy, sz = z_scale[:, 0], z_scale[:, 1], z_scale[:, 2]
    tx, ty, tz = z_pos[:, 0], z_pos[:, 1], z_pos[:, 2]

    # build per-sample 3x4 affine in (x,y,z) order
    theta = torch.zeros(BK, 3, 4, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = sx; theta[:, 0, 3] = tx
    theta[:, 1, 1] = sy; theta[:, 1, 3] = ty
    theta[:, 2, 2] = sz; theta[:, 2, 3] = tz

    grid = F.affine_grid(theta, size=out_dims, align_corners=True)   # [BK, d, h, w, 3]
    crops = F.grid_sample(x, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    return crops


class ParticleAttributeEncoder3D(nn.Module):
    """
    3D version mirroring your 2D ParticleAttributeEncoder:
      - CNN3D backbone on cropped 3D glimpses
      - heads for offsets, scale, obj_on Beta(a,b), and depth
      - optional temporal embedding

    Keypoints are in xyz order in [-1,1]; scales are per-axis fractions in (0,1].
    """

    def __init__(self,
                 anchor_size: float,           # fraction of (D,H,W); used for default crop & scale
                 grid_dhw: tuple,              # (D,H,W) full volume size used for default scale
                 n_particles: int,
                 ch_in: int,                   # input channels of dense voxel volume
                 crop_size: tuple = None,      # optional (d,h,w); else isotropic from anchor_size
                 base_ch=32, ch_mult=(1,2,3), num_res_blocks=2,
                 use_resblock=True, cnn_mid_blocks=False, use_attention=False,
                 hidden_dim=512, activation='gelu',
                 kp_activation='tanh', max_offset=1.0,
                 obj_on=True, depth=False, scale=True,
                 timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 # init
                 init_zero_bias=True, init_conv_layers=True, init_conv_fg_std=0.02,
                 obj_on_min=1e-4, obj_on_max=100.0):
        super().__init__()
        self.anchor_size = float(anchor_size)
        D, H, W = grid_dhw
        self.grid_dhw = (D, H, W)
        # default isotropic cube crop
        if crop_size is None:
            side = int(round(self.anchor_size * (min(D, H, W) - 1)))
            side = max(4, side)
            crop_size = (side, side, side)
        self.crop_d, self.crop_h, self.crop_w = crop_size

        self.n_particles = n_particles
        self.ch_in = ch_in
        self.use_resblock = use_resblock
        self.cnn_mid_blocks = cnn_mid_blocks
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.kp_activation = kp_activation
        self.max_offset = max_offset
        self.with_obj_on = obj_on
        self.with_depth = depth
        self.with_scale = scale
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max

        # 3D CNN backbone; output channel count == final_cnn_ch
        final_cnn_ch = 32
        self.cnn3d = Encoder3D(
            in_channels=ch_in,
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            residual=use_resblock, dropout=0.0,
            use_attention=use_attention,
            mid_blocks=not cnn_mid_blocks,
            out_channels=final_cnn_ch
        )

        # figure out flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, ch_in, self.crop_d, self.crop_h, self.crop_w)
            feat = self.cnn3d(dummy)
            if isinstance(feat, tuple):
                feat = feat[1]
            _, Cc, d2, h2, w2 = feat.shape
            self._feat_shape = (Cc, d2, h2, w2)
            fc_in_dim = Cc * d2 * h2 * w2

        Act = nn.GELU if activation == 'gelu' else nn.ReLU

        # backbone projection (identity like 2D)
        self.backbone = nn.Identity()

        # offsets head: (mu_x, mu_y, mu_z, logvar_x, logvar_y, logvar_z)
        self.offset_head = nn.Sequential(
            nn.Linear(fc_in_dim, hidden_dim), Act(),
            nn.Linear(hidden_dim, 6)
        )

        # scale head
        if self.with_scale:
            self.scale_head = nn.Sequential(
                nn.Linear(fc_in_dim, hidden_dim), Act(),
                nn.Linear(hidden_dim, 6)   # mu_sx, mu_sy, mu_sz, logvar_sx, logvar_sy, logvar_sz
            )
        else:
            self.scale_head = None

        # obj_on Beta head
        if self.with_obj_on:
            self.obj_on_head = nn.Sequential(
                nn.Linear(fc_in_dim, hidden_dim), Act(),
                nn.Linear(hidden_dim, 2, bias=True)   # [l_a, l_b]
            )
        else:
            self.obj_on_head = None

        # depth head (kept for parity)
        if self.with_depth:
            self.depth_head = nn.Sequential(
                nn.Linear(fc_in_dim, hidden_dim), Act(),
                nn.Linear(hidden_dim, 2)  # mu_depth, logvar_depth
            )
        else:
            self.depth_head = None

        # optional temporal embedding (added after flatten)
        if self.add_particle_temp_embed and self.timestep_horizon > 1:
            self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, fc_in_dim))
        else:
            self.temp_embed = None

        # init
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_fg_std = init_conv_fg_std
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # match the 2D biasing strategy:
        # offsets last layer: encourage zero-mean, small variance
        if isinstance(self.offset_head[-1], nn.Linear):
            # mu (first 3) bias -> 0, logvar (last 3) bias -> log(0.01^2)
            with torch.no_grad():
                w = self.offset_head[-1].weight
                b = self.offset_head[-1].bias
                if w is not None:
                    nn.init.constant_(w, 0.0)
                b[:3].fill_(0.0)
                b[3:].fill_(math.log(0.01 ** 2))

        if self.scale_head is not None and isinstance(self.scale_head[-1], nn.Linear):
            with torch.no_grad():
                w = self.scale_head[-1].weight
                b = self.scale_head[-1].bias
                nn.init.constant_(w, 0.0)
                # mu_scale bias ~ 0, logvar_scale bias ~ log(0.1^2) like 2D
                b[:3].fill_(0.0)
                b[3:].fill_(math.log(0.1 ** 2))

        if self.obj_on_head is not None and isinstance(self.obj_on_head[-1], nn.Linear):
            with torch.no_grad():
                nn.init.constant_(self.obj_on_head[-1].weight, 0.0)
                if self.obj_on_head[-1].bias is not None:
                    nn.init.constant_(self.obj_on_head[-1].bias, 0.0)


    def forward(self, x_dense, kp_xyz, z_scale=None, timesteps=None, deterministic=False):
        """
        x_dense: [B, C, D, H, W]
        kp_xyz:  [B, K, 3]   (x,y,z) in [-1,1]
        z_scale: [B, K, 3]   (optional, unnormalized; we'll sigmoid like 2D)
        """
        B, C, D, H, W = x_dense.shape
        _, K, _ = kp_xyz.shape

        # repeat volume per keypoint
        x_rep = x_dense[:, None].expand(B, K, C, D, H, W).reshape(B*K, C, D, H, W)

        # default z_scale = (crop / grid)
        if z_scale is None:
            zs = torch.tensor([
                (self.crop_w - 1) / (W - 1),
                (self.crop_h - 1) / (H - 1),
                (self.crop_d - 1) / (D - 1),
            ], device=x_dense.device, dtype=x_dense.dtype)  # (x,y,z)
            z_scale = zs.view(1, 1, 3).expand(B, K, 3)
        else:
            z_scale = torch.sigmoid(z_scale)
        # always keep scales in a sane range
        z_scale = z_scale.clamp(1e-3, 1.0)


        # flatten positions/scales to match [B*K, ...]
        z_pos   = kp_xyz.reshape(B*K, 3)        # (x,y,z)
        z_scale = z_scale.reshape(B*K, 3)

        # crop
        out_dims = (B*K, C, self.crop_d, self.crop_h, self.crop_w)
        crops = spatial_transform_3d(x_rep, z_pos, z_scale, out_dims, padding_mode='border')  # [B*K,C,d,h,w]

        # encode with 3D CNN
        feat = self.cnn3d(crops)
        if isinstance(feat, tuple):
            feat = feat[1]
        feat_flat = feat.flatten(1)             # [B*K, Fc]

        # optional temporal embedding (kept no-op here)
        if (self.timestep_horizon > 1) and (self.add_particle_temp_embed) and (self.temp_embed is not None) and (timesteps is not None):
            pass

        # presence Beta
        if self.with_obj_on and self.obj_on_head is not None:
            l_a_b = self.obj_on_head(feat_flat)                  # [B*K,2]
            l_a, l_b = l_a_b.split(1, dim=-1)
            gate_a = torch.sigmoid(l_a)
            gate_b = torch.sigmoid(l_b)  # or keep 1 - sigmoid(l_a) if you prefer coupling

            a = (1.0 - gate_a) * self.obj_on_min + gate_a * self.obj_on_max
            b = (1.0 - gate_b) * self.obj_on_min + gate_b * self.obj_on_max

            eps = 1e-8
            a = a.clamp_min(eps)
            b = b.clamp_min(eps)

            beta = torch.distributions.Beta(a, b)

            mu_obj_on = beta.mean
            z_obj_on  = mu_obj_on if deterministic else beta.rsample()
        else:
            a = b = mu_obj_on = z_obj_on = None

        # offsets
        off = self.offset_head(feat_flat)                          # [B*K,6]
        mu, logvar = off[:, :3], off[:, 3:]

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        # scale
        if self.with_scale and self.scale_head is not None:
            sc = self.scale_head(feat_flat)                        # [B*K,6]
            mu_scale, logvar_scale = sc[:, :3], sc[:, 3:]
        else:
            mu_scale, logvar_scale = None, None

        # depth (optional, parity)
        if self.with_depth and self.depth_head is not None:
            dxy = self.depth_head(feat_flat)                       # [B*K,2]
            mu_depth, logvar_depth = dxy.split(1, dim=-1)
        else:
            mu_depth = logvar_depth = None

        # reshape back to [B,K,*]
        def V(t, d):  # view helper
            return t.view(B, K, d) if (t is not None) else None

        mu      = V(mu, 3)
        logvar  = V(logvar, 3)
        mu_scale     = V(mu_scale, 3) if mu_scale is not None else None
        logvar_scale = V(logvar_scale, 3) if logvar_scale is not None else None
        mu_depth     = V(mu_depth, 1) if mu_depth is not None else None
        logvar_depth = V(logvar_depth, 1) if logvar_depth is not None else None
        obj_on_a     = V(a, 1) if a is not None else None
        obj_on_b     = V(b, 1) if b is not None else None
        mu_obj_on    = V(mu_obj_on, 1) if mu_obj_on is not None else None
        z_obj_on     = V(z_obj_on, 1) if z_obj_on is not None else None

        crops = crops.view(B, K, self.ch_in, self.crop_d, self.crop_h, self.crop_w)

        return {
            'mu': mu, 'logvar': logvar,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
            'mu_obj_on': mu_obj_on, 'z_obj_on': z_obj_on,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
            'cropped_objects': crops
        }

