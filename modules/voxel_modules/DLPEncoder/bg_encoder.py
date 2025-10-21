import math
import torch
import torch.nn as nn

from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from utils.util_func import reparameterize


class BgEncoder3D(nn.Module):
    def __init__(self,
                 in_channels=1,                 # voxel feature channels (e.g., density=1 or multi-ch)
                 grid_dhw=(64, 64, 64),        # (D, H, W)
                 learned_feature_dim=32,        # latent feature dim to produce
                 features_dist='gauss',         # 'gauss' -> output logvar; 'categorical' or interaction_features=True -> no logvar
                 interaction_features=False,    # if True: output features directly (no logvar + no sampling)
                 base_ch=32, ch_mult=(1, 2, 3), num_res_blocks=2,
                 use_resblock=True, use_attention=False, cnn_mid_blocks=False,
                 final_cnn_ch=32,              # channels after Encoder3D
                 pad_value=0.0,                # for masks
                 timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 activation='gelu', mlp_hidden_dim=256,

                 # initialization
                 init_zero_bias=True,
                 init_conv_layers=True,
                 init_conv_bg_std=0.005):
        """
        Dense-voxel Background Module -- encodes a latent for the (masked) background, z_bg.
        Input:  x [B, C, D, H, W]; optional bg mask m [B, 1, D, H, W] (1=keep).
        Output: mu_bg, logvar_bg (optional), z_bg.
        """
        super().__init__()

        self.in_channels = in_channels
        self.grid_dhw = tuple(grid_dhw)
        self.features_dim = int(learned_feature_dim)
        self.features_dist = features_dist
        self.interaction_features = interaction_features
        self.final_cnn_ch = final_cnn_ch
        self.pad_value = float(pad_value)

        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.use_resblock = use_resblock
        self.use_attention = use_attention
        self.cnn_mid_blocks = cnn_mid_blocks

        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.add_particle_temp_embed = add_particle_temp_embed
        self.init_std = init_std
        self.activation = activation
        self.mlp_hidden_dim = mlp_hidden_dim

        # init flags
        self.init_zero_bias = init_zero_bias
        self.init_conv_layers = init_conv_layers
        self.init_conv_bg_std = init_conv_bg_std

        # 3D backbone
        self.bg_cnn_enc = Encoder3D(
            in_channels=self.in_channels,
            ch=self.base_ch,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
            residual=self.use_resblock,
            dropout=0.0,
            use_attention=self.use_attention,
            mid_blocks=not self.cnn_mid_blocks,
            out_channels=self.final_cnn_ch
        )

        # figure out the CNN output grid size (D',H',W')
        self.cnn_out_shape = self._get_cnn_shape()  # [C', D', H', W']
        Dp, Hp, Wp = self.cnn_out_shape[-3:]
        feature_map_size = Dp * Hp * Wp

        # decide whether to emit logvar
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        self.output_logvar = output_logvar

        # projection mode: FCN if features_dim is a multiple of feature_map_size; else FC
        if self.features_dim % feature_map_size == 0:
            # FCN mode: 1x1x1 conv to channels = features_dim / (D'*H'*W')
            self.ch_learned_feature_dim = max(self.features_dim // feature_map_size, 1)
            out_ch = 2 * self.ch_learned_feature_dim if output_logvar else self.ch_learned_feature_dim
            self.to_latent = nn.Conv3d(in_channels=self.final_cnn_ch,
                                       out_channels=out_ch,
                                       kernel_size=1)
            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
            # temporal embedding (channels, D', H', W')
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, self.final_cnn_ch, Dp, Hp, Wp)
                )
            else:
                self.temp_embed = None
        else:
            # FC mode: flatten and MLP to features_dim (+logvar)
            self.ch_learned_feature_dim = self.final_cnn_ch
            self.to_latent = nn.Identity()
            self.projection_mode = 'fc'
            flat_dim = self.final_cnn_ch * feature_map_size
            self.to_mu = self._get_mlp(flat_dim, self.features_dim)
            self.to_logvar = self._get_mlp(flat_dim, self.features_dim) if output_logvar else nn.Identity()
            # temporal embedding (flattened)
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, flat_dim)
                )
            else:
                self.temp_embed = None

        self.info = (f'BgEncoder3D: requested latent size: {self.features_dim}, '
                     f'cnn out grid: ({Dp},{Hp},{Wp}) => cells={feature_map_size}, '
                     f'projection={self.projection_mode}')

        self.init_weights()

    # ----------------- utils -----------------

    def _get_cnn_shape(self):
        D, H, W = self.grid_dhw
        dummy = torch.randn(1, self.in_channels, D, H, W)
        out = self.bg_cnn_enc(dummy)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]  # [C', D', H', W']

    def _get_mlp(self, in_dim, out_dim):
        activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
        hidden = self.mlp_hidden_dim
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            activation_f(),
            nn.Linear(hidden, out_dim)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # leave default
                pass

    # ----------------- core -----------------

    def encode_bg_features(self, x: torch.Tensor, mask: torch.Tensor = None, timesteps: int = None):
        """
        x:    [B, C, D, H, W] dense voxels
        mask: [B, 1, D, H, W] (optional), 1=keep; applied multiplicatively
        """
        assert x.dim() == 5, f"x must be [B,C,D,H,W], got {x.shape}"
        B = x.size(0)

        if mask is not None:
            x_in = x * mask + self.pad_value * (1.0 - mask)
        else:
            x_in = x

        enc_out = self.bg_cnn_enc(x_in)
        feats = enc_out[1] if isinstance(enc_out, tuple) else enc_out  # [B, C', D', H', W']

        # temporal embed
        if self.projection_mode == 'fcn' and self.temp_embed is not None and timesteps is not None:
            # add on channel grid
            orig = feats.shape  # [B, C', D', H', W']
            feats = feats.view(B, timesteps, *orig[1:])
            feats = feats + self.temp_embed[:, :timesteps]
            feats = feats.view(orig)

        h = self.to_latent(feats)  # FCN: [B, out_ch, D', H', W']  |  FC: Identity

        if self.projection_mode == 'fc':
            flat = feats.reshape(B, -1)  # [B, C'*D'*H'*W']
            if self.temp_embed is not None and timesteps is not None:
                orig = flat.shape  # [B, F]
                flat = flat.view(B, timesteps, -1)
                flat = flat + self.temp_embed[:, :timesteps]
                flat = flat.view(orig)
            mu_bg = self.to_mu(flat)
            logvar_bg = self.to_logvar(flat) if self.output_logvar else None
        else:
            # FCN mode: split channels -> (mu[, logvar]), then flatten to [B, features_dim]
            if self.output_logvar:
                ch_half = h.size(1) // 2
                mu_h, logvar_h = h[:, :ch_half], h[:, ch_half:]
                mu_bg = mu_h.reshape(B, -1)
                logvar_bg = logvar_h.reshape(B, -1)
            else:
                mu_bg = h.reshape(B, -1)
                logvar_bg = None

        return mu_bg, logvar_bg

    def encode_all(self, x: torch.Tensor, mask: torch.Tensor = None, deterministic: bool = False, timesteps: int = None):
        """
        Returns:
          mu_bg:      [B, F]
          logvar_bg:  [B, F] or None
          z_bg:       [B, F]
          z_kp:       [B, 1, 3] (placeholder to mirror 2D API)
        """
        mu_bg, logvar_bg = self.encode_bg_features(x, mask=mask, timesteps=timesteps)
        if self.interaction_features or self.features_dist == 'categorical' or deterministic:
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg)
        z_kp = torch.zeros(x.size(0), 1, 3, device=x.device, dtype=x.dtype)
        return {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, deterministic: bool = False, timesteps: int = None):
        return self.encode_all(x, mask=mask, deterministic=deterministic, timesteps=timesteps)
