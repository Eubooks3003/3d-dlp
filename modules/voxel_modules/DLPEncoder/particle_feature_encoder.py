import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.point_cloud_modules.vision_modules_3d import Encoder3D
from utils.util_func import reparameterize

# If not already imported in this file:
# from modules.voxel_modules.DLPEncoder.particle_attribute_encoder import spatial_transform_3d
# (or paste the same spatial_transform_3d here)

class ParticleFeaturesEncoder3D(nn.Module):
    """
    Voxel (3D) glimpse encoder:
      - crops a 3D cube around each keypoint with spatial_transform_3d
      - encodes with a small 3D CNN
      - projects to per-particle latent features (mu/logvar) with FCN (1x1x1 conv) or FC head
    Returns: mu_features, logvar_features, (optional) z_features, and cropped_objects.
    """

    def __init__(self,
                 anchor_size: float,            # not strictly needed when crop_size is provided
                 features_dim: int,
                 ch_in: int,                    # channels of dense voxel input
                 crop_size: tuple = (16, 16, 16),  # (d,h,w) of the cropped cube
                 base_ch: int = 32,
                 ch_mult=(1, 2, 3),
                 num_res_blocks: int = 2,
                 use_resblock: bool = True,
                 cnn_mid_blocks: bool = False,
                 use_attention: bool = False,
                 final_cnn_ch: int = 32,
                 output_logvar: bool = True,
                 hidden_dim: int = 256,
                 activation: str = "gelu",
                 timestep_horizon: int = 1,
                 add_particle_temp_embed: bool = False,
                 init_std: float = 0.2,
                 # init
                 init_zero_bias: bool = True,
                 init_conv_layers: bool = True,
                 init_conv_fg_std: float = 0.02,
                 ):
        super().__init__()
        self.anchor_size = float(anchor_size)
        self.features_dim = int(features_dim)
        self.ch_in = int(ch_in)
        self.crop_d, self.crop_h, self.crop_w = tuple(crop_size)
        self.output_logvar = bool(output_logvar)
        self.hidden_dim = int(hidden_dim)
        self.timestep_horizon = int(timestep_horizon)
        self.add_particle_temp_embed = bool(add_particle_temp_embed)
        self.activation = activation

        # 3D CNN over cropped cubes -> final_cnn_ch feature maps
        self.cnn3d = Encoder3D(
            in_channels=self.ch_in,
            ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            residual=use_resblock, dropout=0.0,
            use_attention=use_attention,
            mid_blocks=not cnn_mid_blocks,
            out_channels=final_cnn_ch
        )

        # Probe spatial size after CNN to decide FCN vs FC projection
        with torch.no_grad():
            dummy = torch.zeros(1, self.ch_in, self.crop_d, self.crop_h, self.crop_w)
            feat = self.cnn3d(dummy)
            if isinstance(feat, tuple):
                feat = feat[1]
            _, Cc, d2, h2, w2 = feat.shape
        self._feat_shape = (Cc, d2, h2, w2)              # CNN output shape
        fmap_voxels = d2 * h2 * w2

        # FCN projection (1x1x1 conv) if features_dim is a multiple of spatial size
        if self.features_dim % fmap_voxels == 0:
            self.projection_mode = "fcn"
            self.ch_feature_dim = max(self.features_dim // fmap_voxels, 1)
            z_out_channels = 2 * self.ch_feature_dim if self.output_logvar else self.ch_feature_dim
            self.to_latent = nn.Conv3d(in_channels=final_cnn_ch, out_channels=z_out_channels, kernel_size=1)

            if (self.timestep_horizon > 1) and self.add_particle_temp_embed:
                # (B, T, K, C, d2, h2, w2) add-on — same idea as 2D version
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, 1, final_cnn_ch, d2, h2, w2)
                )
            else:
                self.temp_embed = None

            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
            self._latent_flat = (self.ch_feature_dim, d2, h2, w2)

        else:
            # FC projection from flattened CNN features
            self.projection_mode = "fc"
            self.ch_feature_dim = Cc
            flat_in = Cc * d2 * h2 * w2
            self.to_latent = nn.Identity()
            self.to_mu = self._make_mlp(flat_in, self.features_dim, self.hidden_dim, self.activation)
            self.to_logvar = (self._make_mlp(flat_in, self.features_dim, self.hidden_dim, self.activation)
                              if self.output_logvar else nn.Identity())
            if (self.timestep_horizon > 1) and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, flat_in))
            else:
                self.temp_embed = None
            self._latent_flat = (flat_in,)

        # init conv biases
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, init_conv_fg_std)
                if init_zero_bias and (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

        self.info = (f"ParticleFeaturesEncoder3D: latent {self.features_dim} | "
                     f"CNN out: {self._feat_shape} (vox={fmap_voxels}) -> {self.projection_mode}")

    @staticmethod
    def _make_mlp(in_dim, out_dim, hidden_dim, activation="gelu"):
        Act = nn.GELU if activation == "gelu" else nn.ReLU
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), Act(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self,
                x_dense: torch.Tensor,     # [B, C, D, H, W]
                kp: torch.Tensor,          # [B, K, 3]  centers in [-1,1]^3
                z_scale: torch.Tensor = None,  # [B, K, 3] (unnormalized → sigmoid)
                timesteps: int = None,
                obj_on: torch.Tensor = None,   # [B, K] or [B, K, 1] gate
                deterministic: bool = False):
        B, C, D, H, W = x_dense.shape
        _, K, _ = kp.shape

        # repeat volume per kp -> [B*K, C, D, H, W]
        x_rep = x_dense[:, None].expand(B, K, C, D, H, W).reshape(B*K, C, D, H, W)

        # default scale = crop / grid (per-axis)
        if z_scale is None:
            zs = torch.tensor([self.crop_w/(W-1), self.crop_h/(H-1), self.crop_d/(D-1)],
                              device=x_dense.device, dtype=x_dense.dtype)  # (x,y,z) order
            z_scale = zs.view(1, 1, 3).expand(B, K, 3)
        else:
            z_scale = torch.sigmoid(z_scale)

        # flatten for spatial_transform_3d
        z_pos   = kp.reshape(B*K, 3)
        z_scale = z_scale.reshape(B*K, 3)

        out_dims = (B*K, C, self.crop_d, self.crop_h, self.crop_w)
        crops = spatial_transform_3d(
            x_rep, z_pos, z_scale, out_dims, padding_mode="border"
        )  # [B*K, C, d, h, w]

        # 3D CNN on crops
        feat = self.cnn3d(crops)
        if isinstance(feat, tuple):
            feat = feat[1]     # keep parity with your Encoders
        # obj_on gating on feature maps (broadcast)
        if obj_on is not None:
            gate = obj_on.view(B*K, 1, 1, 1, 1).to(feat.dtype)
            feat = feat * gate

        # (optional) temporal embedding (same semantics as 2D)
        if (self.timestep_horizon > 1) and (self.temp_embed is not None) and (timesteps is not None):
            # reshape to [B, T, K, C, d, h, w] at callsite if you truly pack multi-T
            # here we keep per-call T=timesteps, but typical training feeds one T
            # so this is usually a no-op unless you batch time inside B.
            pass

        if self.projection_mode == "fcn":
            # 1x1x1 conv → [B*K, zC, d2, h2, w2]
            zmap = self.to_latent(feat)
            # split mu/logvar channels
            if self.output_logvar:
                ch = zmap.shape[1] // 2
                mu_map, logvar_map = zmap[:, :ch], zmap[:, ch:]
            else:
                mu_map, logvar_map = zmap, None

            # flatten per particle -> [B, K, F]
            mu_features = mu_map.flatten(2).reshape(B, K, -1)
            if self.output_logvar:
                logvar_features = logvar_map.flatten(2).reshape(B, K, -1)
            else:
                logvar_features = None

        else:
            # FC projection from flattened CNN output
            flat = feat.flatten(1)                       # [B*K, flat_in]
            if (self.timestep_horizon > 1) and (self.temp_embed is not None) and (timesteps is not None):
                # add temporal embedding if you actually pack time
                pass
            mu_features = self.to_mu(flat).reshape(B, K, -1)
            logvar_features = (self.to_logvar(flat).reshape(B, K, -1)
                               if self.output_logvar else None)

        # (optional) reparameterize like your point encoder (for consistency)
        if (not deterministic) and (logvar_features is not None):
            z_features = reparameterize(mu_features, logvar_features)
        else:
            z_features = mu_features

        # reshape crops to [B, K, C, d, h, w] for logging
        crops = crops.view(B, K, C, self.crop_d, self.crop_h, self.crop_w)

        return {
            "mu_features":     mu_features,        # [B, K, F]
            "logvar_features": logvar_features,    # [B, K, F] or None
            "z_features":      z_features,         # [B, K, F]
            "cropped_objects": crops               # [B, K, C, d, h, w]
        }
