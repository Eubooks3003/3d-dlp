# modules/point_cloud_modules/bg/bg_encoder_pc.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util_func import reparameterize

class MLP1D(nn.Module):
    """Tiny PointNet-ish block applied per point (last dim)."""
    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [..., in_ch]
        return self.net(x)


def masked_global_pool(feats: torch.Tensor,
                       valid: torch.Tensor,
                       mode: str = "max") -> torch.Tensor:
    """
    feats: [B, N, C]
    valid: [B, N] bool (True=use)
    returns: [B, C]
    """
    if mode == "max":
        v = feats.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        out = v.max(dim=1).values
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    elif mode == "mean":
        v = feats * valid.unsqueeze(-1).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        out = v.sum(dim=1) / denom
    else:
        raise ValueError("pool must be 'max' or 'mean'")
    return out


class BgEncoderPC(nn.Module):
    def __init__(self,
                 feature_dim: int = 16,
                 in_feat: int = 0,
                 hidden: int = 128,
                 pool: str = "max",
                 features_dist: str = "gauss",
                 interaction_features: bool = False,
                 timestep_horizon: int = 1,
                 add_particle_temp_embed: bool = False,
                 init_std: float = 0.2,
                 output_logvar: bool = True,
                 init_zero_bias: bool = True):
        super().__init__()
        assert pool in ("max", "mean")
        assert features_dist in ("gauss",), "only 'gauss' is implemented"
        self.feature_dim = feature_dim
        self.in_feat = in_feat
        self.hidden = hidden
        self.pool = pool
        self.features_dist = features_dist
        self.interaction_features = interaction_features
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.add_particle_temp_embed = add_particle_temp_embed
        self.output_logvar = output_logvar
        self.init_zero_bias = init_zero_bias

        in_ch = 3 + in_feat  # xyz + optional extras

        # Per-point backbone
        self.per_point = MLP1D(in_ch, hidden)

        # Optional temporal embedding (applied after pooling, like a small bias per time step)
        if self.add_particle_temp_embed and self.timestep_horizon > 1:
            self.temp_embed = nn.Parameter(
                init_std * torch.randn(1, self.timestep_horizon, hidden)
            )
        else:
            self.temp_embed = None

        # Projection to latent
        self.to_mu = nn.Linear(hidden, feature_dim)
        self.to_logvar = nn.Linear(hidden, feature_dim) if output_logvar else nn.Identity()

        if init_zero_bias:
            if self.to_mu.bias is not None:
                nn.init.constant_(self.to_mu.bias, 0.0)
            if output_logvar and self.to_logvar.bias is not None:
                nn.init.constant_(self.to_logvar.bias, math.log(0.2 ** 2))

        # info string
        self._info = (f'BgEncoderPC: feat_dim={feature_dim}, per-point hidden={hidden}, '
                      f'pool={pool}, in_ch=3+{in_feat}, '
                      f"temp_embed={'on' if self.temp_embed is not None else 'off'}, "
                      f"dist={'gauss(mu,logvar)' if output_logvar else 'deterministic'}")
        self.init_weights()

    @torch.no_grad()
    def info(self) -> str:
        return self._info
    
    def init_weights(self):
        """
        Roughly mirrors the spirit of 2D BgEncoder init:
        - Linear layers: Xavier init for weights, zero bias (unless set explicitly).
        - Last logvar bias to a small variance prior.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and self.init_zero_bias:
                    nn.init.constant_(m.bias, 0.0)

        # Set a mild prior for logvar head (if present)
        if isinstance(self.to_logvar, nn.Linear) and self.to_logvar.bias is not None:
            nn.init.constant_(self.to_logvar.bias, math.log(0.2 ** 2))

    def encode_bg_features(self,
                           points: torch.Tensor,              # [B,N,3(+F)]
                           masks: Optional[torch.Tensor] = None,   # [B,N] valid points
                           bg_mask: Optional[torch.Tensor] = None, # [B,N] keep as bg
                           timesteps: Optional[int] = None):
        B, N, C = points.shape
        assert C >= 3, f"expected last dim >=3 (xyz), got {C}"

        valid = torch.ones(B, N, dtype=torch.bool, device=points.device) if masks is None else masks.bool()
        if bg_mask is not None:
            valid = valid & bg_mask.bool()

        # per-point features -> pool
        feats = self.per_point(points)                    # [B,N,H]
        pooled = masked_global_pool(feats, valid, self.pool)  # [B,H]

        # optional temporal embed
        if self.temp_embed is not None and timesteps is not None:
            t = min(int(timesteps), self.timestep_horizon)
            pooled = pooled + self.temp_embed[:, t-1]     # [B,H] add per-time bias

        # to latent
        mu_bg = self.to_mu(pooled)                        # [B,D]
        logvar_bg = self.to_logvar(pooled) if self.output_logvar else None
        return mu_bg, logvar_bg

    def encode_all(self,
                   points: torch.Tensor,
                   masks: Optional[torch.Tensor] = None,
                   bg_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False,
                   timesteps: Optional[int] = None):
        mu_bg, logvar_bg = self.encode_bg_features(points, masks, bg_mask, timesteps)
        if self.interaction_features or deterministic or (logvar_bg is None):
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg)
        # For API parity with 2D, we also return a dummy z_kp (not used in PC bg)
        z_kp = torch.zeros(mu_bg.shape[0], 1, 3, device=points.device, dtype=torch.float)
        return {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}

    def forward(self,
                points: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                bg_mask: Optional[torch.Tensor] = None,
                deterministic: bool = False,
                timesteps: Optional[int] = None):
        return self.encode_all(points, mask, bg_mask, deterministic, timesteps)
