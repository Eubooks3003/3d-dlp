import torch
import torch.nn as nn
from typing import Optional
import math
import numpy as np
from utils.util_func import reparameterize

# TODO: eliminate the overlap between this and particle attribute encoder
def safe_topk_knn(
    centers: torch.Tensor,   # [B, K, 3]
    points:  torch.Tensor,   # [B, N, 3(+F)]
    mask:    Optional[torch.Tensor],  # [B, N] (bool) True=valid
    k: int
) -> torch.Tensor:
    """
    Return KNN indices for each center (xyz distance only).

    - If `mask` is provided, invalid (padded) points are ignored.
    - If N < k, pads by repeating the last valid neighbor.
    """
    assert centers.dim() == 3 and centers.size(-1) >= 3, f"centers must be [B,K,3], got {tuple(centers.shape)}"
    assert points.dim()  == 3 and points.size(-1)  >= 3, f"points must be [B,N,3(+F)], got {tuple(points.shape)}"

    B, N, _ = points.shape
    K = centers.size(1)

    # pairwise distances in xyz
    d = torch.cdist(centers[..., :3], points[..., :3])  # [B,K,N]
    if mask is not None:
        assert mask.shape == (B, N), f"mask must be [B,N], got {tuple(mask.shape)}"
        d = d + (~mask).unsqueeze(1) * 1e6

    k_eff = min(k, N)
    idx = d.topk(k=k_eff, dim=-1, largest=False).indices  # [B,K,k_eff]

    if k_eff < k:
        pad = idx[..., -1:].expand(B, K, k - k_eff)
        idx = torch.cat([idx, pad], dim=-1)               # [B,K,k]

    return idx


class MLP1D(nn.Module):
    """
    Small MLP operating on the last dimension (no shared 1x1 conv assumptions).
    Expects input shaped [..., C_in] and returns [..., C_out].
    """
    def __init__(self, in_ch: int, out_ch: int, hidden: Optional[int] = None, act: str = "relu"):
        super().__init__()
        hidden = out_ch if hidden is None else hidden
        if act == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act == "gelu":
            act_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported act='{act}'")

        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden),
            act_layer,
            nn.Linear(hidden, out_ch),
            act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_ch]
        return self.net(x)


# --- only the differences vs your last version; replace your class with this one ---

class ParticleFeaturesEncoderPoint(nn.Module):
    def __init__(self,
                 anchor_size: float,
                 features_dim: int,
                 in_feat: int = 0,
                 k_neighbors: int = 128,
                 output_logvar: bool = True,
                 features_dist: str = 'gauss',
                 hidden: int = 128,
                 interaction_features: bool = False,
                 use_null_features_embed: bool = False,
                 embed_init_std: float = 0.02,
                 base_radius: float = 0.25,         # <- sets physical crop size before scale
                 clamp_after_st: bool = True        # <- clamp normalized coords to [-1,1]
                 ):
        super().__init__()
        assert features_dist in ('gauss', 'categorical'), "Only 'gauss' implemented here."
        self.features_dim = features_dim
        self.k = k_neighbors
        self.output_logvar = output_logvar
        self.features_dist = features_dist
        self.interaction_features = interaction_features
        self.use_null_features_embed = use_null_features_embed
        self.base_radius = base_radius
        self.clamp_after_st = clamp_after_st

        # input per neighbor: [x',y',z' (normalized), radius', (optional extra features)]
        in_ch = 4 + in_feat

        mid = hidden
        self.block1 = MLP1D(in_ch, mid)
        self.block2 = MLP1D(mid, mid)
        out_ch = mid

        self.to_mu = nn.Linear(out_ch, features_dim)
        self.to_logvar = nn.Linear(out_ch, features_dim) if output_logvar else nn.Identity()

        if use_null_features_embed:
            self.null_feature_embed = nn.Parameter(
                embed_init_std * torch.randn(1, 1, features_dim)
            )
        else:
            self.null_feature_embed = None
    def info(self) -> str:
        lines = [
            "ParticleFeaturesEncoderPoint",
            f"  features_dim            = {self.features_dim}",
            f"  k_neighbors             = {self.k}",
            f"  output_logvar           = {self.output_logvar}",
            f"  features_dist           = '{self.features_dist}'",
            f"  hidden                  = {next(self.block1.net[0].parameters()).shape[-1] if hasattr(self, 'block1') else 'N/A'}",
            f"  use_null_features_embed = {self.use_null_features_embed}",
            f"  interaction_features    = {self.interaction_features}",
            f"  base_radius             = {getattr(self, 'base_radius', 'N/A')}",
            f"  clamp_after_st          = {getattr(self, 'clamp_after_st', 'N/A')}",
        ]
        # infer extra per-point channels beyond xyz from first linear layer
        try:
            in_ch = self.block1.net[0].in_features
            extra_feat = max(in_ch - 4, 0)  # (we add [dx,dy,dz,r] internally)
            lines.append(f"  in_feat(extra per-point) = {extra_feat}")
        except Exception:
            pass
        return "\n".join(lines)
    @staticmethod
    def _gate_with_null(mu: torch.Tensor,
                        gate: torch.Tensor,
                        null_embed: Optional[torch.Tensor]) -> torch.Tensor:
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)  # [B,K,1]
        if null_embed is None:
            return mu * gate
        return gate * mu + (1.0 - gate) * null_embed

    def sample_gauss(self, mu: torch.Tensor, logvar: Optional[torch.Tensor],
                     deterministic: bool) -> torch.Tensor:
        if deterministic or (logvar is None):
            return mu
        return reparameterize(mu, logvar)

    def forward(self,
                points: torch.Tensor,     # [B,N,3(+F)]
                kp: torch.Tensor,          # [B,K,3]
                z_scale: Optional[torch.Tensor] = None,  # [B,K,3] logits, optional
                deterministic: bool = False,
                obj_on: Optional[torch.Tensor] = None,   # [B,K] or [B,K,1]
                mask: Optional[torch.Tensor] = None      # [B,N] True=valid
                ) -> dict:
        B, N, C = points.shape
        assert kp.dim() == 3 and kp.size(-1) == 3, f"kp must be [B,K,3], got {tuple(kp.shape)}"
        K = kp.size(1)

        # ---- 1) choose neighborhood ----
        idx = safe_topk_knn(kp, points, mask, self.k)        # [B,K,k]
        idx_exp_c = idx.unsqueeze(-1).expand(B, K, self.k, C)
        neigh = torch.gather(points.unsqueeze(1).expand(B, K, N, C), 2, idx_exp_c)  # [B,K,k,C]

        # ---- 2) 3D "spatial transform": translate & scale into a canonical cube ----
        # base radius scaled by z_scale (if provided)
        if z_scale is not None:
            # z_scale are logits -> turn into (0,1) with sigmoid, then average over xyz
            s = torch.sigmoid(z_scale).mean(dim=-1, keepdim=True)  # [B,K,1]
        else:
            s = torch.ones(B, K, 1, device=points.device)          # no scaling

        r = self.base_radius * s                                    # [B,K,1]
        rel = neigh[..., :3] - kp.unsqueeze(2)                      # [B,K,k,3]  (translate)
        # normalize to unit-ish cube by dividing by r, small epsilon for safety
        rel_norm = rel / (r.unsqueeze(2) + 1e-6)                    # [B,K,k,3]
        if self.clamp_after_st:
            rel_norm = rel_norm.clamp_(-1.0, 1.0)

        # optional: strict crop (keep only neighbors inside radius)
        # keep = (rel.norm(dim=-1, keepdim=True) <= (r.unsqueeze(2) + 1e-6)).float()   # [B,K,k,1]
        # rel_norm = rel_norm * keep

        # radius channel in normalized space
        r_norm = torch.linalg.norm(rel_norm, dim=-1, keepdim=True)  # [B,K,k,1]

        feats = [rel_norm, r_norm]
        if C > 3:
            feats.append(neigh[..., 3:])                             # propagate extra per-point features
        x = torch.cat(feats, dim=-1)                                 # [B,K,k,4+Fin]

        # ---- 3) PointNet-like encode + pool ----
        x = self.block1(x)
        x = self.block2(x)
        x = x.max(dim=2).values                                      # [B,K,hidden]

        # ---- 4) project to latent; optional obj_on gating ----
        mu = self.to_mu(x)
        logvar = self.to_logvar(x) if self.output_logvar else None

        if obj_on is not None:
            gate = (obj_on > 0.2).to(mu.dtype)
            mu = self._gate_with_null(mu, gate, self.null_feature_embed)

        # ---- 5) sample (or passthrough during interaction stage) ----
        if self.features_dist == 'categorical':
            z_features = None
        else:
            z_features = mu if self.interaction_features else self.sample_gauss(mu, logvar, deterministic)

        print("Encoded Logvar features is None? ", logvar is None)
        return {
            'mu_features':           mu,             # [B,K,F]
            'logvar_features':       logvar,         # [B,K,F] or None
            'z_features':            z_features,     # [B,K,F]
            'mu_features_total':     mu,
            'logvar_features_total': logvar,
            'z_features_total':      z_features,
        }
