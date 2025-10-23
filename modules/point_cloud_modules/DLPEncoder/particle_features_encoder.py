import torch
import torch.nn as nn
from typing import Optional
import math
import numpy as np
from utils.util_func import reparameterize

# TODO: eliminate the overlap between this and particle attribute encoder
def soft_knn_over_topM(points, kp, mask=None, z_scale=None, M=128, tau=0.2,
                       base_radius=0.25, clamp_after_st=True):
    """
    points:  [B,N,3(+F)]
    kp:      [B,K,3]
    mask:    [B,N] (bool) True=valid
    z_scale: [B,K,3] (optional), scales receptive field via sigmoid mean
    Returns:
      x_pool: [B,K,4+Fin] pooled soft neighborhood features (rel, r_norm, extras)
    """
    B, N, C = points.shape
    K = kp.shape[1]

    # 1) coarse candidate set (non-diff selection)
    d = torch.cdist(kp, points[..., :3])                      # [B,K,N]
    if mask is not None:
        d = d + (~mask).unsqueeze(1) * 1e6
    M_eff = min(M, N)
    cand_idx = d.topk(k=M_eff, dim=-1, largest=False).indices # [B,K,M]
    # gather candidates
    idx_exp = cand_idx.unsqueeze(-1).expand(B, K, M_eff, C)
    neigh = torch.gather(points.unsqueeze(1).expand(B, K, N, C), 2, idx_exp)  # [B,K,M,C]

    # 2) differentiable weighting within candidates
    if z_scale is not None:
        s = torch.sigmoid(z_scale).mean(dim=-1, keepdim=True) # [B,K,1]
    else:
        s = torch.ones(B, K, 1, device=points.device, dtype=points.dtype)
    r = base_radius * s                                       # [B,K,1]

    rel = (neigh[..., :3] - kp.unsqueeze(2)) / (r.unsqueeze(2) + 1e-6)  # [B,K,M,3]
    if clamp_after_st:
        rel = rel.clamp_(-1, 1)

    r_norm = rel.norm(dim=-1, keepdim=True)                   # [B,K,M,1]
    feats = [rel, r_norm]
    if C > 3:
        feats.append(neigh[..., 3:])
    x_all = torch.cat(feats, dim=-1)                          # [B,K,M,4+Fin]

    logits = -(rel.square().sum(-1)) / (tau + 1e-6)           # [B,K,M]
    w = torch.softmax(logits, dim=-1).unsqueeze(-1)           # [B,K,M,1]
    x_pool = (w * x_all).sum(dim=2)                           # [B,K,4+Fin]
    return x_pool



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
            points: torch.Tensor,           # [B,N,3(+F)]
            kp: torch.Tensor,               # [B,K,3]
            z_scale: Optional[torch.Tensor] = None,  # [B,K,3] logits (optional)
            deterministic: bool = False,
            obj_on: Optional[torch.Tensor] = None,   # [B,K] or [B,K,1]
            mask: Optional[torch.Tensor] = None,     # [B,N] True=valid
            *,
            point_tile_ids: Optional[torch.Tensor] = None,  # [B,N] or None if soft path
            kp_tile_ids: Optional[torch.Tensor] = None,      # [B,K] or None if soft path
            tau: float = 0.2,
            soft_weights: Optional[torch.Tensor] = None,     # [B,N,K] (training soft-ownership)
            keep_w: Optional[torch.Tensor] = None,           # [B,K] relaxed Top-K weights
            eps_tile_bleed: float = 0.02) -> dict:

        B, N, C = points.shape
        assert kp.dim() == 3 and kp.size(-1) == 3, f"kp must be [B,K,3], got {tuple(kp.shape)}"
        K = kp.size(1)

        device, dtype = points.device, points.dtype

        # per-KP receptive radius
        if z_scale is not None:
            s = torch.sigmoid(z_scale).mean(dim=-1, keepdim=True)   # [B,K,1]
        else:
            s = torch.ones(B, K, 1, device=device, dtype=dtype)
        r = self.base_radius * s                                     # [B,K,1]

        # relative coords (vectorized)
        pts_xyz = points[..., :3].unsqueeze(1).expand(B, K, N, 3)    # [B,K,N,3]
        rel = (pts_xyz - kp.unsqueeze(2)) / (r.unsqueeze(2) + 1e-6)  # [B,K,N,3]
        if self.clamp_after_st:
            rel = rel.clamp_(-1, 1)

        # base per-point features
        r_norm = rel.norm(dim=-1, keepdim=True)                      # [B,K,N,1]
        feats = [rel, r_norm]
        if C > 3:
            feats.append(points[..., 3:].unsqueeze(1).expand(B, K, N, C-3))
        x_all = torch.cat(feats, dim=-1)                             # [B,K,N,4+Fin]

        # valid points
        if mask is not None:
            valid_pts = mask.unsqueeze(1)                             # [B,1,N]
        else:
            valid_pts = torch.ones(B, 1, N, device=device, dtype=torch.bool)

        # ---------------- weighting ----------------
        if soft_weights is not None:
            # TRAIN-SOFT: use provided [B,N,K] ownership
            assert soft_weights.shape == (B, N, K), f"soft_weights must be [B,N,K], got {tuple(soft_weights.shape)}"
            w = soft_weights.permute(0, 2, 1).unsqueeze(-1)           # [B,K,N,1]
            w = w * valid_pts.unsqueeze(-1)
            denom = w.sum(dim=2, keepdim=True).clamp_min(1e-6)
            w = w / denom
        else:
            # TILE ROUTING (eval/fast) + tiny global bleed for gradient health
            assert point_tile_ids is not None and kp_tile_ids is not None, \
                "point_tile_ids/kp_tile_ids required when soft_weights is None"

            cand_mask = (point_tile_ids.unsqueeze(1) == kp_tile_ids.unsqueeze(-1))  # [B,K,N]
            cand_mask = cand_mask & valid_pts

            logits_local = -(rel.square().sum(dim=-1)) / (tau + 1e-6)               # [B,K,N]
            logits_local = logits_local.masked_fill(~cand_mask, float('-inf'))

            logits_global = -(rel.square().sum(dim=-1)) / (tau + 1e-6)
            logits_global = logits_global.masked_fill(~valid_pts, float('-inf'))

            w_local  = torch.softmax(logits_local, dim=-1).unsqueeze(-1)            # [B,K,N,1]
            w_global = torch.softmax(logits_global, dim=-1).unsqueeze(-1)           # [B,K,N,1]

            empty = ~cand_mask.any(dim=-1)                                          # [B,K]
            alpha = (~empty).float().unsqueeze(-1).unsqueeze(-1)                    # [B,K,1,1]
            w = alpha * ((1.0 - eps_tile_bleed) * w_local + eps_tile_bleed * w_global) \
                + (1.0 - alpha) * w_global

        # --------------- pooling & heads ---------------
        x = (w * x_all).sum(dim=2)                                                  # [B,K,4+Fin]
        x = self.block1(x)
        x = self.block2(x)
        mu = self.to_mu(x)                                                          # [B,K,F]
        logvar = self.to_logvar(x) if self.output_logvar else None

        # continuous gating for presence / relaxed Top-K
        gate = None
        if obj_on is not None:
            g = obj_on
            if g.dim() == 3 and g.size(-1) == 1: g = g[..., 0]
            gate = g.to(mu.dtype).clamp(0, 1)
        if keep_w is not None:
            gate = keep_w.to(mu.dtype) if gate is None else (gate * keep_w.to(mu.dtype))
        if gate is not None:
            mu = self._gate_with_null(mu, gate, self.null_feature_embed)

        if self.features_dist == 'categorical':
            z_features = None
        else:
            z_features = mu if self.interaction_features else (
                mu if (logvar is None or deterministic) else reparameterize(mu, logvar)
            )

        return {
            'mu_features':           mu,
            'logvar_features':       logvar,
            'z_features':            z_features,
            'mu_features_total':     mu,
            'logvar_features_total': logvar,
            'z_features_total':      z_features,
        }
