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
            kp: torch.Tensor,               # [B,K,3]  (same normalized frame as tiles)
            z_scale: Optional[torch.Tensor] = None,  # [B,K,3] logits (optional)
            deterministic: bool = False,
            obj_on: Optional[torch.Tensor] = None,   # [B,K] or [B,K,1]
            mask: Optional[torch.Tensor] = None,     # [B,N] True=valid
            *,
            point_tile_ids: torch.Tensor,            # [B,N] linear tile id per point
            kp_tile_ids: torch.Tensor,               # [B,K] linear tile id per kp
            tau: float = 0.2) -> dict:
        """
        Tile-routed soft pooling with robust empty-tile fallback (soft-topM).
        """
        B, N, C = points.shape
        assert kp.dim() == 3 and kp.size(-1) == 3, f"kp must be [B,K,3], got {tuple(kp.shape)}"
        K = kp.size(1)
        assert point_tile_ids.shape == (B, N), f"point_tile_ids must be [B,N], got {tuple(point_tile_ids.shape)}"
        assert kp_tile_ids.shape == (B, K),    f"kp_tile_ids must be [B,K], got {tuple(kp_tile_ids.shape)}"
        

        B, K = kp_tile_ids.shape
        nmatch = (point_tile_ids.unsqueeze(1) == kp_tile_ids.unsqueeze(-1)).sum(-1)  # [B,K]
        print(f"[route] cand points per kp: min={int(nmatch.min())} "
            f"mean={float(nmatch.float().mean()):.1f} "
            f"max={int(nmatch.max())} | empty_kp={(nmatch==0).sum().item()}/{B*K}")
        device, dtype = points.device, points.dtype

        # ---- per-KP receptive radius from z_scale (optional) ----
        if z_scale is not None:
            s = torch.sigmoid(z_scale).mean(dim=-1, keepdim=True)   # [B,K,1]
        else:
            s = torch.ones(B, K, 1, device=device, dtype=dtype)
        r = self.base_radius * s                                     # [B,K,1]

        # ---- fixed candidate set by tile id (no neighbor switching) ----
        cand_mask = (point_tile_ids.unsqueeze(1) == kp_tile_ids.unsqueeze(-1))    # [B,K,N]
        if mask is not None:
            cand_mask = cand_mask & mask.unsqueeze(1)                              # respect padding
        empty = ~cand_mask.any(dim=-1)                                            # [B,K]

        # ---- relative coords (once) ----
        pts_xyz = points[..., :3].unsqueeze(1).expand(B, K, N, 3)                 # [B,K,N,3]
        rel = (pts_xyz - kp.unsqueeze(2)) / (r.unsqueeze(2) + 1e-6)               # [B,K,N,3]
        if self.clamp_after_st:
            rel = rel.clamp_(-1, 1)

        # ---- logits for candidates; -inf elsewhere ----
        logits = -(rel.square().sum(dim=-1)) / (tau + 1e-6)                        # [B,K,N]
        logits = logits.masked_fill(~cand_mask, float('-inf'))

        # ---- fallback: if a kp's tile has no points, use soft-topM globally for that (b,k) only ----
        if empty.any():
            print("KP WITH NO POINTS")
            M = min(self.k, N)
            d2_all = rel.pow(2).sum(dim=-1)                                        # [B,K,N]
            if mask is not None:
                d2_all = d2_all.masked_fill(~mask.unsqueeze(1), float('inf'))
            topM_idx = d2_all.topk(k=M, dim=-1, largest=False).indices             # [B,K,M]
            vals = (-d2_all / (tau + 1e-6)).gather(-1, topM_idx)                   # [B,K,M]

            logits_fb = logits.new_full((B, K, N), float('-inf'))                  # [B,K,N]
            logits_fb.scatter_(-1, topM_idx, vals)                                 # fill only top-M

            e = empty.unsqueeze(-1)                                                # [B,K,1]
            logits = torch.where(e, logits_fb, logits)                             # row-wise select

        # ---- soft weights & pooling ----
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)                            # [B,K,N,1]
        r_norm = rel.norm(dim=-1, keepdim=True)                                    # [B,K,N,1]
        feats = [rel, r_norm]
        if C > 3:
            feats.append(points[..., 3:].unsqueeze(1).expand(B, K, N, C-3))
        x_all = torch.cat(feats, dim=-1)                                           # [B,K,N,4+Fin]
        x = (w * x_all).sum(dim=2)                                                 # [B,K,4+Fin]

        # ---- MLP heads ----
        x = self.block1(x)                                                         # [B,K,H]
        x = self.block2(x)                                                         # [B,K,H]
        mu = self.to_mu(x)                                                         # [B,K,F]
        logvar = self.to_logvar(x) if self.output_logvar else None

        if obj_on is not None:
            gate = (obj_on > 0.2).to(mu.dtype)
            mu = self._gate_with_null(mu, gate, self.null_feature_embed)

        if self.features_dist == 'categorical':
            z_features = None
        else:
            z_features = mu if self.interaction_features else self.sample_gauss(mu, logvar, deterministic)

        return {
            'mu_features':           mu,
            'logvar_features':       logvar,
            'z_features':            z_features,
            'mu_features_total':     mu,
            'logvar_features_total': logvar,
            'z_features_total':      z_features,
        }
