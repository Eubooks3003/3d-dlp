import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util_func import reparameterize

def safe_topk_knn(centers: torch.Tensor,
                  points: torch.Tensor,
                  mask: Optional[torch.Tensor],
                  k: int) -> torch.Tensor:
    """
    centers: [B,K,3], points: [B,N,3(+F)], mask: [B,N] or None
    return idx: [B,K,k]
    """
    B, N, _ = points.shape
    K = centers.size(1)
    d2 = torch.cdist(centers[..., :3], points[..., :3])  # [B,K,N]
    if mask is not None:
        d2 = d2 + (~mask).unsqueeze(1) * 1e6
    k_eff = min(k, N)
    idx = d2.topk(k=k_eff, dim=-1, largest=False).indices
    if k_eff < k:
        pad = idx[..., -1:].expand(B, K, k - k_eff)
        idx = torch.cat([idx, pad], dim=-1)
    return idx

class MLP1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch), nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):  # x: [..., in_ch]
        return self.net(x)

class ParticleAttributeEncoderPoint(nn.Module):
    """
    Point-cloud attribute encoder (3D):
      - predicts per-particle offset μ/logvar (3D), scale μ/logvar (3D),
        optional obj_on (Beta) and optional depth latent.

    Inputs to forward():
      points : [B, N, 3(+F)]    coords in [-1,1]^3 (+ optional per-point feats)
      kp     : [B, K, 3]        anchor centers in [-1,1]^3
      mask   : [B, N] (bool)    True = valid, optional
      deterministic: bool       sampling flag for obj_on/depth

    Returns (keys match your 2D pipeline):
      {
        'mu','logvar',                # [B,K,3] offset stats
        'mu_scale','logvar_scale',    # [B,K,3] scale stats (logits/params as you prefer)
        'lobj_on_a','lobj_on_b','obj_on',  # raw head & convenience tensor
        'obj_on_a','obj_on_b','z_obj_on','mu_obj_on',  # Beta params & samples
        'mu_depth','logvar_depth',    # [B,K,1] (optional)
      }
    """
    def __init__(self,
                 n_particles: int,
                 in_feat: int = 0,            # extra per-point features beyond xyz
                 k_neighbors: int = 128,
                 hidden: int = 256,
                 kp_activation: str = 'tanh', # {'tanh','sigmoid'}
                 max_offset: float = 1.0,
                 with_scale: bool = True,
                 with_obj_on: bool = True,
                 with_depth: bool = False,
                 base_radius: float = 0.25,   # crop radius in normalized coords (scaled per kp internally)
                 clamp_after_st: bool = True,
                 obj_on_min: float = 1e-4,
                 obj_on_max: float = 100.0,
                 init_zero_bias: bool = True):
        super().__init__()
        self.n_particles   = n_particles
        self.k             = k_neighbors
        self.hidden        = hidden
        self.kp_activation = kp_activation
        self.max_offset    = max_offset
        self.with_scale    = with_scale
        self.with_obj_on   = with_obj_on
        self.with_depth    = with_depth
        self.base_radius   = base_radius
        self.clamp_after_st= clamp_after_st
        self.obj_on_min    = obj_on_min
        self.obj_on_max    = obj_on_max

        # per-neighbor feature: [dx',dy',dz', r', (+ extra per-point features)]
        in_ch = 4 + in_feat

        self.block1 = MLP1D(in_ch, hidden)
        self.block2 = MLP1D(hidden, hidden)

        # heads
        # offsets: [μx, logσx^2, μy, logσy^2, μz, logσz^2]
        self.offset_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6)
        )

        if with_scale:
            # scales: [μsx, logσsx^2, μsy, logσsy^2, μsz, logσsz^2]
            self.scale_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6)
            )
        else:
            # predict only μs (3), let caller set logvar if needed
            self.scale_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
            )

        if with_obj_on:
            self.obj_on_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1, bias=False)
            )
        else:
            self.obj_on_head = None

        if with_depth:
            # simple scalar depth latent per kp (radial or along-z depending on your decoder)
            self.depth_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)  # μ_depth, logσ^2_depth
            )
        else:
            self.depth_head = None

        # init last-layer biases similar to the 2D defaults
        with torch.no_grad():
            # offsets: set μ bias = 0, logvar bias small (0.01^2)
            self.offset_head[-1].bias[:] = 0.0
            # arrange bias as [μx, logσx^2, μy, logσy^2, μz, logσz^2]
            self.offset_head[-1].bias[1] = math.log(0.01**2)
            self.offset_head[-1].bias[3] = math.log(0.01**2)
            self.offset_head[-1].bias[5] = math.log(0.01**2)

            if with_scale:
                self.scale_head[-1].bias[:] = 0.0
                self.scale_head[-1].bias[1] = math.log(0.1**2)
                self.scale_head[-1].bias[3] = math.log(0.1**2)
                self.scale_head[-1].bias[5] = math.log(0.1**2)

            if with_obj_on and hasattr(self.obj_on_head[-1], 'bias') and self.obj_on_head[-1].bias is not None:
                self.obj_on_head[-1].bias.zero_()
    def info(self) -> str:
        lines = [
            "ParticleAttributeEncoderPoint",
            f"  n_particles   = {self.n_particles if hasattr(self, 'n_particles') else 'configured externally'}",
            f"  k_neighbors   = {self.k}",
            f"  hidden        = {next(self.mlp_offset.net[0].parameters()).shape[-1] if hasattr(self, 'mlp_offset') else 'N/A'}",
            f"  kp_activation = '{self.kp_activation}'",
            f"  max_offset    = {self.max_offset}",
            f"  with_scale    = {self.with_scale}",
            f"  with_obj_on   = {self.with_obj_on}",
            f"  with_depth    = {self.with_depth}",
            f"  base_radius   = {self.base_radius}",
            f"  clamp_after_st= {self.clamp_after_st}",
            f"  obj_on_min    = {self.obj_on_min}",
            f"  obj_on_max    = {self.obj_on_max}",
        ]
        # infer in_feat from first MLP input width if available:
        try:
            # we build per-neighbor feat as [dx,dy,dz,r,(+in_feat)]
            in_ch = self.mlp_offset.net[0].in_features
            extra_feat = max(in_ch - 4, 0)
            lines.append(f"  in_feat(extra per-point) = {extra_feat}")
        except Exception:
            pass
        return "\n".join(lines)
    def _gather_neighbors(self, points, kp, mask):
        """
        points: [B,N,C], kp: [B,K,3]
        return neigh [B,K,k,C]
        """
        B, N, C = points.shape
        K = kp.size(1)
        idx = safe_topk_knn(kp, points, mask, self.k)           # [B,K,k]
        idx_exp = idx.unsqueeze(-1).expand(B, K, self.k, C)     # [B,K,k,C]
        neigh = torch.gather(points.unsqueeze(1).expand(B, K, N, C), 2, idx_exp)
        return neigh, idx

    def forward(self,
                points: torch.Tensor,    # [B,N,3(+F)]
                kp: torch.Tensor,        # [B,K,3] anchor centers (z_base)
                mask: Optional[torch.Tensor] = None,
                timesteps: Optional[int] = None,
                deterministic: bool = False):
        B, N, C = points.shape
        assert kp.dim() == 3 and kp.size(-1) == 3, f"kp must be [B,K,3], got {tuple(kp.shape)}"
        K = kp.size(1)

        # 1) gather neighbors
        neigh, _ = self._gather_neighbors(points, kp, mask)     # [B,K,k,C]

        # 2) "spatial transform": translate to kp, scale by local radius
        #    use a fixed base radius here; if you have a prev scale estimate, you could multiply it in.
        r = self.base_radius * torch.ones(B, K, 1, device=points.device)  # [B,K,1]
        rel = neigh[..., :3] - kp.unsqueeze(2)                              # [B,K,k,3]
        rel_n = rel / (r.unsqueeze(2) + 1e-6)
        if self.clamp_after_st:
            rel_n = rel_n.clamp_(-1.0, 1.0)
        rch = torch.linalg.norm(rel_n, dim=-1, keepdim=True)                # [B,K,k,1]

        feats = [rel_n, rch]
        if C > 3:
            feats.append(neigh[..., 3:])                                     # propagate extra per-point feats
        x = torch.cat(feats, dim=-1)                                         # [B,K,k,4+Fin]

        # 3) encode + pool
        x = self.block1(x)
        x = self.block2(x)
        x = x.max(dim=2).values                                              # [B,K,H]

        # 4) heads
        off = self.offset_head(x)                                            # [B,K,6]
        mu, logvar = torch.chunk(off, 2, dim=-1)                             # [B,K,3] each

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        sc = self.scale_head(x)                                              # [B,K,6] or [B,K,3]
        if self.with_scale:
            mu_scale, logvar_scale = torch.chunk(sc, 2, dim=-1)              # [B,K,3] each
        else:
            mu_scale, logvar_scale = sc, None                                # [B,K,3], None

        if self.with_obj_on:
            # single-logit head; we keep a/b sym like your 2D code
            lobj = self.obj_on_head(x)                                       # [B,K,1]
            obj_on = lobj                                                    # alias for compatibility
            lobj_on_a = lobj_on_b = obj_on
            gate_a = lobj_on_a.sigmoid()
            obj_on_a = ((1 - gate_a) * self.obj_on_min + gate_a * self.obj_on_max).exp()
            gate_b = (1 - (lobj_on_b * 0 + lobj_on_a)).sigmoid()
            obj_on_b = ((1 - gate_b) * self.obj_on_min + gate_b * self.obj_on_max).exp()
            beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = beta_dist.mean
            z_obj_on = mu_obj_on if deterministic else beta_dist.rsample()
        else:
            lobj_on_a = lobj_on_b = obj_on = None
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        if self.with_depth:
            d = self.depth_head(x)                                           # [B,K,2]
            mu_depth, logvar_depth = torch.chunk(d, 2, dim=-1)               # [B,K,1] each
        else:
            mu_depth = logvar_depth = None

        return {
            'mu': mu, 'logvar': logvar,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
            'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
            'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
            'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
        }
