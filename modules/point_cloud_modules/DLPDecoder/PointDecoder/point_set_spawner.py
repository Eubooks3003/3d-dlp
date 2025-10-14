import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: [B,K,3] axis-angle
    returns R: [B,K,3,3]
    """
    eps = 1e-9
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)  # [B,K,1]
    axis = aa / angle
    x, y, z = axis.unbind(-1)  # each [B,K]
    c = torch.cos(angle)[..., 0]
    s = torch.sin(angle)[..., 0]
    C = 1.0 - c

    R = torch.stack([
        c + x * x * C,     x * y * C - z * s,  x * z * C + y * s,
        y * x * C + z * s, c + y * y * C,      y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s,  c + z * z * C
    ], dim=-1).reshape(*aa.shape[:2], 3, 3)
    return R


# --- small sinusoidal time-embedding (like diffusion) ---
def timestep_embed_scalar(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [T] or [B,K] scalar in [0,1]
    returns: [*, dim]
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / (half - 1 + 1e-9)))
    args  = t.unsqueeze(-1) * freqs  # [..., half]
    emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0,1))
    return emb

class PointSetSpawnerNSP(nn.Module):
    """
    Point-NSP-style spawn head with:
      - bounded per-scale radii (tanh * r_s)
      - small T-step denoising conditioned on z_feat and timestep
      - optional per-point color and confidence
    Keeps your forward signature and outputs.
    """
    def __init__(self,
                 features_dim: int,
                 n_scales: int = 2,
                 points_per_scale: Tuple[int, ...] = (128, 256),
                 hidden: int = 256,
                 predict_color: bool = False,
                 use_rotation: bool = True,
                 scale_activation: str = "sigmoid",
                 color_activation: str = "sigmoid",
                 n_steps: int = 4,                    # NSP denoise steps (2–6 recommended)
                 time_dim: int = 64,
                 predict_confidence: bool = True):
        super().__init__()
        assert n_scales == len(points_per_scale)
        self.F = features_dim
        self.S = n_scales
        self.Ms = tuple(points_per_scale)
        self.predict_color = predict_color
        self.use_rotation = use_rotation
        self.scale_activation = scale_activation
        self.color_activation = color_activation
        self.n_steps = n_steps
        self.time_dim = time_dim
        self.predict_confidence = predict_confidence

        # slot feature pre-embed
        self.seed = nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(inplace=True))
        # learnable per-scale radii (meters): start roomy, we will anneal in training
        init_rs = torch.tensor([0.03, 0.01][:self.S], dtype=torch.float32)  # e.g., 3cm & 1cm
        self.scale_radius = nn.Parameter(torch.log(torch.exp(init_rs) - 1.0))

        # initial offsets per scale (deterministic, one-shot like your original xyz_heads)
        self.init_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(True), nn.Linear(hidden, m*3))
            for m in self.Ms
        ])

        # per-scale NSP epsilon heads: point-wise MLP on [u, h, t_emb]
        # implemented as linear layers on the last dim; we flatten points, apply, then reshape
        self.eps_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 + hidden + time_dim, hidden), nn.ReLU(True),
                nn.Linear(hidden, hidden), nn.ReLU(True),
                nn.Linear(hidden, 3)
            ) for _ in self.Ms
        ])

        # optional per-point confidence heads (used at the final step)
        self.conf_heads = nn.ModuleList([
            (nn.Sequential(
                nn.Linear(3 + hidden, hidden), nn.ReLU(True),
                nn.Linear(hidden, 1), nn.Sigmoid()
            ) if predict_confidence else None) for _ in self.Ms
        ])

        # optional color heads (same as your current)
        self.rgb_heads = nn.ModuleList([
            (nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(True),
                nn.Linear(hidden, m*3)
            ) if predict_color else None) for m in self.Ms
        ])

        # per-object rotation & scale heads (same behavior as yours)
        self.rot_head = (nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(True), nn.Linear(hidden, 3))
                         if use_rotation else None)
        self.scale_head = nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(True), nn.Linear(hidden, 3))

        # fallback gating if obj_on not provided
        self.gate_head = nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(True), nn.Linear(hidden, 1))

        # guidance controls (optional; set via set_guidance)
        self._use_guidance = False
        self._guidance_weight = 0.0
        self._guidance_fn = None  # callable(u_world->[B,K,M,3] -> returns grad wrt u_local)

    def set_guidance(self, fn, weight: float):
        """
        Plug-and-play guidance (optional): during denoising, we will subtract 'weight * grad'
        where grad = fn(u_world).detach().
        'fn' must accept world-space points and return same-shaped gradient tensor.
        """
        self._use_guidance = (fn is not None) and (weight > 0)
        self._guidance_weight = float(weight)
        self._guidance_fn = fn

    def _act_color(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) if self.color_activation == "sigmoid" else torch.tanh(x)

    def forward(self,
                z_pos: torch.Tensor,        # [B,K,3]
                z_scale: torch.Tensor,      # [B,K,3]
                z_feat: torch.Tensor,       # [B,K,F]
                z_depth: Optional[torch.Tensor] = None,
                z_obj_on: Optional[torch.Tensor] = None
                ) -> dict:
        B, K, _ = z_pos.shape
        device = z_pos.device

        # slot embeddings
        h = self.seed(z_feat)  # [B,K,H]

        # ---- initialize canonical offsets per scale (bounded) ----
        u_scales, rgb_scales, w_scales = [], [], []
        for s, m in enumerate(self.Ms):
            r_s = F.softplus(self.scale_radius[s]) + 1e-4         # scalar radius (>0)
            raw = self.init_heads[s](h).view(B, K, m, 3)          # [B,K,m,3]
            u = torch.tanh(raw) * r_s                             # bounded local offsets
            u_scales.append(u)

            if self.predict_color and (self.rgb_heads[s] is not None):
                rgb = self.rgb_heads[s](h).view(B, K, m, 3)       # unactivated; color later
                rgb_scales.append(rgb)

        # ---- T-step NSP denoising in canonical frame ----
        # schedule (simple linear in [0,1]) for time embedding
        if self.n_steps > 0:
            # precompute time embeddings per step
            t_list = torch.linspace(1.0, 0.0, steps=self.n_steps, device=device)
            t_emb_list = [timestep_embed_scalar(t_list[i], self.time_dim) for i in range(self.n_steps)]  # [time_dim]
            # per-step alphas (you can tune/anneal)
            alphas = torch.linspace(0.6, 0.3, steps=self.n_steps, device=device)  # coarse->fine

            for i in range(self.n_steps):
                t_emb = t_emb_list[i].view(1, 1, 1, -1).expand(B, K, 1, -1)  # [B,K,1,E]
                for s, m in enumerate(self.Ms):
                    r_s = F.softplus(self.scale_radius[s]) + 1e-4
                    u = u_scales[s]                                          # [B,K,m,3]
                    # build per-point input: concat u, h, t_emb
                    h_exp  = h.unsqueeze(2).expand(B, K, m, h.shape[-1])
                    te_exp = t_emb.expand(B, K, m, self.time_dim)
                    inp = torch.cat([u, h_exp, te_exp], dim=-1)              # [B,K,m,3+H+E]
                    # point-wise eps prediction
                    eps = self.eps_heads[s](inp.reshape(B*K*m, -1)).view(B, K, m, 3)
                    # denoise step in canonical frame
                    u = u - alphas[i] * eps
                    # optional plug-and-play guidance (small world-space nudge)
                    if self._use_guidance and (self._guidance_fn is not None):
                        # project to world to compute guidance grad
                        u_world = self._to_world(u, z_pos, z_scale, z_feat, s_index=s)
                        grad = self._guidance_fn(u_world).detach()
                        # map grad back to canonical (here approx identity since local is rotated/scaled below)
                        u = u - self._guidance_weight * grad
                    # re-bound to local radius
                    u = torch.tanh(u / (r_s + 1e-9)) * r_s
                    u_scales[s] = u

        # ---- optional per-point confidence (at final step) ----
        for s, m in enumerate(self.Ms):
            if self.predict_confidence and (self.conf_heads[s] is not None):
                h_exp = h.unsqueeze(2).expand(B, K, m, h.shape[-1])
                w = self.conf_heads[s](torch.cat([u_scales[s], h_exp], dim=-1))   # [B,K,m,1], in (0,1)
            else:
                w = None
            w_scales.append(w)

        # ---- concat scales, then apply R/scale/pos, color & gate ----
        u_canon = torch.cat(u_scales, dim=2)                                   # [B,K,M,3]
        if self.use_rotation:
            aa = self.rot_head(z_feat)                                         # [B,K,3]
            R = axis_angle_to_matrix(aa)                                       # [B,K,3,3]
            u_rot = torch.matmul(u_canon, R.transpose(-1, -2))
        else:
            u_rot = u_canon

        # object scale (same semantics as your original; feel free to switch to exp + clamp)
        s_res = self.scale_head(z_feat)                                        # [B,K,3]
        if self.scale_activation == "sigmoid":
            s_base = torch.sigmoid(z_scale)
            s_res  = torch.sigmoid(s_res)
            scale  = s_base * (0.5 + s_res)
        else:
            scale  = torch.exp(z_scale + s_res)

        pts_world = u_rot * scale.unsqueeze(2) + z_pos.unsqueeze(2)            # [B,K,M,3]

        # gate
        gate = torch.sigmoid(self.gate_head(z_feat)) if (z_obj_on is None) else z_obj_on  # [B,K,1]

        # color
        if self.predict_color and len(rgb_scales) > 0:
            rgb = torch.cat(rgb_scales, dim=2)                                 # [B,K,M,3]
            rgb = self._act_color(rgb) * gate.unsqueeze(2)
        else:
            rgb = None

        # confidence (pack alongside for training; keep API keys the same)
        conf = None
        if any(w is not None for w in w_scales):
            conf = torch.cat([w if w is not None else torch.ones(B,K,m,1, device=device)
                              for w, m in zip(w_scales, self.Ms)], dim=2)      # [B,K,M,1]

        out = {
            "points_obj":  pts_world,      # [B,K,M,3]
            "rgb_obj":     rgb,            # [B,K,M,3] or None
            "obj_weights": gate,           # [B,K,1]
        }
        if conf is not None:
            out["point_conf"] = conf       # optional: [B,K,M,1] for loss weighting / pruning
        return out

    # helper: quick projection to world for guidance (uses current scale/rot/pos)
    def _to_world(self, u_local, z_pos, z_scale, z_feat, s_index: int):
        B,K,m,_ = u_local.shape
        if self.use_rotation:
            aa = self.rot_head(z_feat); R = axis_angle_to_matrix(aa)
            u_rot = torch.matmul(u_local, R.transpose(-1, -2))
        else:
            u_rot = u_local
        s_res = self.scale_head(z_feat)
        if self.scale_activation == "sigmoid":
            scale = torch.sigmoid(z_scale) * (0.5 + torch.sigmoid(s_res))
        else:
            scale = torch.exp(z_scale + s_res)
        return u_rot * scale.unsqueeze(2) + z_pos.unsqueeze(2)  # [B,K,m,3]
