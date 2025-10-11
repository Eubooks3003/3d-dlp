import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


# -------------------------------------- #
class PointSetSpawner(nn.Module):
    """
    Point-NSP-style spawner: directly 'spawns' points per slot without a template.
    - Predict S scales of points around canonical origin.
    - Optional per-object rotation and scale.
    - Transform to world via diag(scale)*R*p + translation(z_pos).
    - Optional per-point color.
    """
    def __init__(self,
                 features_dim: int,
                 n_scales: int = 2,
                 points_per_scale: Tuple[int, ...] = (128, 256),
                 hidden: int = 256,
                 predict_color: bool = False,
                 use_rotation: bool = True,
                 scale_activation: str = "sigmoid",
                 color_activation: str = "sigmoid"):
        super().__init__()
        assert n_scales == len(points_per_scale), "n_scales must equal len(points_per_scale)"
        self.F = features_dim
        self.S = n_scales
        self.Ms = tuple(points_per_scale)
        self.predict_color = predict_color
        self.use_rotation = use_rotation
        self.scale_activation = scale_activation
        self.color_activation = color_activation

        # Per-scale heads operate on slot code h (derived from z_feat)
        self.seed = nn.Sequential(
            nn.Linear(self.F, hidden), nn.ReLU(inplace=True)
        )
        self.xyz_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, m * 3)
            ) for m in self.Ms
        ])
        self.rgb_heads = nn.ModuleList([
            (nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, m * 3)
            ) if predict_color else None)
            for m in self.Ms
        ])

        # Per-object rotation & scale heads
        self.rot_head = (nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 3))
                         if use_rotation else None)
        self.scale_head = nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 3))

        # Fallback gating if z_obj_on is not provided
        self.gate_head = nn.Sequential(nn.Linear(self.F, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def _act_color(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) if self.color_activation == "sigmoid" else torch.tanh(x)

    def forward(self,
                z_pos: torch.Tensor,        # [B,K,3]
                z_scale: torch.Tensor,      # [B,K,3] (logits or log-scale)
                z_feat: torch.Tensor,       # [B,K,F]
                z_depth: Optional[torch.Tensor] = None,   # [B,K,1] or None (unused here, but kept for API parity)
                z_obj_on: Optional[torch.Tensor] = None   # [B,K,1] in [0,1]
                ) -> dict:
        B, K, _ = z_pos.shape
        device = z_pos.device

        # Slot code h per (B,K)
        h = self.seed(z_feat)  # [B,K,H]

        # Spawn points per scale (canonical coords)
        pts_canon_list, rgb_list = [], []
        for s in range(self.S):
            m = self.Ms[s]
            xyz = self.xyz_heads[s](h)               # [B,K,m*3]
            xyz = xyz.view(B, K, m, 3)               # [B,K,m,3]
            pts_canon_list.append(xyz)

            if self.predict_color and (self.rgb_heads[s] is not None):
                rgb = self.rgb_heads[s](h).view(B, K, m, 3)  # [B,K,m,3]
                rgb_list.append(rgb)

        # Concatenate scales
        pts_canon = torch.cat(pts_canon_list, dim=2)         # [B,K,M,3]  (M=sum Ms)
        rgb = torch.cat(rgb_list, dim=2) if (self.predict_color and len(rgb_list) > 0) else None  # [B,K,M,3] or None

        # Rotation
        if self.use_rotation:
            aa = self.rot_head(z_feat)                       # [B,K,3]
            R = axis_angle_to_matrix(aa)                    # [B,K,3,3]
            pts_rot = torch.matmul(pts_canon, R.transpose(-1, -2))
        else:
            pts_rot = pts_canon

        # Scale (combine z_scale + learned residual)
        s_res = self.scale_head(z_feat)                      # [B,K,3]
        if self.scale_activation == "sigmoid":
            s_base = torch.sigmoid(z_scale)
            s_res = torch.sigmoid(s_res)
            scale = s_base * (0.5 + s_res)                  # keep in a reasonable range
        else:
            scale = torch.exp(z_scale + s_res)              # exp-param

        pts_scaled = pts_rot * scale.unsqueeze(2)           # [B,K,M,3]

        # Translate to world
        pts_world = pts_scaled + z_pos.unsqueeze(2)         # [B,K,M,3]

        # Gate (obj_on)
        if z_obj_on is None:
            gate = torch.sigmoid(self.gate_head(z_feat))    # [B,K,1]
        else:
            gate = z_obj_on

        # Color
        if rgb is not None:
            rgb = self._act_color(rgb) * gate.unsqueeze(2)  # [B,K,M,3]

        return {
            "points_obj":   pts_world,                      # [B,K,M,3]
            "rgb_obj":      rgb,                            # [B,K,M,3] or None
            "obj_weights":  gate,                           # [B,K,1]
        }
