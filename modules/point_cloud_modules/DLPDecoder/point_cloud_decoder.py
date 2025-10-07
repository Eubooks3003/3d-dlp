import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: [B,K,3] axis-angle
    returns R: [B,K,3,3]
    """
    # small-angle safe Rodrigues
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True) + 1e-9  # [B,K,1]
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    c = torch.cos(angle)[..., 0]
    s = torch.sin(angle)[..., 0]
    C = 1 - c
    R = torch.stack([
        c + x*x*C,       x*y*C - z*s,     x*z*C + y*s,
        y*x*C + z*s,     c + y*y*C,       y*z*C - x*s,
        z*x*C - y*s,     z*y*C + x*s,     c + z*z*C
    ], dim=-1).reshape(*aa.shape[:2], 3, 3)
    return R

class PointCloudDecoderParticles(nn.Module):
    """
    Particle-wise point cloud decoder.
    For each particle k:
      - start from a canonical template of M points in [-1,1]^3
      - deform with an MLP conditioned on z_features[k]
      - apply diag(scale) * R * (...) + translation
      - (optional) predict per-point RGB
    """
    def __init__(self,
                 features_dim: int,
                 points_per_obj: int = 256,
                 template: str = "sphere",        # {'sphere','cube','learned'}
                 learn_template: bool = False,
                 predict_color: bool = False,
                 hidden: int = 256,
                 use_rotation: bool = True,
                 scale_activation: str = "sigmoid",  # {'sigmoid','exp'}
                 color_activation: str = "sigmoid"):
        super().__init__()
        self.F = features_dim
        self.M = points_per_obj
        self.predict_color = predict_color
        self.use_rotation = use_rotation
        self.scale_activation = scale_activation
        self.color_activation = color_activation

        # Canonical template
        tpl = self._make_template(self.M, template)
        if learn_template or template == "learned":
            self.template = nn.Parameter(tpl)    # [1,1,M,3]
        else:
            self.register_buffer("template", tpl, persistent=False)

        # Deformation network: f([p_canon, feat]) -> delta_p (and maybe color)
        in_dim = 3 + self.F
        out_dim = 3 + (3 if predict_color else 0)
        self.deform = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

        # Optional rotation head (axis-angle)
        if self.use_rotation:
            self.rot_head = nn.Sequential(
                nn.Linear(self.F, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 3)
            )
        else:
            self.rot_head = None

        # Optional per-object scale refinement (multiplicative)
        self.scale_head = nn.Sequential(
            nn.Linear(self.F, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3)
        )

        # Optional per-object gating if you have z_obj_on elsewhere; still expose a learned fallback
        self.gate_head = nn.Sequential(
            nn.Linear(self.F, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    @staticmethod
    def _make_template(M: int, which: str) -> torch.Tensor:
        # returns [1,1,M,3]
        if which == "sphere":
            # fibonacci sphere
            idx = torch.arange(M, dtype=torch.float32)
            phi = (1 + 5 ** 0.5) / 2
            z = 1 - 2 * (idx + 0.5) / M
            r = torch.sqrt(torch.clamp(1 - z * z, min=0))
            theta = 2 * torch.pi * idx / phi
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            pts = torch.stack([x, y, z], dim=-1)
        elif which == "cube":
            # random points in cube [-1,1]^3 (fixed seed-ish)
            g = torch.Generator().manual_seed(0)
            pts = torch.rand((M, 3), generator=g) * 2 - 1
        else:
            # learned placeholder (init small)
            pts = 0.01 * torch.randn(M, 3)
        return pts.view(1, 1, M, 3)

    def forward(self,
                z_pos: torch.Tensor,        # [B,K,3]
                z_scale: torch.Tensor,      # [B,K,3] (logits or log-scale)
                z_feat: torch.Tensor,       # [B,K,F]
                z_obj_on: Optional[torch.Tensor] = None,  # [B,K,1] in [0,1]
                z_depth: Optional[torch.Tensor] = None    # [B,K,1], near small -> neg, far large -> pos (your convention)
                ) -> dict:
        B, K, _ = z_pos.shape
        device = z_pos.device

        # ----- template & features -----
        tpl  = self.template.to(device).expand(B, K, self.M, 3)            # [B,K,M,3]
        feat = z_feat.unsqueeze(2).expand(B, K, self.M, self.F)            # [B,K,M,F]

        # ----- deform canonical points -----
        x_in  = torch.cat([tpl, feat], dim=-1)                             # [B,K,M,3+F]
        x_out = self.deform(x_in)                                          # [B,K,M,3(+3)]
        if self.predict_color:
            delta_p, color = torch.split(x_out, [3, 3], dim=-1)
        else:
            delta_p, color = x_out, None

        p_canon = tpl + delta_p                                            # [B,K,M,3]

        # ----- (optional) rotation -----
        if self.use_rotation:
            aa = self.rot_head(z_feat)                                     # [B,K,3]
            R  = axis_angle_to_matrix(aa)                                  # [B,K,3,3]
            p_rot = torch.matmul(p_canon, R.transpose(-1, -2))             # [B,K,M,3]
        else:
            p_rot = p_canon

        # ----- scale from z_scale (+ learned residual) -----
        scale_res = self.scale_head(z_feat)                                # [B,K,3]
        if self.scale_activation == "sigmoid":
            s_base = torch.sigmoid(z_scale)                                 # [B,K,3]
            s_res  = torch.sigmoid(scale_res)
            s = s_base * (0.5 + s_res)                                     # keep magnitude reasonable
        else:
            s = torch.exp(z_scale + scale_res)

        # ----- depth-aware modulation (optional) -----
        # depth_factor in (min_depth_scale, 1]; farther -> closer to min_depth_scale
        if z_depth is not None:
            # flip sign so "larger depth" (farther) => smaller factor
            depth_factor = self.min_depth_scale + (1.0 - self.min_depth_scale) * torch.sigmoid(-z_depth)  # [B,K,1]
        else:
            depth_factor = torch.ones(B, K, 1, device=device)

        # apply depth to size (optional; comment out to disable)
        s = s * depth_factor                                               # [B,K,3]

        p_scaled = p_rot * s.unsqueeze(2)                                  # [B,K,M,3]

        # ----- translate to world -----
        p_world = p_scaled + z_pos.unsqueeze(2)                            # [B,K,M,3]

        # ----- gating (obj_on) -----
        if z_obj_on is None:
            gate = torch.sigmoid(self.gate_head(z_feat))                   # [B,K,1]
        else:
            gate = z_obj_on

        # optionally damp by depth as well (uncomment if desired):
        # gate = gate * depth_factor.clamp(0, 1)

        # ----- color (optional) -----
        if color is not None:
            rgb = torch.sigmoid(color) if self.color_activation == "sigmoid" else torch.tanh(color)
            rgb = rgb * gate.unsqueeze(2)                                   # [B,K,M,3]
        else:
            rgb = None

        # ----- pack -----
        points_obj   = p_world                                             # [B,K,M,3]
        points_recon = points_obj.reshape(B, K * self.M, 3)                # [B,KM,3]
        rgb_recon    = rgb.reshape(B, K * self.M, 3) if rgb is not None else None

        return {
            "points_obj":   points_obj,     # [B,K,M,3] per-object point sets in world coords
            "points_recon": points_recon,   # [B,KM,3] concatenated object points (no bg)
            "rgb_recon":    rgb_recon,      # [B,KM,3] per-point color or None
            "obj_weights":  gate,           # [B,K,1] per-object on/off gate (0..1)
            "depth_factor": depth_factor,   # [B,K,1] (diagnostics; how depth scaled size)
        }

    def info(self) -> str:
        return (f"PointCloudDecoderParticles(M={self.M}, F={self.F}, "
                f"predict_color={self.predict_color}, use_rotation={self.use_rotation})")
