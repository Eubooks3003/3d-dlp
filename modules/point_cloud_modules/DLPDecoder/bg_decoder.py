# modules/point_cloud_modules/bg_decoder_pc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BgDecoderPC(nn.Module):
    """
    Background point cloud generator.
    Decodes a global bg latent z_bg -> M_bg points in world coords ([-1,1]^3),
    optionally with per-point RGB.

    Outputs:
      - 'bg_points': [B, M_bg, 3]
      - 'bg_rgb':    [B, M_bg, 3] or None
    """
    def __init__(self,
                 learned_bg_feature_dim: int = 16,
                 points_bg: int = 2048,
                 hidden: int = 256,
                 predict_color: bool = False,
                 use_context: bool = False,
                 context_dim: int = 0,
                 template: str = "sphere",   # {'sphere','cube','gridxy'}
                 learn_template: bool = False,
                 color_activation: str = "sigmoid"):
        super().__init__()
        self.F = learned_bg_feature_dim
        self.M = points_bg
        self.predict_color = predict_color
        self.use_context = use_context and (context_dim > 0)
        self.color_activation = color_activation

        # Canonical template (bg-wide support)
        tpl = self._make_template(self.M, template)  # [1,M,3]
        if learn_template or template == "learned":
            self.template = nn.Parameter(tpl)
        else:
            self.register_buffer("template", tpl, persistent=False)

        in_dim = self.F + (context_dim if self.use_context else 0)
        out_dim = 3 + (3 if predict_color else 0)

        # simple MLP that broadcasts the same code across M template points
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        # point-wise head: takes [tpl, fused_code] -> delta + optional rgb
        self.head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

        # lightweight affine to nudge bg extent/pose
        self.affine_scale = nn.Linear(self.F, 3)   # diagonal scale
        self.affine_bias  = nn.Linear(self.F, 3)   # translation

        # info + init
        self._info = (f"BgDecoderPC(M_bg={self.M}, F={self.F}, "
                      f"predict_color={self.predict_color}, context={self.use_context})")
        self.init_weights()

    def init_weights(self):
        # keep default init mostly; optionally zero biases in last layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _make_template(M: int, which: str) -> torch.Tensor:
        # [1,M,3] roughly covering a unit volume/surface
        if which == "sphere":
            idx = torch.arange(M, dtype=torch.float32)
            phi = (1 + 5 ** 0.5) / 2
            z = 1 - 2 * (idx + 0.5) / M
            r = torch.sqrt(torch.clamp(1 - z * z, min=0))
            theta = 2 * torch.pi * idx / phi
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            pts = torch.stack([x, y, z], dim=-1)
        elif which == "gridxy":
            # square grid in XY with small Z jitter
            s = int(max(1, round(M ** 0.5)))
            gx, gy = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing='ij')
            pts = torch.stack([gx.flatten(), gy.flatten(),
                               0.02 * torch.randn(s * s)], dim=-1)
            if pts.shape[0] > M:
                pts = pts[:M]
            elif pts.shape[0] < M:
                pad = M - pts.shape[0]
                pts = torch.cat([pts, pts[:pad]], dim=0)
        else:  # 'cube' or fallback learned seed
            g = torch.Generator().manual_seed(0)
            pts = torch.rand((M, 3), generator=g) * 2 - 1
        return pts.unsqueeze(0)  # [1,M,3]

    def forward(self,
                z_bg: torch.Tensor,          # [B, F]

                ) -> dict:
        B = z_bg.shape[0]
        device = z_bg.device


        code = z_bg                                  # [B, F]

        h = self.fuse(code)                               # [B, H]
        h = h.unsqueeze(1).expand(B, self.M, h.size(-1))  # [B, M, H]

        # broadcast template
        tpl = self.template.to(device).expand(B, self.M, 3)  # [B,M,3]
        pt_in = torch.cat([tpl, h], dim=-1)                  # [B,M, H+3]
        out = self.head(pt_in)                               # [B,M, 3(+3)]
        if self.predict_color:
            delta, rgb = out.split([3, 3], dim=-1)
        else:
            delta, rgb = out, None

        # lightweight affine from z_bg
        s = torch.sigmoid(self.affine_scale(z_bg)) * 1.5     # keep bg reasonably large
        t = torch.tanh(self.affine_bias(z_bg))               # bias in [-1,1]
        p = (tpl + delta) * s.unsqueeze(1) + t.unsqueeze(1)  # [B,M,3], world-ish coords

        if rgb is not None:
            rgb = torch.sigmoid(rgb) if self.color_activation == "sigmoid" else torch.tanh(rgb)

        return {
            "bg_points": p,         # [B,M,3]
            "bg_rgb":    rgb,       # [B,M,3] or None
        }

    def info(self) -> str:
        return self._info
