import torch
import torch.nn as nn

class AlternativeSpatialSoftmax3D(nn.Module):
    """
    3D marginalizing spatial softmax (D,H,W) -> expected (z,y,x) in [-1,1].

    Input:
      heatmap: [B, K, D, H, W]  (K channels = per-kp heatmaps)
    Args:
      kp_range: tuple giving coord range (default [-1,1])
      temperature: softmax temperature
    Returns:
      kp: [B, K, 3] with (z, y, x) in [-1,1]
      If probs=True: also returns (sm_d, sm_h, sm_w) marginals
      If variance=True: also returns full covariance matrix [B, K, 3, 3]
    """
    def __init__(self, kp_range=(-1., 1.), temperature=1.0):
        super().__init__()
        self.kp_range = kp_range
        self.temperature = temperature

    def forward(self, heatmap, probs: bool = False, variance: bool = False):
        B, K, D, H, W = heatmap.shape

        # softmax over all voxels
        logits = heatmap.view(B, K, -1) / max(self.temperature, 1e-6)
        scores = torch.softmax(logits, dim=-1).view(B, K, D, H, W)  # [B,K,D,H,W]

        # coordinate axes in [-1,1]
        lo, hi = self.kp_range
        z_axis = torch.linspace(lo, hi, D, device=heatmap.device, dtype=heatmap.dtype).view(1,1,D,1,1)
        y_axis = torch.linspace(lo, hi, H, device=heatmap.device, dtype=heatmap.dtype).view(1,1,1,H,1)
        x_axis = torch.linspace(lo, hi, W, device=heatmap.device, dtype=heatmap.dtype).view(1,1,1,1,W)

        # marginals
        sm_d = scores.sum(dim=(-1, -2))      # [B,K,D]   (sum over H,W)
        sm_h = scores.sum(dim=(-1, -3))      # [B,K,H]   (sum over D,W)
        sm_w = scores.sum(dim=(-2, -3))      # [B,K,W]   (sum over D,H)

        # expectations E[z], E[y], E[x]
        Ez = (sm_d * z_axis.view(1,1,D)).sum(dim=-1)                  # [B,K]
        Ey = (sm_h * y_axis.view(1,1,H)).sum(dim=-1)                  # [B,K]
        Ex = (sm_w * x_axis.view(1,1,W)).sum(dim=-1)                  # [B,K]
        kp = torch.stack([Ez, Ey, Ex], dim=-1)                        # [B,K,3] (z,y,x)

        if variance or probs:
            # broadcasted coordinate fields
            Z = z_axis.expand(B, K, D, H, W)
            Y = y_axis.expand(B, K, D, H, W)
            X = x_axis.expand(B, K, D, H, W)

        if variance:
            # second moments
            Ez2 = (scores * (Z * Z)).sum(dim=(-1, -2, -3))            # [B,K]
            Ey2 = (scores * (Y * Y)).sum(dim=(-1, -2, -3))
            Ex2 = (scores * (X * X)).sum(dim=(-1, -2, -3))

            # cross moments
            Ezy = (scores * (Z * Y)).sum(dim=(-1, -2, -3))
            Ezx = (scores * (Z * X)).sum(dim=(-1, -2, -3))
            Eyx = (scores * (Y * X)).sum(dim=(-1, -2, -3))

            # variances and covariances: Var = E[u^2] - (E[u])^2, Cov = E[uv] - E[u]E[v]
            Var_z = (Ez2 - Ez * Ez).clamp_min(1e-8)
            Var_y = (Ey2 - Ey * Ey).clamp_min(1e-8)
            Var_x = (Ex2 - Ex * Ex).clamp_min(1e-8)
            Cov_zy = Ezy - Ez * Ey
            Cov_zx = Ezx - Ez * Ex
            Cov_yx = Eyx - Ey * Ex

            # full 3x3 covariance per kp
            cov = torch.zeros(B, K, 3, 3, device=heatmap.device, dtype=heatmap.dtype)
            cov[..., 0, 0] = Var_z
            cov[..., 1, 1] = Var_y
            cov[..., 2, 2] = Var_x
            cov[..., 0, 1] = cov[..., 1, 0] = Cov_zy
            cov[..., 0, 2] = cov[..., 2, 0] = Cov_zx
            cov[..., 1, 2] = cov[..., 2, 1] = Cov_yx

            if probs:
                return kp, cov, sm_d, sm_h, sm_w
            return kp, cov

        if probs:
            return kp, sm_d, sm_h, sm_w
        return kp
