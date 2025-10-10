import torch
import torch.nn as nn

class GridVoxelizer(nn.Module):
    """
    Tri-linear splat from batched point clouds -> dense voxel grid (single-frame).
    Inputs:
      pts:   [B, N, C], C = 3 (xyz) or 3+F (xyz + F per-point features)
            xyz must be in [-1, 1]^3
      mask:  [B, N] boolean (optional), True = valid point
    Args:
      D,H,W: grid depth/height/width
      out_feat: number of channels to accumulate into the grid.
                If pts has features and (C-3) == out_feat, uses those per-point features.
                Otherwise uses a constant 1 channel per point.
      pooling: "sum" or "mean"
    Returns:
      vox:  [B, out_feat, D, H, W]
      meta: {"grid_shape": (D,H,W)}
    """
    def __init__(self, D=48, H=48, W=48, out_feat=1, pooling="mean"):
        super().__init__()
        assert pooling in {"sum", "mean"}
        self.D, self.H, self.W = D, H, W
        self.out_feat = out_feat
        self.pooling = pooling

    def forward(self, pts: torch.Tensor, mask: torch.Tensor = None, weights=None, with_moments=True):
        assert pts.dim() == 3 and pts.size(-1) >= 3, f"pts should be [B,N,3(+F)], got {tuple(pts.shape)}"
        B, N, C = pts.shape
        device = pts.device

        # mask
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask_f = mask.float()  # [B,N]

        if weights is None:
            weights = torch.ones(B, N, device=device)
        weights = weights * mask_f  # [B,N]

        # split coords / features
        xyz = pts[..., :3]  # [-1,1]
        if C > 3 and (C - 3) == self.out_feat:
            feat = pts[..., 3:]                       # [B,N,out_feat]
        else:
            feat = torch.ones(B, N, self.out_feat, device=device)  # [B,N,out_feat]


        # per-point features to splat
        if with_moments:
            # [1] density
            f_den = weights[:, :, None]                             # [B,N,1]
            # [2] first moments (weighted xyz)
            f_xyz = weights[:, :, None] * xyz                       # [B,N,3]
            # [3] second moments (weighted xyz^2)
            f_xyz2 = weights[:, :, None] * (xyz * xyz)              # [B,N,3]
            feat = torch.cat([f_den, f_xyz, f_xyz2], dim=-1)        # [B,N,7]
            Cg = 7
        else:
            feat = torch.ones(B, N, self.out_feat, device=device)   # fallback
            Cg = self.out_feat

        # map to voxel index space [0, size-1]
        px = (xyz[..., 0] + 1) * 0.5 * (self.W - 1)
        py = (xyz[..., 1] + 1) * 0.5 * (self.H - 1)
        pz = (xyz[..., 2] + 1) * 0.5 * (self.D - 1)

        x0 = px.floor().clamp_(0, self.W - 1); x1 = (x0 + 1).clamp_(0, self.W - 1)
        y0 = py.floor().clamp_(0, self.H - 1); y1 = (y0 + 1).clamp_(0, self.H - 1)
        z0 = pz.floor().clamp_(0, self.D - 1); z1 = (z0 + 1).clamp_(0, self.D - 1)

        wx = px - x0; wy = py - y0; wz = pz - z0
        w000 = (1-wx)*(1-wy)*(1-wz)
        w100 = (wx  )*(1-wy)*(1-wz)
        w010 = (1-wx)*(wy  )*(1-wz)
        w110 = (wx  )*(wy  )*(1-wz)
        w001 = (1-wx)*(1-wy)*(wz  )
        w101 = (wx  )*(1-wy)*(wz  )
        w011 = (1-wx)*(wy  )*(wz  )
        w111 = (wx  )*(wy  )*(wz  )

        # zero-out padded points
        def apply_mask(w):
            return w * mask_f

        # prep output grids
        vox = torch.zeros(B, self.out_feat, self.D, self.H, self.W, device=device)
        cnt = torch.zeros(B, 1,            self.D, self.H, self.W, device=device) if self.pooling == "mean" else None

        # linearize 3D indices: idx = z*H*W + y*W + x
        def lin_idx(xi, yi, zi):
            xi = xi.long(); yi = yi.long(); zi = zi.long()
            return (zi * (self.H * self.W) + yi * self.W + xi)  # [B,N]

        # scatter function using scatter_add on the flattened grid
        def scatter(w, xi, yi, zi):
            w = apply_mask(w).unsqueeze(-1)           # [B,N,1]
            contrib = (w * feat)                      # [B,N,Cg]
            idx = lin_idx(xi, yi, zi)                 # [B,N]

            # flatten grid per-batch for fast scatter_add
            vox_flat = vox.view(B, self.out_feat, -1)     # [B,Cg, D*H*W]
            if cnt is not None:
                cnt_flat = cnt.view(B, 1, -1)             # [B,1,  D*H*W]

            # For each batch independently
            for b in range(B):
                # repeat indices for all channels, then scatter_add
                # contrib[b]: [N,Cg] -> [Cg,N]
                vox_flat[b].scatter_add_(
                    dim=1,
                    index=idx[b].unsqueeze(0).expand(self.out_feat, -1),
                    src=contrib[b].transpose(0, 1)         # [Cg,N]
                )
                if cnt is not None:
                    cnt_flat[b].scatter_add_(
                        dim=1,
                        index=idx[b].unsqueeze(0),          # [1,N]
                        src=w[b].transpose(0, 1)            # [1,N]
                    )

        # eight corner splats
        scatter(w000, x0, y0, z0)
        scatter(w100, x1, y0, z0)
        scatter(w010, x0, y1, z0)
        scatter(w110, x1, y1, z0)
        scatter(w001, x0, y0, z1)
        scatter(w101, x1, y0, z1)
        scatter(w011, x0, y1, z1)
        scatter(w111, x1, y1, z1)

        if with_moments:
            den   = vox[:, 0:1]                    # [B,1,D,H,W]
            sum_x = vox[:, 1:2]; sum_y = vox[:, 2:3]; sum_z = vox[:, 3:4]
            sum_x2= vox[:, 4:5]; sum_y2= vox[:, 5:6]; sum_z2= vox[:, 6:7]
            eps = 1e-6
            mean_x = sum_x / (den + eps); mean_y = sum_y / (den + eps); mean_z = sum_z / (den + eps)
            var_x  = (sum_x2 / (den + eps) - mean_x**2).clamp_min(0.)
            var_y  = (sum_y2 / (den + eps) - mean_y**2).clamp_min(0.)
            var_z  = (sum_z2 / (den + eps) - mean_z**2).clamp_min(0.)
            vox = torch.cat([den, mean_x, mean_y, mean_z, var_x, var_y, var_z], dim=1)

        # mean pooling if requested
        if cnt is not None:
            vox = torch.where(cnt > 0, vox / (cnt + 1e-6), vox)

        return vox, {"grid_shape": (self.D, self.H, self.W)}
