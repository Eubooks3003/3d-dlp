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

    @staticmethod
    def _pad_to_multiple(x, mult):
        """Pad [B,C,D,H,W] so D,H,W are multiples of mult=(md,mh,mw)."""
        B,C,D,H,W = x.shape
        md,mh,mw = mult
        padD = (md - D % md) % md
        padH = (mh - H % mh) % mh
        padW = (mw - W % mw) % mw
        if padD or padH or padW:
            # pad = (W_l, W_r, H_l, H_r, D_l, D_r)
            x = F.pad(x, (0,padW, 0,padH, 0,padD))
        return x, (padD, padH, padW)

    def get_tile_location_idx(self, Dp, Hp, Wp, tile_size, device=None, dtype=torch.int32):
        """Return starts (z,y,x) for each tile in the **padded** grid."""
        td,th,tw = map(int, tile_size)
        Td,Th,Tw = Dp//td, Hp//th, Wp//tw
        zz = torch.arange(Td, device=device, dtype=dtype) * td
        yy = torch.arange(Th, device=device, dtype=dtype) * th
        xx = torch.arange(Tw, device=device, dtype=dtype) * tw
        Z,Y,X = torch.meshgrid(zz, yy, xx, indexing="ij")
        starts = torch.stack([Z, Y, X], dim=-1).reshape(-1, 3)   # [T,3]
        return starts

    def get_tile_centers(self, Dp, Hp, Wp, tile_size, device=None, dtype=torch.float32):
        """Centers (z,y,x) of every tile in voxel coordinates (padded grid)."""
        starts = self.get_tile_location_idx(Dp, Hp, Wp, tile_size, device=device, dtype=torch.int32)
        td,th,tw = map(float, tile_size)
        mid = torch.tensor([td/2, th/2, tw/2], device=starts.device, dtype=torch.float32)
        return starts.to(torch.float32) + mid  # [T,3]

    def vox_to_tiles(self, vox, tile_size):
        """
        Split [B,C,D,H,W] -> tiles [B,C,T,td,th,tw], returns tiles + meta (pads, shapes, grid).
        """
        B,C,D,H,W = vox.shape
        td,th,tw = map(int, tile_size)
        vox_padded, pads = self._pad_to_multiple(vox, (td,th,tw))
        _,_,Dp,Hp,Wp = vox_padded.shape
        Td,Th,Tw = Dp//td, Hp//th, Wp//tw

        # [B,C,Td,td,Th,th,Tw,tw] -> [B,C,Td,Th,Tw,td,th,tw] -> [B,C,T,td,th,tw]
        g = vox_padded.view(B, C, Td, td, Th, th, Tw, tw).permute(0,1,2,4,6,3,5,7).contiguous()
        tiles = g.view(B, C, Td*Th*Tw, td, th, tw)

        meta = {
            "pads": (pads[0], pads[1], pads[2]),      # (padD,padH,padW)
            "padded_shape": (Dp,Hp,Wp),               # after padding
            "tiles_grid": (Td,Th,Tw),                 # number of tiles per axis
            "tile_size": (td,th,tw),                  # tile extent
        }
        return tiles, meta

    def tiles_to_vox(self, tiles, meta):
        """
        Inverse of vox_to_tiles for debugging/visualization.
        tiles: [B,C,T,td,th,tw] -> [B,C,Dp,Hp,Wp] (still padded)
        """
        B,C,T,td,th,tw = tiles.shape
        Td,Th,Tw = meta["tiles_grid"]
        Dp,Hp,Wp = meta["padded_shape"]
        g = tiles.view(B, C, Td, Th, Tw, td, th, tw).permute(0,1,2,5,3,6,4,7).contiguous()
        return g.view(B, C, Dp, Hp, Wp)

    def forward(self, pts: torch.Tensor, mask: torch.Tensor = None, weights=None, with_moments=True):
        """
        pts: [B,N,3(+F)] with xyz in [-1,1]^3
        returns:
        vox:  [B,Cg,D,H,W] where Cg = 7 if with_moments else self.out_feat
        meta: {"grid_shape": (D,H,W)}
        """
        B, N, C = pts.shape
        device = pts.device

        # mask & weights
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask_f = mask.float()
        if weights is None:
            weights = torch.ones(B, N, device=device)
        weights = weights * mask_f

        # coords/features
        xyz = pts[..., :3]  # [-1,1]
        if with_moments:
            # features to splat: [den, w*x, w*y, w*z, w*x^2, w*y^2, w*z^2]
            f_den  = weights[:, :, None]                      # [B,N,1]
            f_xyz  = weights[:, :, None] * xyz                # [B,N,3]
            f_xyz2 = weights[:, :, None] * (xyz * xyz)        # [B,N,3]
            feat = torch.cat([f_den, f_xyz, f_xyz2], dim=-1)  # [B,N,7]
            Cg = 7
        else:
            if C > 3 and (C - 3) == self.out_feat:
                feat = pts[..., 3:]                           # [B,N,out_feat]
            else:
                feat = torch.ones(B, N, self.out_feat, device=device)
            Cg = feat.shape[-1]

        # map to voxel index space [0..size-1]
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

        def apply_mask(w): return w * mask_f
        def lin_idx(xi, yi, zi):
            xi = xi.long(); yi = yi.long(); zi = zi.long()
            return (zi * (self.H * self.W) + yi * self.W + xi)

        # allocate with Cg channels (IMPORTANT)
        vox = torch.zeros(B, Cg, self.D, self.H, self.W, device=device)
        cnt = torch.zeros(B, 1, self.D, self.H, self.W, device=device) if self.pooling == "mean" else None

        def scatter(w, xi, yi, zi):
            w = apply_mask(w).unsqueeze(-1)    # [B,N,1]
            contrib = (w * feat)               # [B,N,Cg]
            idx = lin_idx(xi, yi, zi)          # [B,N]
            vox_flat = vox.view(B, Cg, -1)
            if cnt is not None:
                cnt_flat = cnt.view(B, 1, -1)
            for b in range(B):
                vox_flat[b].scatter_add_(dim=1,
                                        index=idx[b].unsqueeze(0).expand(Cg, -1),
                                        src=contrib[b].transpose(0, 1))
                if cnt is not None:
                    cnt_flat[b].scatter_add_(dim=1,
                                            index=idx[b].unsqueeze(0),
                                            src=w[b].transpose(0, 1))

        # eight corner splats
        scatter(w000, x0, y0, z0); scatter(w100, x1, y0, z0)
        scatter(w010, x0, y1, z0); scatter(w110, x1, y1, z0)
        scatter(w001, x0, y0, z1); scatter(w101, x1, y0, z1)
        scatter(w011, x0, y1, z1); scatter(w111, x1, y1, z1)

        if with_moments:
            den   = vox[:, 0:1]
            sum_x = vox[:, 1:2]; sum_y = vox[:, 2:3]; sum_z = vox[:, 3:4]
            sum_x2= vox[:, 4:5]; sum_y2= vox[:, 5:6]; sum_z2= vox[:, 6:7]
            eps = 1e-6
            mean_x = sum_x / (den + eps)
            mean_y = sum_y / (den + eps)
            mean_z = sum_z / (den + eps)
            var_x  = (sum_x2 / (den + eps) - mean_x**2).clamp_min(0.)
            var_y  = (sum_y2 / (den + eps) - mean_y**2).clamp_min(0.)
            var_z  = (sum_z2 / (den + eps) - mean_z**2).clamp_min(0.)
            vox = torch.cat([den, mean_x, mean_y, mean_z, var_x, var_y, var_z], dim=1)  # [B,7,D,H,W]

        if cnt is not None:
            vox = torch.where(cnt > 0, vox / (cnt + 1e-6), vox)

        return vox, {"grid_shape": (self.D, self.H, self.W)}
