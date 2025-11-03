class VoxelGridXYZ:
    """
    Voxelization wrapper supporting multiple basic representations:

    modes:
      - 'occupancy'  : 1 ch, {0,1}
      - 'density'    : 1 ch, counts per voxel
      - 'avg_rgb'    : 3 ch, per-voxel mean RGB (needs colors)
      - 'moments'    : 7 ch, [count, mean_x, mean_y, mean_z, var_x, var_y, var_z]  (legacy; world coords)
      - 'moments_rel': 10 ch, [count, μx_rel, μy_rel, μz_rel, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz]
                       (means/cov in *voxel units*, relative to voxel centers)
      - 'moments_peak': 11 ch, same as moments_rel + surface_peak = exp(-λ_min / σ²)

    Notes:
      * "voxel units" mean coordinates normalized by voxel size, so a voxel spans ~[-0.5,0.5] per axis around center.
      * For moments_rel / moments_peak we compute biased covariance: E[xx]-μx^2, etc.
    """
    def __init__(self, points_xyz: torch.Tensor, colors: torch.Tensor = None,
                 grid_whd=(64, 64, 64), bounds=None, mode="density",
                 peak_sigma_vox: float = 0.15):
        assert points_xyz.dim() == 2 and points_xyz.size(-1) == 3, "points must be [N,3]"
        self.device, self.dtype = points_xyz.device, points_xyz.dtype
        self.W, self.H, self.D = map(int, grid_whd)

        # bounds
        if bounds is None:
            pmin = points_xyz.amin(dim=0)
            pmax = points_xyz.amax(dim=0)
        else:
            pmin = torch.as_tensor(bounds[0], device=self.device, dtype=self.dtype)
            pmax = torch.as_tensor(bounds[1], device=self.device, dtype=self.dtype)

        span = (pmax - pmin).clamp_min(1e-6)
        self.meta = VoxelMetaXYZ(
            grid_whd=(self.W, self.H, self.D),
            pmin=pmin, pmax=pmax,
            voxel_size=torch.stack([span[0] / (self.W - 1),
                                    span[1] / (self.H - 1),
                                    span[2] / (self.D - 1)])
        )
        vx = self.meta.voxel_size  # [3]

        # bin points to voxel indices
        p01 = (points_xyz - pmin) / span
        ix = (p01[:, 0] * (self.W - 1)).floor().clamp(0, self.W - 1).long()
        iy = (p01[:, 1] * (self.H - 1)).floor().clamp(0, self.H - 1).long()
        iz = (p01[:, 2] * (self.D - 1)).floor().clamp(0, self.D - 1).long()

        # bookkeeping for points_in_voxel
        lin = self._lin(ix, iy, iz)
        order = torch.argsort(lin)
        self.sorted_lin = lin[order]
        self.sorted_pidx = order
        uniq, counts = torch.unique_consecutive(self.sorted_lin, return_counts=True)
        self.occ_lin = uniq
        self.occ_counts = counts
        self.occ_offsets = torch.zeros_like(counts)
        self.occ_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        self._lin2occ = {int(l.item()): i for i, l in enumerate(self.occ_lin)}

        # allocate grid by mode
        self.mode = mode
        if mode in ("occupancy", "density"):
            C = 1
        elif mode == "moments":
            C = 7
        elif mode == "avg_rgb":
            if colors is None:
                raise ValueError("colors required for mode='avg_rgb'")
            assert colors.shape[0] == points_xyz.shape[0] and colors.shape[1] == 3
            C = 3
        elif mode == "moments_rel":
            C = 10
        elif mode == "moments_peak":
            C = 11
        else:
            raise ValueError(f"unknown mode '{mode}'")
        self.grid = torch.zeros(C, self.D, self.H, self.W, device=self.device, dtype=self.dtype)

        # fill by mode
        if mode == "occupancy":
            self.grid[0, iz, iy, ix] = 1.0

        elif mode == "density":
            self.grid.index_put_((torch.zeros_like(iz), iz, iy, ix),
                                 torch.ones_like(iz, dtype=self.dtype),
                                 accumulate=True)

        elif mode == "avg_rgb":
            acc = torch.zeros(1, self.D, self.H, self.W, device=self.device, dtype=self.dtype)
            for c in range(3):
                self.grid[c].index_put_((iz, iy, ix), colors[:, c].to(self.dtype), accumulate=True)
            acc.index_put_((iz, iy, ix), torch.ones_like(iz, dtype=self.dtype), accumulate=True)
            self.grid = torch.where(acc > 0, self.grid / acc, self.grid)

        elif mode == "moments":
            # legacy: world-mean & diagonal var
            one = torch.ones_like(iz, dtype=self.dtype)
            self.grid[0].index_put_((iz, iy, ix), one, accumulate=True)                 # count
            self.grid[1].index_put_((iz, iy, ix), points_xyz[:, 0], accumulate=True)    # sum x
            self.grid[2].index_put_((iz, iy, ix), points_xyz[:, 1], accumulate=True)    # sum y
            self.grid[3].index_put_((iz, iy, ix), points_xyz[:, 2], accumulate=True)    # sum z
            self.grid[4].index_put_((iz, iy, ix), points_xyz[:, 0] ** 2, accumulate=True)
            self.grid[5].index_put_((iz, iy, ix), points_xyz[:, 1] ** 2, accumulate=True)
            self.grid[6].index_put_((iz, iy, ix), points_xyz[:, 2] ** 2, accumulate=True)
            den = self.grid[0].clamp_min(1e-6)
            mean = self.grid[1:4] / den
            ex2 = self.grid[4:7] / den
            var = (ex2 - mean ** 2).clamp_min(0.0)
            self.grid[1:4] = mean
            self.grid[4:7] = var

        elif mode in ("moments_rel", "moments_peak"):
            # Per-point offsets relative to voxel centers, in voxel units
            # voxel center in world coords for each point's (ix,iy,iz)
            cx = pmin[0] + ix.to(self.dtype) * vx[0]
            cy = pmin[1] + iy.to(self.dtype) * vx[1]
            cz = pmin[2] + iz.to(self.dtype) * vx[2]
            dx = (points_xyz[:, 0] - cx) / vx[0]
            dy = (points_xyz[:, 1] - cy) / vx[1]
            dz = (points_xyz[:, 2] - cz) / vx[2]
            # (optionally clamp to [-0.5,0.5] to bound outliers)
            # dx = dx.clamp(-0.5, 0.5); dy = dy.clamp(-0.5, 0.5); dz = dz.clamp(-0.5, 0.5)

            # allocate accumulators
            # [count, sumx, sumy, sumz, sumxx, sumyy, sumzz, sumxy, sumxz, sumyz]
            acc = torch.zeros(10, self.D, self.H, self.W, device=self.device, dtype=self.dtype)

            one = torch.ones_like(dx, dtype=self.dtype)
            acc[0].index_put_((iz, iy, ix), one, accumulate=True)       # count
            acc[1].index_put_((iz, iy, ix), dx, accumulate=True)
            acc[2].index_put_((iz, iy, ix), dy, accumulate=True)
            acc[3].index_put_((iz, iy, ix), dz, accumulate=True)
            acc[4].index_put_((iz, iy, ix), dx * dx, accumulate=True)
            acc[5].index_put_((iz, iy, ix), dy * dy, accumulate=True)
            acc[6].index_put_((iz, iy, ix), dz * dz, accumulate=True)
            acc[7].index_put_((iz, iy, ix), dx * dy, accumulate=True)
            acc[8].index_put_((iz, iy, ix), dx * dz, accumulate=True)
            acc[9].index_put_((iz, iy, ix), dy * dz, accumulate=True)

            n = acc[0].clamp_min(1e-6)
            mux = acc[1] / n
            muy = acc[2] / n
            muz = acc[3] / n

            cxx = (acc[4] / n - mux * mux).clamp_min(0.0)
            cyy = (acc[5] / n - muy * muy).clamp_min(0.0)
            czz = (acc[6] / n - muz * muz).clamp_min(0.0)
            cxy = (acc[7] / n - mux * muy)
            cxz = (acc[8] / n - mux * muz)
            cyz = (acc[9] / n - muy * muz)

            # pack into grid
            self.grid[0] = acc[0]                  # count
            self.grid[1] = mux
            self.grid[2] = muy
            self.grid[3] = muz
            self.grid[4] = cxx
            self.grid[5] = cyy
            self.grid[6] = czz
            self.grid[7] = cxy
            self.grid[8] = cxz
            self.grid[9] = cyz

            if mode == "moments_peak":
                # build 3x3 symmetric covariance per voxel and compute λ_min
                D, H, W = self.D, self.H, self.W
                cov = torch.zeros(D * H * W, 3, 3, device=self.device, dtype=self.dtype)
                cxx_f = cxx.reshape(-1); cyy_f = cyy.reshape(-1); czz_f = czz.reshape(-1)
                cxy_f = cxy.reshape(-1); cxz_f = cxz.reshape(-1); cyz_f = cyz.reshape(-1)
                cov[:, 0, 0] = cxx_f
                cov[:, 1, 1] = cyy_f
                cov[:, 2, 2] = czz_f
                cov[:, 0, 1] = cov[:, 1, 0] = cxy_f
                cov[:, 0, 2] = cov[:, 2, 0] = cxz_f
                cov[:, 1, 2] = cov[:, 2, 1] = cyz_f

                # mask: trust only voxels with >= 2 points
                valid = (acc[0].reshape(-1) >= 2)
                lam_min = torch.zeros(D * H * W, device=self.device, dtype=self.dtype)
                if valid.any():
                    # eigvalsh: symmetric eigvals ascending (λ1<=λ2<=λ3); we want λ1 (smallest)
                    lam = torch.linalg.eigvalsh(cov[valid])  # [Nv,3]
                    lam_min[valid] = lam[:, 0].clamp_min(0.0)

                lam_min = lam_min.reshape(D, H, W)
                sigma2 = torch.as_tensor(peak_sigma_vox ** 2, dtype=self.dtype, device=self.device)
                surface_peak = torch.exp(-lam_min / sigma2)
                # Zero-out where invalid
                surface_peak = torch.where(acc[0] >= 2, surface_peak, torch.zeros_like(surface_peak))
                self.grid[10] = surface_peak

    def _lin(self, ix, iy, iz):
        return ix + self.W * (iy + self.H * iz)

    def points_in_voxel(self, ix=None, iy=None, iz=None, lin=None):
        if lin is None:
            lin = int(ix + self.W * (iy + self.H * iz))
        slot = self._lin2occ.get(int(lin), None)
        if slot is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        start = self.occ_offsets[slot].item()
        cnt = self.occ_counts[slot].item()
        return self.sorted_pidx[start:start + cnt]

    def to_dense(self):
        return self.grid

    def meta_dict(self):
        m = self.meta
        return dict(W=self.W, H=self.H, D=self.D, pmin=m.pmin, pmax=m.pmax, voxel_size=m.voxel_size)
