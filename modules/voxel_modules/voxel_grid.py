# ---------- voxelizer (xyz API, dense [C,D,H,W]) ----------
import torch
from dataclasses import dataclass

@dataclass
class VoxelMetaXYZ:
    grid_whd: tuple            # (W,H,D) in xyz
    pmin: torch.Tensor         # [3] (x,y,z)
    pmax: torch.Tensor         # [3]
    voxel_size: torch.Tensor   # [3] (Δx,Δy,Δz)

class VoxelGridXYZ:
    """
    Public API in (x,y,z). Internal dense tensor is [C, D, H, W] (PyTorch layout).
    Modes: 'occupancy' | 'density' | 'avg_rgb'
    Provides fast voxel->point queries.
    """
    def __init__(self, points_xyz: torch.Tensor, colors: torch.Tensor=None,
                 grid_whd=(64,64,64), bounds=None, mode="density"):
        assert points_xyz.dim()==2 and points_xyz.size(-1)==3
        self.device = points_xyz.device
        self.dtype  = points_xyz.dtype
        self.W, self.H, self.D = grid_whd   # xyz

        # bounds (xyz)
        if bounds is None:
            pmin = points_xyz.amin(dim=0)
            pmax = points_xyz.amax(dim=0)
        else:
            pmin = torch.as_tensor(bounds[0], device=self.device, dtype=self.dtype)
            pmax = torch.as_tensor(bounds[1], device=self.device, dtype=self.dtype)

        span = (pmax - pmin).clamp_min(1e-6)
        self.meta = VoxelMetaXYZ(
            grid_whd=(self.W,self.H,self.D),
            pmin=pmin, pmax=pmax,
            voxel_size=torch.stack([span[0]/(self.W-1), span[1]/(self.H-1), span[2]/(self.D-1)])
        )

        # quantize xyz -> (ix,iy,iz)
        p01 = (points_xyz - pmin) / span
        ix = (p01[:,0]*(self.W-1)).round().clamp(0,self.W-1).long()
        iy = (p01[:,1]*(self.H-1)).round().clamp(0,self.H-1).long()
        iz = (p01[:,2]*(self.D-1)).round().clamp(0,self.D-1).long()
        lin = self._lin(ix, iy, iz)  # xyz-linear (x fastest)

        # CSR-like voxel→points
        order = torch.argsort(lin)
        self.sorted_lin  = lin[order]
        self.sorted_pidx = order
        uniq, counts = torch.unique_consecutive(self.sorted_lin, return_counts=True)
        self.occ_lin     = uniq
        self.occ_counts  = counts
        self.occ_offsets = torch.zeros_like(counts)
        self.occ_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        self._lin2occ = {int(l.item()): i for i,l in enumerate(self.occ_lin)}

        # dense grid [C,D,H,W]
        C = 1 if mode in ("occupancy","density") else 3
        self.grid = torch.zeros(C, self.D, self.H, self.W, device=self.device, dtype=self.dtype)
        if mode == "occupancy":
            self.grid[0, iz, iy, ix] = 1.0
        elif mode == "density":
            self.grid.index_put_((torch.zeros_like(iz), iz, iy, ix),
                                 torch.ones_like(iz, dtype=self.dtype), accumulate=True)
        elif mode == "moments":
            C = 7
            self.grid = torch.zeros(C, self.D, self.H, self.W, device=self.device, dtype=self.dtype)
            # accumulators
            one = torch.ones_like(iz, dtype=self.dtype)
            self.grid[0].index_put_((iz, iy, ix), one, accumulate=True)           # density (count)
            # sums for means
            self.grid[1].index_put_((iz, iy, ix), points_xyz[:,0], accumulate=True)  # sum x
            self.grid[2].index_put_((iz, iy, ix), points_xyz[:,1], accumulate=True)  # sum y
            self.grid[3].index_put_((iz, iy, ix), points_xyz[:,2], accumulate=True)  # sum z
            # sums for variances (E[x^2], etc.)
            self.grid[4].index_put_((iz, iy, ix), points_xyz[:,0]**2, accumulate=True)
            self.grid[5].index_put_((iz, iy, ix), points_xyz[:,1]**2, accumulate=True)
            self.grid[6].index_put_((iz, iy, ix), points_xyz[:,2]**2, accumulate=True)
            # finalize mean/var where density > 0
            den = self.grid[0].clamp_min(1e-6)
            self.grid[1:4] = self.grid[1:4] / den                # mean
            ex2 = self.grid[4:7] / den                           # E[x^2], etc.
            self.grid[4:7] = (ex2 - self.grid[1:4]**2).clamp_min(0.0)
        elif mode == "avg_rgb":
            if colors is None:
                raise ValueError("colors required for mode='avg_rgb'")
            acc = torch.zeros(1, self.D, self.H, self.W, device=self.device, dtype=self.dtype)
            for c in range(3):
                self.grid[c].index_put_((iz, iy, ix), colors[:,c], accumulate=True)
            acc.index_put_((iz, iy, ix), torch.ones_like(iz, dtype=self.dtype), accumulate=True)
            self.grid = torch.where(acc>0, self.grid/acc, self.grid)
        else:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode

    # ---------- xyz-linear helpers ----------
    def _lin(self, ix, iy, iz):
        return ix + self.W*(iy + self.H*iz)

    # ---------- queries (xyz) ----------
    def points_in_voxel(self, ix=None, iy=None, iz=None, lin=None):
        if lin is None:
            lin = int(ix + self.W*(iy + self.H*iz))
        slot = self._lin2occ.get(int(lin), None)
        if slot is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        start = self.occ_offsets[slot].item()
        cnt   = self.occ_counts[slot].item()
        return self.sorted_pidx[start:start+cnt]

    def to_dense(self):  # [C,D,H,W]
        return self.grid

    def meta_dict(self):
        m = self.meta
        return dict(W=self.W, H=self.H, D=self.D,
                    pmin=m.pmin, pmax=m.pmax, voxel_size=m.voxel_size)

# ---------- batch wrapper ----------
def build_voxel_batch(
    x_pts: torch.Tensor,              # [B,N,3] or [B,N,6] (xyz[,+rgb])
    grid_whd=(64,64,64),
    mode="density",
    bounds_mode="per_batch",          # 'per_batch' | 'global' | ((pmin),(pmax))
    valid_mask: torch.Tensor=None     # [B,N] bool, optional
):
    """
    Returns:
      dense: [B,C,D,H,W]
      vg_list: list[VoxelGridXYZ] length B
      metas: dict of stacked tensors for convenience
    """
    assert x_pts.dim()==3 and x_pts.size(-1) in (3,6), "x must be [B,N,3] or [B,N,6]"
    B, N, C_in = x_pts.shape
    has_rgb = (C_in == 6)
    device, dtype = x_pts.device, x_pts.dtype

    # bounds
    if bounds_mode == "global":
        p = x_pts[...,:3] if valid_mask is None else x_pts[...,:3][valid_mask]
        pmin_g = p.amin(dim=0) if p.dim()==2 else p.amin(dim=(0,1))
        pmax_g = p.amax(dim=0) if p.dim()==2 else p.amax(dim=(0,1))
        bounds = (pmin_g, pmax_g)
        bounds_list = [bounds]*B
    elif isinstance(bounds_mode, (tuple, list)):
        pmin, pmax = bounds_mode
        bounds_list = [(torch.as_tensor(pmin, device=device, dtype=dtype),
                        torch.as_tensor(pmax, device=device, dtype=dtype))]*B
    else:
        bounds_list = [None]*B  # per-batch fit

    vg_list = []
    dense_list = []
    metas = {"pmin": [], "pmax": [], "voxel_size": []}

    for b in range(B):
        mask = (valid_mask[b] if valid_mask is not None else torch.ones(N, device=device, dtype=torch.bool))
        pb = x_pts[b, mask, :3]
        cb = x_pts[b, mask, 3:6] if (has_rgb and mode=="avg_rgb") else None
        vg = VoxelGridXYZ(pb, cb, grid_whd=grid_whd, bounds=bounds_list[b], mode=mode)
        vg_list.append(vg)
        dense_list.append(vg.to_dense().unsqueeze(0))  # [1,C,D,H,W]
        md = vg.meta
        metas["pmin"].append(md.pmin)
        metas["pmax"].append(md.pmax)
        metas["voxel_size"].append(md.voxel_size)

    dense = torch.cat(dense_list, dim=0)  # [B,C,D,H,W]
    metas = {k: torch.stack(v, dim=0) for k,v in metas.items()}
    metas.update({"W": grid_whd[0], "H": grid_whd[1], "D": grid_whd[2]})
    return dense, vg_list, metas
