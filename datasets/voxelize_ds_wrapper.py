# voxelized_dataset.py
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class VoxelMetaXYZ:
    grid_whd: tuple
    pmin: torch.Tensor
    pmax: torch.Tensor
    voxel_size: torch.Tensor


class VoxelGridXYZ:
    def __init__(self, points_xyz: torch.Tensor, colors: torch.Tensor = None,
                 grid_whd=(64, 64, 64), bounds=None, mode="density"):
        assert points_xyz.dim() == 2 and points_xyz.size(-1) == 3, "points must be [N,3]"
        self.device, self.dtype = points_xyz.device, points_xyz.dtype
        self.W, self.H, self.D = map(int, grid_whd)

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
            voxel_size=torch.stack([span[0] / (self.W - 1), span[1] / (self.H - 1), span[2] / (self.D - 1)])
        )

        p01 = (points_xyz - pmin) / span
        ix = (p01[:, 0] * (self.W - 1)).floor().clamp(0, self.W - 1).long()
        iy = (p01[:, 1] * (self.H - 1)).floor().clamp(0, self.H - 1).long()
        iz = (p01[:, 2] * (self.D - 1)).floor().clamp(0, self.D - 1).long()

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

        if mode in ("occupancy", "density"):
            C = 1
        elif mode == "moments":
            C = 7
        elif mode == "avg_rgb":
            C = 3
            if colors is None:
                raise ValueError("colors required for mode='avg_rgb'")
            assert colors.shape[0] == points_xyz.shape[0] and colors.shape[1] == 3
        else:
            raise ValueError(f"unknown mode '{mode}'")
        self.grid = torch.zeros(C, self.D, self.H, self.W, device=self.device, dtype=self.dtype)

        if mode == "occupancy":
            self.grid[0, iz, iy, ix] = 1.0
        elif mode == "density":
            self.grid.index_put_(
                (torch.zeros_like(iz), iz, iy, ix),
                torch.ones_like(iz, dtype=self.dtype),
                accumulate=True
            )
        elif mode == "moments":
            one = torch.ones_like(iz, dtype=self.dtype)
            self.grid[0].index_put_((iz, iy, ix), one, accumulate=True)
            self.grid[1].index_put_((iz, iy, ix), points_xyz[:, 0], accumulate=True)
            self.grid[2].index_put_((iz, iy, ix), points_xyz[:, 1], accumulate=True)
            self.grid[3].index_put_((iz, iy, ix), points_xyz[:, 2], accumulate=True)
            self.grid[4].index_put_((iz, iy, ix), points_xyz[:, 0] ** 2, accumulate=True)
            self.grid[5].index_put_((iz, iy, ix), points_xyz[:, 1] ** 2, accumulate=True)
            self.grid[6].index_put_((iz, iy, ix), points_xyz[:, 2] ** 2, accumulate=True)
            den = self.grid[0].clamp_min(1e-6)
            mean = self.grid[1:4] / den
            ex2 = self.grid[4:7] / den
            var = (ex2 - mean ** 2).clamp_min(0.0)
            self.grid[1:4] = mean
            self.grid[4:7] = var
        elif mode == "avg_rgb":
            acc = torch.zeros(1, self.D, self.H, self.W, device=self.device, dtype=self.dtype)
            for c in range(3):
                self.grid[c].index_put_((iz, iy, ix), colors[:, c].to(self.dtype), accumulate=True)
            acc.index_put_((iz, iy, ix), torch.ones_like(iz, dtype=self.dtype), accumulate=True)
            self.grid = torch.where(acc > 0, self.grid / acc, self.grid)

        self.mode = mode

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


class VoxelizedDataset(Dataset):
    """
    Precompute voxel grids for every item in base_ds (with caching).
    If `cache_dir` is provided:
      - If a compatible manifest exists, load precomputed voxels/metas from disk.
      - Otherwise, build once, save per-item files, and load next runs.

    Each __getitem__ returns:
      {
        "voxels": [C,D,H,W]  (on `device`)
        "meta":   {pmin,pmax,voxel_size,W,H,D} (tensors on `device`)
        ... passthrough fields from base_ds ...
      }
    """
    def __init__(self,
                 base_ds,
                 grid_whd: Tuple[int, int, int] = (64, 64, 64),
                 mode: str = "density",
                 bounds_mode: Union[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = "per_item",
                 keep_points: bool = False,
                 device: Optional[torch.device] = None,
                 cache_dir: Optional[str] = None,
                 cache_extras: bool = False,   # store extra fields besides points (may be large / non-serializable)
                 force_rebuild: bool = False):

        self.base_ds = base_ds
        self.grid_whd = tuple(map(int, grid_whd))
        self.mode = mode
        self.keep_points = keep_points
        self.device = device or torch.device("cpu")
        self.cache_dir = cache_dir
        self.cache_extras = cache_extras

        # build or load bounds
        self.bounds = None
        bmode_str = "per_item"
        if bounds_mode == "global":
            bmode_str = "global"
            pmin_g, pmax_g = self._compute_global_bounds()
            self.bounds = (pmin_g, pmax_g)
        elif isinstance(bounds_mode, (tuple, list)):
            bmode_str = "fixed"
            pmin, pmax = bounds_mode
            self.bounds = (
                torch.as_tensor(pmin, device=self.device, dtype=torch.float32),
                torch.as_tensor(pmax, device=self.device, dtype=torch.float32)
            )

        # Try cache path
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            manifest_path = os.path.join(self.cache_dir, "manifest.json")

            # Decide whether to load or rebuild
            if (not force_rebuild) and os.path.isfile(manifest_path):
                with open(manifest_path, "r") as f:
                    mani = json.load(f)

                cache_ok = (
                    mani.get("length") == len(base_ds) and
                    tuple(mani.get("grid_whd")) == self.grid_whd and
                    mani.get("mode") == self.mode and
                    mani.get("bounds_mode") == bmode_str
                )

                # verify bounds if present
                if cache_ok and bmode_str in ("global", "fixed"):
                    pm = torch.tensor(mani["pmin"])
                    px = torch.tensor(mani["pmax"])
                    if self.bounds is None:
                        cache_ok = False
                    else:
                        cache_ok = cache_ok and torch.allclose(self.bounds[0].cpu(), pm) and torch.allclose(self.bounds[1].cpu(), px)

                if cache_ok and self._all_item_files_exist():
                    # Load indices only; tensors will be lazy-loaded on demand in __getitem__ for memory friendliness
                    self._use_cache = True
                    self._manifest = mani
                else:
                    self._use_cache = False
            else:
                self._use_cache = False
        else:
            self._use_cache = False

        # If no cache (or rebuild), precompute and save
        if not self._use_cache:
            self._build_and_optionally_cache(bmode_str)

    # ---------- cache helpers ----------
    def _item_paths(self, idx: int):
        base = os.path.join(self.cache_dir, f"{idx:06d}") if self.cache_dir else None
        if base is None:
            return None, None, None
        return base + "_voxels.pt", base + "_meta.pt", base + "_extras.pt"

    def _all_item_files_exist(self):
        for i in range(len(self.base_ds)):
            v_p, m_p, _ = self._item_paths(i)
            if not (os.path.isfile(v_p) and os.path.isfile(m_p)):
                return False
        return True

    def _write_manifest(self, bmode_str: str):
        if self.cache_dir is None:
            return
        mani = {
            "length": len(self.base_ds),
            "grid_whd": list(self.grid_whd),
            "mode": self.mode,
            "bounds_mode": bmode_str,
        }
        if self.bounds is not None:
            mani["pmin"] = self.bounds[0].cpu().tolist()
            mani["pmax"] = self.bounds[1].cpu().tolist()

        with open(os.path.join(self.cache_dir, "manifest.json"), "w") as f:
            json.dump(mani, f, indent=2)

    def _compute_global_bounds(self):
        pmins, pmaxs = [], []
        for i in tqdm(range(len(self.base_ds)), desc="Scanning global bounds", leave=False):
            item = self.base_ds[i]
            pts = torch.as_tensor(item["points"], device=self.device, dtype=torch.float32)[..., :3]
            pmins.append(pts.amin(dim=0))
            pmaxs.append(pts.amax(dim=0))
        pmin_g = torch.stack(pmins, 0).amin(dim=0)
        pmax_g = torch.stack(pmaxs, 0).amax(dim=0)
        return pmin_g, pmax_g

    def _build_and_optionally_cache(self, bmode_str: str):
        # eager precompute (in-memory) + optional write to disk
        self._voxels = []
        self._metas = []
        self._pass = []

        if self.cache_dir:
            self._write_manifest(bmode_str)

        for i in tqdm(range(len(self.base_ds)), desc="Voxelizing dataset"):
            item = self.base_ds[i]
            pts_all = torch.as_tensor(item["points"], device=self.device, dtype=torch.float32)
            pts_xyz = pts_all[:, :3]
            colors = pts_all[:, 3:6] if (pts_all.shape[-1] == 6 and self.mode == "avg_rgb") else None

            vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=self.grid_whd, bounds=self.bounds, mode=self.mode)
            vox = vg.to_dense().to(self.device)  # [C,D,H,W]
            md = vg.meta_dict()
            md.update({"W": self.grid_whd[0], "H": self.grid_whd[1], "D": self.grid_whd[2]})
            md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}

            extra = {k: v for k, v in item.items() if k != "points"}
            if self.keep_points:
                extra["points"] = pts_all
            # Try to keep extras minimal/non-serialized by default
            extra_to_save = (extra if self.cache_extras else None)

            # save if requested
            if self.cache_dir is not None:
                vox_p, meta_p, extras_p = self._item_paths(i)
                torch.save(vox.cpu(), vox_p)
                torch.save(md_cpu, meta_p)
                if extra_to_save is not None:
                    try:
                        torch.save(extra_to_save, extras_p)
                    except Exception:
                        # if it fails (non-serializable), just skip caching extras
                        pass

            # keep in memory for this run
            self._voxels.append(vox)
            self._metas.append(md)
            self._pass.append(extra)

        # mark cache usable for future runs
        if self.cache_dir is not None:
            self._use_cache = True
            # lightweight manifest is already written

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        if self.cache_dir is not None and self._use_cache and not hasattr(self, "_voxels"):
            # lazy load per-item from disk (to keep memory small)
            vox_p, meta_p, extras_p = self._item_paths(idx)
            vox = torch.load(vox_p, map_location=self.device)
            md = torch.load(meta_p, map_location=self.device)
            # ensure tensors on device
            for k in ("pmin", "pmax", "voxel_size"):
                if isinstance(md[k], torch.Tensor):
                    md[k] = md[k].to(self.device)
            # passthrough: prefer base_ds live fields (they might be dynamic)
            extra = {k: v for k, v in self.base_ds[idx].items() if k != "points"}
            if self.keep_points:
                extra["points"] = torch.as_tensor(self.base_ds[idx]["points"], device=self.device, dtype=torch.float32)
            else:
                # optionally fallback to cached extras
                if self.cache_extras and os.path.isfile(extras_p):
                    try:
                        cached_extras = torch.load(extras_p, map_location=self.device)
                        extra.update(cached_extras)
                    except Exception:
                        pass
            out = {"voxels": vox, "meta": md}
            out.update(extra)
            return out

        # in-memory path (same run we built):
        if hasattr(self, "_voxels"):
            out = {"voxels": self._voxels[idx], "meta": self._metas[idx]}
            extra = self._pass[idx] if hasattr(self, "_pass") else {}
            out.update(extra)
            return out

        # fallback (shouldn’t happen): compute on the fly
        item = self.base_ds[idx]
        pts_all = torch.as_tensor(item["points"], device=self.device, dtype=torch.float32)
        pts_xyz = pts_all[:, :3]
        colors = pts_all[:, 3:6] if (pts_all.shape[-1] == 6 and self.mode == "avg_rgb") else None
        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=self.grid_whd, bounds=self.bounds, mode=self.mode)
        out = {"voxels": vg.to_dense().to(self.device), "meta": vg.meta_dict()}
        out.update({k: v for k, v in item.items() if k != "points"})
        if self.keep_points:
            out["points"] = pts_all
        return out