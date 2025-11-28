# voxelized_dataset_fixed.py
import json, os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ----------------- Meta container -----------------

@dataclass
class VoxelMetaXYZ:
    grid_whd: tuple
    pmin: torch.Tensor
    pmax: torch.Tensor
    voxel_size: torch.Tensor


# ----------------- Voxel grid -----------------

class VoxelGridXYZ:
    """
    Points -> voxel grid in XYZ, modes:
      - "occupancy": binary occupancy
      - "density":   point count per voxel
      - "moments":   [count, mean(xyz), var(xyz)] -> 7 channels
      - "avg_rgb":   mean RGB per voxel (requires colors [N,3])

    Output grid shape: [C, D, H, W]
    """

    def __init__(
        self,
        points_xyz: torch.Tensor,       # [N,3]
        colors: Optional[torch.Tensor] = None,   # [N,3] or None
        grid_whd: Tuple[int, int, int] = (64, 64, 64),
        bounds=None,
        mode: str = "density",
    ):
        assert points_xyz.dim() == 2 and points_xyz.size(-1) == 3, \
            f"points_xyz must be [N,3], got {points_xyz.shape}"

        self.device, self.dtype = points_xyz.device, points_xyz.dtype
        self.W, self.H, self.D = map(int, grid_whd)
        self.mode = str(mode)

        # ----- bounds -----
        if bounds is None:
            pmin = points_xyz.amin(dim=0)
            pmax = points_xyz.amax(dim=0)
        else:
            pmin = torch.as_tensor(bounds[0], device=self.device, dtype=self.dtype)
            pmax = torch.as_tensor(bounds[1], device=self.device, dtype=self.dtype)

        span = (pmax - pmin).clamp_min(1e-6)

        voxel_size = torch.stack([
            span[0] / (self.W - 1),
            span[1] / (self.H - 1),
            span[2] / (self.D - 1),
        ])

        self.meta = VoxelMetaXYZ(
            grid_whd=(self.W, self.H, self.D),
            pmin=pmin,
            pmax=pmax,
            voxel_size=voxel_size,
        )

        # ----- map points -> [0,1] and voxel indices -----
        p01 = (points_xyz - pmin) / span
        p01 = p01.clamp(0.0, 1.0 - 1e-6)

        ix = (p01[:, 0] * (self.W - 1)).floor().long()
        iy = (p01[:, 1] * (self.H - 1)).floor().long()
        iz = (p01[:, 2] * (self.D - 1)).floor().long()

        # ----- channel count -----
        if self.mode in ("occupancy", "density"):
            C = 1
        elif self.mode == "moments":
            C = 7
        elif self.mode == "avg_rgb":
            C = 3
            if colors is None:
                raise ValueError("VoxelGridXYZ(mode='avg_rgb') requires colors [N,3]")
            assert colors.shape[0] == points_xyz.shape[0] and colors.shape[1] == 3, \
                f"colors must be [N,3], got {colors.shape}"
        else:
            raise ValueError(f"Unknown voxelization mode '{self.mode}'")

        # grid: [C, D, H, W]
        self.grid = torch.zeros(C, self.D, self.H, self.W,
                                device=self.device, dtype=self.dtype)

        # ----- modes -----

        if self.mode == "occupancy":
            # binary occupancy
            self.grid[0, iz, iy, ix] = 1.0

        elif self.mode == "density":
            # count points per voxel
            ones = torch.ones_like(iz, dtype=self.dtype)
            # operate on channel 0 only
            self.grid[0].index_put_((iz, iy, ix), ones, accumulate=True)

        elif self.mode == "moments":
            # channels:
            #   0: count
            #   1–3: mean x,y,z
            #   4–6: var  x,y,z
            one = torch.ones_like(iz, dtype=self.dtype)

            self.grid[0].index_put_((iz, iy, ix), one,              accumulate=True)
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

        elif self.mode == "avg_rgb":
            # accumulate sum of RGB per voxel + count
            colors = colors.to(self.dtype)
            acc = torch.zeros(self.D, self.H, self.W,
                              device=self.device, dtype=self.dtype)

            for c in range(3):
                self.grid[c].index_put_(
                    (iz, iy, ix),
                    colors[:, c],
                    accumulate=True,
                )

            acc.index_put_(
                (iz, iy, ix),
                torch.ones_like(iz, dtype=self.dtype),
                accumulate=True,
            )

            # avoid divide-by-zero; broadcast acc to [1,D,H,W] then to [3,D,H,W]
            acc_clamped = acc.clamp_min(1e-6)
            self.grid = self.grid / acc_clamped.unsqueeze(0)

    def to_dense(self) -> torch.Tensor:
        return self.grid

    def meta_dict(self) -> Dict[str, Any]:
        m = self.meta
        return dict(
            W=self.W, H=self.H, D=self.D,
            pmin=m.pmin,
            pmax=m.pmax,
            voxel_size=m.voxel_size,
        )


# ----------------- Voxelized dataset -----------------

class VoxelizedDataset(Dataset):
    """
    Wraps a base dataset that yields {"points": [N,C], ...}. When cache_dir is given,
    it can precompute voxels once and later read them lazily.

    Each sample:
      {
        "voxels": [C,D,H,W],
        "meta":   {...},
        "points": [N,C]          (if keep_points=True),
        ... passthrough from base_ds (id, path, mask, ...)
      }
    """

    def __init__(
        self,
        base_ds,
        grid_whd: Tuple[int, int, int] = (64, 64, 64),
        mode: str = "density",
        bounds_mode: Union[str, Tuple[Tuple[float, float, float],
                                      Tuple[float, float, float]]] = "per_item",
        keep_points: bool = True,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
        cache_extras: bool = False,
        force_rebuild: bool = False,
    ):
        self.base_ds = base_ds
        self.grid_whd = tuple(map(int, grid_whd))
        self.mode = str(mode)
        self.keep_points = bool(keep_points)
        self.device = device or torch.device("cpu")
        self.cache_dir = cache_dir
        self.cache_extras = bool(cache_extras)

        # manifest if using cache
        mani_path = os.path.join(self.cache_dir, "manifest.json") if self.cache_dir else None
        mani = None
        if mani_path and os.path.isfile(mani_path):
            with open(mani_path, "r") as f:
                mani = json.load(f)

        # ---- bounds selection ----
        self.bounds = None
        if bounds_mode == "global":
            if mani is not None and "pmin" in mani and "pmax" in mani:
                pm = torch.tensor(mani["pmin"], dtype=torch.float32, device=self.device)
                px = torch.tensor(mani["pmax"], dtype=torch.float32, device=self.device)
                self.bounds = (pm, px)
                bmode_str = "global"
            else:
                pmin_g, pmax_g = self._compute_global_bounds()
                self.bounds = (pmin_g, pmax_g)
                bmode_str = "global"
        elif isinstance(bounds_mode, (tuple, list)):
            pmin, pmax = bounds_mode
            self.bounds = (
                torch.as_tensor(pmin, dtype=torch.float32, device=self.device),
                torch.as_tensor(pmax, dtype=torch.float32, device=self.device),
            )
            bmode_str = "fixed"
        else:
            bmode_str = "per_item"

        # ---- cache decision ----
        self._use_cache = False
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            if (not force_rebuild) and (mani is not None):
                cache_ok = (
                    mani.get("length") == len(self.base_ds) and
                    tuple(mani.get("grid_whd", [])) == self.grid_whd and
                    mani.get("mode") == self.mode and
                    mani.get("bounds_mode") == bmode_str
                )
                if cache_ok and self._all_item_files_exist():
                    self._use_cache = True

        # if no usable cache → build and write manifest
        if not self._use_cache:
            self._build_and_optionally_cache(bmode_str)
        else:
            # lazy path on later runs
            pass

    # ---------- helpers ----------

    def _item_paths(self, idx: int):
        if not self.cache_dir:
            return None, None, None
        base = os.path.join(self.cache_dir, f"{idx:06d}")
        return base + "_voxels.pt", base + "_meta.pt", base + "_extras.pt"

    def _all_item_files_exist(self) -> bool:
        for i in range(len(self.base_ds)):
            vp, mp, _ = self._item_paths(i)
            if not (os.path.isfile(vp) and os.path.isfile(mp)):
                return False
        return True

    def _write_manifest(self, bmode_str: str):
        if not self.cache_dir:
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
            pts = torch.as_tensor(item["points"], dtype=torch.float32)[..., :3]
            pmins.append(pts.amin(dim=0))
            pmaxs.append(pts.amax(dim=0))
        pmin_g = torch.stack(pmins, 0).amin(dim=0)
        pmax_g = torch.stack(pmaxs, 0).amax(dim=0)
        return pmin_g, pmax_g

    def _build_and_optionally_cache(self, bmode_str: str):
        # in-memory storage for the current run
        self._voxels = []
        self._metas = []
        self._pass = []

        if self.cache_dir:
            self._write_manifest(bmode_str)

        for i in tqdm(range(len(self.base_ds)), desc="Voxelizing dataset"):
            item = self.base_ds[i]
            pts_all = torch.as_tensor(item["points"], dtype=torch.float32)
            pts_xyz = pts_all[:, :3]
            colors = pts_all[:, 3:6] if (pts_all.shape[-1] == 6 and self.mode == "avg_rgb") else None

            vg = VoxelGridXYZ(
                pts_xyz,
                colors,
                grid_whd=self.grid_whd,
                bounds=self.bounds,
                mode=self.mode,
            )

            vox = vg.to_dense().to(self.device)   # [C,D,H,W]
            md = vg.meta_dict()
            for k in ("pmin", "pmax", "voxel_size"):
                md[k] = md[k].to(self.device)

            # passthrough from base item
            extra = {k: v for k, v in item.items() if k != "points"}
            if self.keep_points:
                extra["points"] = pts_all.to(self.device)

            # save to disk if needed
            if self.cache_dir:
                vp, mp, ep = self._item_paths(i)
                torch.save(vox.cpu(), vp)

                md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}
                torch.save(md_cpu, mp)

                if extra:
                    try:
                        extra_cpu = {}
                        for k, v in extra.items():
                            if torch.is_tensor(v):
                                extra_cpu[k] = v.cpu()
                            else:
                                extra_cpu[k] = v
                        torch.save(extra_cpu, ep)
                    except Exception:
                        # extras might contain non-serializable objects
                        pass

            self._voxels.append(vox)
            self._metas.append(md)
            self._pass.append(extra)

        if self.cache_dir:
            self._use_cache = True

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # lazy load from cache on later runs (when _voxels is NOT present)
        if self.cache_dir and self._use_cache and not hasattr(self, "_voxels"):
            vp, mp, ep = self._item_paths(idx)

            vox = torch.load(vp, map_location="cpu")
            md = torch.load(mp, map_location="cpu")
            for k in ("pmin", "pmax", "voxel_size"):
                if k in md and torch.is_tensor(md[k]):
                    md[k] = md[k].to(self.device)

            extras: Dict[str, Any] = {}
            if ep and os.path.isfile(ep):
                extras = torch.load(ep, map_location="cpu")

            # merge with base_ds metadata, but don't override extras
            base_item = self.base_ds[idx]
            if isinstance(base_item, dict):
                for k, v in base_item.items():
                    if k == "points":
                        continue
                    if k not in extras:
                        extras[k] = v

            out: Dict[str, Any] = {}

            # attach points (if requested)
            if self.keep_points:
                pts = None
                if isinstance(extras, dict) and "points" in extras:
                    pts = torch.as_tensor(extras["points"], dtype=torch.float32)
                elif isinstance(base_item, dict) and "points" in base_item:
                    pts = torch.as_tensor(base_item["points"], dtype=torch.float32)

                if pts is not None:
                    out["points"] = pts.to(self.device)

            # attach extras (except points)
            if isinstance(extras, dict):
                for k, v in extras.items():
                    if k == "points":
                        continue
                    out[k] = v

            out["voxels"] = vox.to(self.device)
            out["meta"] = md
            return out

        # in-memory path (first run)
        vox = self._voxels[idx].to(self.device)
        md = self._metas[idx]
        for k in ("pmin", "pmax", "voxel_size"):
            if torch.is_tensor(md[k]):
                md[k] = md[k].to(self.device)

        extra = self._pass[idx]
        out: Dict[str, Any] = {}

        if self.keep_points and isinstance(extra, dict) and "points" in extra:
            out["points"] = extra["points"].to(self.device)

        if isinstance(extra, dict):
            for k, v in extra.items():
                if k == "points":
                    continue
                out[k] = v

        out["voxels"] = vox
        out["meta"] = md
        return out
