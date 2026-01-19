import os
import glob
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import open3d as o3d

class TODataset(Dataset):
    """
    TO dataset with flat .ply files under:
        <root>/
          TO-vanilla/
            train/
              id0.ply
              id1.ply
              ...
            val/
              ...
            test/
              ...

    Returns one point cloud per __getitem__ (use sample_length=1).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_points: int = 4096,
        normalize_to_unit_cube: bool = True,
        include_rgb: bool = False,
        proportion: float = 0.4,
        max_items: Optional[int] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)
        self.max_items = max_items

        self.split_dir = os.path.join(self.root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # enumerate .ply files
        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.ply")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .ply files found in {self.split_dir}")


        proportion = float(proportion)
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {proportion}")

        n_total = len(self.files)
        n_keep = max(1, int(round(n_total * proportion)))


        if n_keep < n_total:
            self.files = self.files[:n_keep]
        else:
            self.files = self.files
        self._o3d = None  # lazy

    # ------------- helpers -------------

    def _read_ply(self, path: str) -> np.ndarray:
        """
        Reads a .ply file with Open3D. Returns np.ndarray [N, C] where C=3 (xyz) or 6 (xyzrgb).
        """
        pc = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pc.points, dtype=np.float32)
        if self.include_rgb and len(pc.colors) > 0:
            rgb = np.asarray(pc.colors, dtype=np.float32)  # already 0..1 in Open3D
            if rgb.shape[0] == xyz.shape[0]:
                return np.concatenate([xyz, rgb], axis=-1)
        return xyz

    @staticmethod
    def _center_scale_unit_cube(pts: np.ndarray) -> np.ndarray:
        """
        Center and isotropically scale to fit in [-1,1]^3 (only xyz affected).
        """
        out = pts.copy()
        xyz = out[:, :3]
        if xyz.shape[0] == 0:
            return out
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        scale = (maxs - mins).max() + 1e-8
        out[:, :3] = (xyz - center[None, :]) / scale * 2.0
        return out

    def _downsample(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] <= self.max_points:
            return pts
        idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
        return pts[idx]

    # ------------- Dataset API -------------
    def __len__(self) -> int:
        base_len = len(self.files)
        if self.max_items is not None:
            return min(base_len, self.max_items)
        return base_len

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files[idx]
        pts = self._read_ply(path)                  # [N, 3] or [N, 6]

        if self.normalize_to_unit_cube:
            pts = self._center_scale_unit_cube(pts)

        pts = self._downsample(pts)

        pts_t = torch.from_numpy(pts).float()       # [N, C]
        sample = {
            "points": pts_t,                        # [N, C], C=3 or 6
            "mask": torch.ones(pts_t.shape[0], dtype=torch.bool),
            "path": path,
            "id": os.path.splitext(os.path.basename(path))[0],
        }
        return sample

