# datasets/point_cloud_datasets/mimicgen_ds.py

import os
import glob
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class MimicGenPointCloudDataset(Dataset):
    """
    MimicGen fused point-cloud dataset.

    Expects flat .ply files under:
        <root>/
          demo_0_frame000000_fused.ply
          demo_0_frame000001_fused.ply
          ...

    Train/val/test are done *internally* via index splitting, NOT folders.

    Returns one point cloud per __getitem__ (use sample_length=1).

    Each sample:
        {
          "points": [N, C]  (C=3 or 6 for xyz[+rgb]),
          "mask":   [N] bool (all True),
          "path":   str,
          "id":     str   # filename without extension
        }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",             # "train" | "val" | "test"
        max_points: int = 4096,
        normalize_to_unit_cube: bool = True,
        include_rgb: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,          # optional extra downsampling of scenes
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)

        # enumerate all .ply files in flat directory
        all_files = sorted(glob.glob(os.path.join(self.root, "*.ply")))
        if len(all_files) == 0:
            raise RuntimeError(f"No .ply files found in {self.root}")

        # optional scene-level proportion (like TODataset)
        proportion = float(proportion)
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {proportion}")
        n_total = len(all_files)
        n_keep = max(1, int(round(n_total * proportion)))
        all_files = all_files[:n_keep]

        # ------- internal train/val/test split via indices -------
        train_ratio = float(train_ratio)
        val_ratio = float(val_ratio)
        if train_ratio + val_ratio > 1.0 + 1e-6:
            raise ValueError(f"train_ratio + val_ratio must be <= 1, got {train_ratio + val_ratio}")

        N = len(all_files)
        indices = np.arange(N)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        n_train = int(round(train_ratio * N))
        n_val = int(round(val_ratio * N))
        n_train = min(n_train, N)
        n_val = min(n_val, max(0, N - n_train))
        n_test = N - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        split_l = split.lower()
        if split_l == "train":
            chosen_idx = np.sort(train_idx)
        elif split_l == "val":
            chosen_idx = np.sort(val_idx)
        elif split_l == "test":
            chosen_idx = np.sort(test_idx)
        else:
            raise ValueError(f"Unknown split '{split}' (expected 'train'/'val'/'test')")

        # finally, keep only the files for this split
        self.files = [all_files[i] for i in chosen_idx]
        if len(self.files) == 0:
            raise RuntimeError(f"No files assigned to split '{split}' (N={N})")

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
        return len(self.files)

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