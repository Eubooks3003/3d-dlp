# datasets/point_cloud_datasets/mimicgen_multitask_ds.py

import os
import glob
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class MimicGenMultiTaskDataset(Dataset):
    """
    Multi-task MimicGen fused point-cloud dataset.

    Expects folder structure:
        <root>/
          <task1>_d0/
            core/
              mimicgen_from_depth_pcd/
                demo_0/
                  frame000000_fused_envcalib.ply
                  frame000001_fused_envcalib.ply
                  ...
                demo_1/
                  ...
          <task2>_d0/
            core/
              mimicgen_from_depth_pcd/
                ...

    Example path:
        /home/ubuntu/pathaklab/data/mimicgen/coffee_d0/core/mimicgen_from_depth_pcd/demo_479/frame000010_fused_envcalib.ply

    Args:
        root: Root directory containing task folders
        tasks: List of task names to include (e.g., ["coffee", "stack"]).
               If None, auto-discovers all *_d0 folders.
        split: "train" | "val" | "test"
        max_points: Max points per cloud (random downsample if exceeded)
        normalize_to_unit_cube: Center and scale to [-1,1]^3
        include_rgb: Include RGB channels if available
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        seed: Random seed for split
        proportion: Optional downsampling of total scenes

    Returns per sample:
        {
          "points": [N, C],  (C=3 or 6 for xyz[+rgb])
          "mask":   [N] bool,
          "path":   str,
          "id":     str,
          "task":   str,      # task name (e.g., "coffee")
          "demo":   int,      # demo number
          "frame":  int,      # frame number within demo
        }
    """

    def __init__(
        self,
        root: str,
        tasks: Optional[List[str]] = None,
        split: str = "train",
        max_points: int = 4096,
        normalize_to_unit_cube: bool = True,
        include_rgb: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)

        # Discover tasks
        if tasks is None:
            # Auto-discover all *_d0 folders
            task_dirs = sorted(glob.glob(os.path.join(self.root, "*_d0")))
            tasks = [os.path.basename(d).replace("_d0", "") for d in task_dirs]
            if len(tasks) == 0:
                raise RuntimeError(f"No *_d0 task folders found in {self.root}")
            print(f"[MimicGenMultiTaskDataset] Auto-discovered {len(tasks)} tasks: {tasks}")

        self.tasks = tasks
        self._task_to_idx = {t: i for i, t in enumerate(tasks)}

        # Collect all .ply files across all tasks
        all_files = []  # List of (task, demo_idx, frame_idx, path)

        for task in tasks:
            task_pcd_root = os.path.join(
                self.root, f"{task}_d0", "core", "mimicgen_from_depth_pcd"
            )
            if not os.path.isdir(task_pcd_root):
                print(f"[MimicGenMultiTaskDataset] Warning: {task_pcd_root} not found, skipping")
                continue

            # Find all demo folders
            demo_dirs = sorted(
                glob.glob(os.path.join(task_pcd_root, "demo_*")),
                key=lambda x: int(os.path.basename(x).split("_")[1])
            )

            for demo_dir in demo_dirs:
                demo_idx = int(os.path.basename(demo_dir).split("_")[1])

                # Find all frame .ply files in this demo
                ply_files = sorted(glob.glob(os.path.join(demo_dir, "*.ply")))

                for ply_path in ply_files:
                    # Parse frame number from filename like "frame000010_fused_envcalib.ply"
                    fname = os.path.basename(ply_path)
                    try:
                        frame_idx = int(fname.split("_")[0].replace("frame", ""))
                    except (ValueError, IndexError):
                        frame_idx = 0

                    all_files.append((task, demo_idx, frame_idx, ply_path))

        if len(all_files) == 0:
            raise RuntimeError(f"No .ply files found for tasks {tasks} in {self.root}")

        print(f"[MimicGenMultiTaskDataset] Found {len(all_files)} total frames across {len(tasks)} tasks")

        # Optional proportion downsampling
        proportion = float(proportion)
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {proportion}")
        n_total = len(all_files)
        n_keep = max(1, int(round(n_total * proportion)))
        all_files = all_files[:n_keep]

        # Internal train/val/test split via indices
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

        # Keep only the files for this split
        self.files = [all_files[i] for i in chosen_idx]
        if len(self.files) == 0:
            raise RuntimeError(f"No files assigned to split '{split}' (N={N})")

        # Count per task
        task_counts = {}
        for task, _, _, _ in self.files:
            task_counts[task] = task_counts.get(task, 0) + 1
        print(f"[MimicGenMultiTaskDataset] Split '{split}': {len(self.files)} samples")
        for t, c in sorted(task_counts.items()):
            print(f"  - {t}: {c} frames")

    def get_cache_dir_for_task(self, task: str) -> str:
        """
        Returns the cache directory path for a given task.
        Cache is stored under: {root}/{task}_d0/core/voxel_cache/
        """
        return os.path.join(self.root, f"{task}_d0", "core", "voxel_cache")

    def get_all_cache_dirs(self) -> Dict[str, str]:
        """
        Returns a dict mapping task name -> cache directory path.
        """
        return {task: self.get_cache_dir_for_task(task) for task in self.tasks}

    # ------------- helpers -------------

    def _read_ply(self, path: str) -> np.ndarray:
        """
        Reads a .ply file with Open3D. Returns np.ndarray [N, C] where C=3 (xyz) or 6 (xyzrgb).
        """
        pc = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pc.points, dtype=np.float32)
        if self.include_rgb and len(pc.colors) > 0:
            rgb = np.asarray(pc.colors, dtype=np.float32)
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
        task, demo_idx, frame_idx, path = self.files[idx]
        pts = self._read_ply(path)

        if self.normalize_to_unit_cube:
            pts = self._center_scale_unit_cube(pts)

        pts = self._downsample(pts)

        # Avoid torch.from_numpy() for AARCH compatibility
        pts_t = torch.tensor(pts.tolist(), dtype=torch.float32)
        sample = {
            "points": pts_t,
            "mask": torch.ones(pts_t.shape[0], dtype=torch.bool),
            "path": path,
            "id": os.path.splitext(os.path.basename(path))[0],
            "task": task,
            "task_idx": self._task_to_idx[task],
            "demo": demo_idx,
            "frame": frame_idx,
        }
        return sample
