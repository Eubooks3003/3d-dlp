# datasets/point_cloud_datasets/rlbench_ds.py

import os
import glob
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class RLBenchPointCloudDataset(Dataset):
    """
    Multi-task RLBench fused point-cloud dataset.

    Expects folder structure:
        <root>/
          <split>/                         # e.g., train, test, test_data
            <task>/                        # e.g., slide_block_to_color_target
              all_variations/
                episodes/
                  episode0/
                    fused_pcd/
                      000000.ply
                      000001.ply
                      ...
                  episode1/
                    fused_pcd/
                      ...

    Example path:
        /home/ubuntu/pathaklab/data/lyuxing/rlbench/full/test_data/slide_block_to_color_target/all_variations/episodes/episode0/fused_pcd/000000.ply

    Args:
        root: Root directory containing split folders
        splits: List of split names to include (e.g., ["train", "test_data"]).
                If None, auto-discovers all.
        tasks: List of task names to include. If None, auto-discovers all.
        split: "train" | "val" | "test" (for internal train/val/test split)
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
          "split_name": str,   # data split (train/test_data)
          "task":   str,       # task name
          "episode": int,      # episode number
          "frame":  int,       # frame number within episode
        }
    """

    def __init__(
        self,
        root: str,
        splits: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        split: str = "train",
        max_points: int = 4096,
        normalize_to_unit_cube: bool = True,
        include_rgb: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,
        max_items: Optional[int] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)
        self.max_items = max_items

        # Discover splits
        if splits is None:
            split_dirs = sorted(glob.glob(os.path.join(self.root, "*")))
            splits = [os.path.basename(d) for d in split_dirs if os.path.isdir(d)]
            if len(splits) == 0:
                raise RuntimeError(f"No split folders found in {self.root}")
            print(f"[RLBenchPointCloudDataset] Auto-discovered {len(splits)} splits: {splits}")

        self.splits_list = splits

        # Discover tasks
        if tasks is None:
            all_tasks = set()
            for s in splits:
                split_path = os.path.join(self.root, s)
                if os.path.isdir(split_path):
                    task_dirs = glob.glob(os.path.join(split_path, "*"))
                    for td in task_dirs:
                        if os.path.isdir(td):
                            all_tasks.add(os.path.basename(td))
            tasks = sorted(list(all_tasks))
            if len(tasks) == 0:
                raise RuntimeError(f"No task folders found in {self.root}")
            print(f"[RLBenchPointCloudDataset] Auto-discovered {len(tasks)} tasks: {tasks}")

        self.tasks = tasks
        self._task_to_idx = {t: i for i, t in enumerate(tasks)}

        # Collect all .ply files across all splits/tasks
        all_files = []  # List of (split_name, task, episode, frame, path)

        for split_name in splits:
            for task in tasks:
                # Standard RLBench structure
                episodes_root = os.path.join(
                    self.root, split_name, task, "all_variations", "episodes"
                )
                if not os.path.isdir(episodes_root):
                    # Try without all_variations
                    episodes_root = os.path.join(self.root, split_name, task, "episodes")

                if not os.path.isdir(episodes_root):
                    continue

                # Find all episode folders
                episode_dirs = sorted(
                    glob.glob(os.path.join(episodes_root, "episode*")),
                    key=lambda x: int(os.path.basename(x).replace("episode", ""))
                )

                for episode_dir in episode_dirs:
                    episode_idx = int(os.path.basename(episode_dir).replace("episode", ""))
                    pcd_dir = os.path.join(episode_dir, "fused_pcd")

                    if not os.path.isdir(pcd_dir):
                        continue

                    # Find all PLY files
                    ply_files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")))

                    for ply_path in ply_files:
                        fname = os.path.basename(ply_path)
                        try:
                            frame_idx = int(os.path.splitext(fname)[0])
                        except ValueError:
                            frame_idx = 0

                        all_files.append((split_name, task, episode_idx, frame_idx, ply_path))

        if len(all_files) == 0:
            raise RuntimeError(f"No .ply files found in {self.root}")

        print(f"[RLBenchPointCloudDataset] Found {len(all_files)} total frames")

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
        for _, task, _, _, _ in self.files:
            task_counts[task] = task_counts.get(task, 0) + 1
        print(f"[RLBenchPointCloudDataset] Split '{split}': {len(self.files)} samples")
        for t, c in sorted(task_counts.items()):
            print(f"  - {t}: {c} frames")

    def get_cache_dir_for_episode(self, split_name: str, task: str, episode: int) -> str:
        """
        Returns the voxel cache directory path for a given episode.
        """
        return os.path.join(
            self.root, split_name, task, "all_variations", "episodes",
            f"episode{episode}", "voxel_cache"
        )

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
        base_len = len(self.files)
        if self.max_items is not None:
            return min(base_len, self.max_items)
        return base_len

    def __getitem__(self, idx: int) -> Dict[str, object]:
        split_name, task, episode_idx, frame_idx, path = self.files[idx]
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
            "split_name": split_name,
            "task": task,
            "task_idx": self._task_to_idx[task],
            "episode": episode_idx,
            "frame": frame_idx,
        }
        return sample
