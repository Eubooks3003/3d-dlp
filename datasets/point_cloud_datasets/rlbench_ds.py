# datasets/point_cloud_datasets/rlbench_ds.py

import os
import glob
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

from datasets.voxelize_ds_wrapper import load_voxel


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


class RLBenchMultiTaskVoxelDataset(Dataset):
    """
    Multi-task RLBench voxel dataset — reads per-episode voxel caches produced
    by the preprocess_rlbench_rgbo_all pipeline.

    Cache structure:
        <root>/
          <split_dir>/                     # e.g. train_data, test_data
            <task>/
              all_variations/
                episodes/
                  episode<N>/
                    voxel_cache/
                      manifest.json
                      000000_voxels.pt      # [4,D,H,W] RGBO (compressed sparse ok)
                      000000_meta.pt
                      000000_extras.pt
                      ...
                    kmeans_cache/           # (optional, populated at train start)
                      000000_kmeans.pt
                      ...

    Per sample:
        {
          "voxels":   [3,D,H,W] float  (RGB only — channel 3 = occupancy is split out as fg_mask)
          "fg_mask":  [D,H,W]   bool   (GT occupancy from the RGBO 4th channel)
          "meta":     dict (pmin/pmax/voxel_size from preprocess, + kmeans if cached)
          "id": "<task>_ep<N>_frame<F>",
          "task": str, "task_idx": int,
          "split_dir": str, "episode": int, "frame": int,
          "voxels_path": str,
        }

    Train/val/test split is internal, by (task, episode) to avoid leakage —
    frames within the same episode stay together. The filesystem split
    directories (train_data/test_data) are merged and re-split.

    Args:
        root: data root (e.g. /home/ellina/Desktop/data/rlbench)
        tasks: list of task names; if None, auto-discovers all task dirs with voxel caches
        splits: list of filesystem split dirs to pull from; if None, auto-discovers (typically train_data+test_data)
        split: internal "train" | "val" | "test"
        frame_stride: keep only every Nth frame per episode (default 1 = all)
        max_frames_per_episode: optional cap on frames per episode (applied after stride)
        train_ratio / val_ratio / seed / proportion: standard split knobs
        precompute_kmeans: enable get_kmeans_path() so train_dlp_voxel.py caches per frame
    """

    def __init__(
        self,
        root: str,
        tasks: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        split: str = "train",
        frame_stride: int = 1,
        max_frames_per_episode: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,
        device: Optional[torch.device] = None,
        precompute_kmeans: bool = False,
        kmeans_cache_dir: Optional[str] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.device = device
        self.precompute_kmeans = precompute_kmeans
        self.kmeans_cache_dir = kmeans_cache_dir
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames_per_episode = max_frames_per_episode
        self.proportion = float(proportion)
        if not (0.0 < self.proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {self.proportion}")

        if splits is None:
            splits = sorted(
                os.path.basename(d) for d in glob.glob(os.path.join(self.root, "*"))
                if os.path.isdir(d)
            )
            if len(splits) == 0:
                raise RuntimeError(f"No split dirs under {self.root}")
        self.splits_list = splits

        if tasks is None:
            discovered = set()
            for s in splits:
                for d in glob.glob(os.path.join(self.root, s, "*")):
                    if os.path.isdir(d):
                        discovered.add(os.path.basename(d))
            tasks = sorted(discovered)
            if len(tasks) == 0:
                raise RuntimeError(f"No tasks under {self.root}/{{splits}}")
            print(f"[RLBenchMultiTaskVoxelDataset] Auto-discovered {len(tasks)} tasks: {tasks}")
        self.tasks = tasks
        self._task_to_idx = {t: i for i, t in enumerate(tasks)}

        all_frames: List[Tuple[str, str, int, int, str, str]] = []
        for split_dir in splits:
            for task in tasks:
                episodes_root = os.path.join(
                    self.root, split_dir, task, "all_variations", "episodes"
                )
                if not os.path.isdir(episodes_root):
                    continue

                episode_dirs = sorted(
                    glob.glob(os.path.join(episodes_root, "episode*")),
                    key=lambda p: int(os.path.basename(p).replace("episode", ""))
                )
                for ep_dir in episode_dirs:
                    ep_idx = int(os.path.basename(ep_dir).replace("episode", ""))
                    cache_dir = os.path.join(ep_dir, "voxel_cache")
                    if not os.path.isdir(cache_dir):
                        continue

                    vox_files = sorted(glob.glob(os.path.join(cache_dir, "*_voxels.pt")))
                    kept = 0
                    for vp in vox_files:
                        stem = os.path.basename(vp).replace("_voxels.pt", "")
                        try:
                            frame_idx = int(stem)
                        except ValueError:
                            continue
                        if frame_idx % self.frame_stride != 0:
                            continue
                        if self.max_frames_per_episode is not None and kept >= self.max_frames_per_episode:
                            break
                        meta_path = os.path.join(cache_dir, f"{stem}_meta.pt")
                        all_frames.append(
                            (split_dir, task, ep_idx, frame_idx, vp, meta_path)
                        )
                        kept += 1

        if len(all_frames) == 0:
            raise RuntimeError(
                f"No voxel frames found under {self.root} "
                f"(splits={splits}, tasks={tasks}, stride={self.frame_stride})"
            )

        # optional per-task proportion downsample (before split)
        if self.proportion < 1.0:
            by_task: Dict[str, list] = {}
            for f in all_frames:
                by_task.setdefault(f[1], []).append(f)
            rng = np.random.RandomState(seed)
            all_frames = []
            for t in sorted(by_task):
                ft = by_task[t]
                n_keep = max(1, int(round(len(ft) * self.proportion)))
                idx_keep = sorted(rng.choice(len(ft), size=n_keep, replace=False).tolist())
                all_frames.extend(ft[i] for i in idx_keep)
                print(f"  - {t}: kept {n_keep}/{len(ft)} frames ({self.proportion:.0%})")

        # group by (task, split_dir, episode) so a whole episode lands in one split
        ep_groups: Dict[Tuple[str, str, int], list] = {}
        for f in all_frames:
            split_dir, task, ep_idx = f[0], f[1], f[2]
            ep_groups.setdefault((task, split_dir, ep_idx), []).append(f)

        ep_keys = sorted(ep_groups.keys())
        rng = np.random.RandomState(seed)
        shuffled = ep_keys.copy()
        rng.shuffle(shuffled)

        train_ratio = float(train_ratio)
        val_ratio = float(val_ratio)
        if train_ratio + val_ratio > 1.0 + 1e-6:
            raise ValueError(f"train_ratio+val_ratio must be <= 1, got {train_ratio+val_ratio}")
        N = len(shuffled)
        n_train = min(int(round(train_ratio * N)), N)
        n_val = min(int(round(val_ratio * N)), max(0, N - n_train))

        split_l = split.lower()
        if split_l == "train":
            chosen = set(shuffled[:n_train])
        elif split_l == "val":
            chosen = set(shuffled[n_train:n_train + n_val])
        elif split_l == "test":
            chosen = set(shuffled[n_train + n_val:])
        else:
            raise ValueError(f"Unknown split '{split}'")

        self.items: List[Tuple[str, str, int, int, str, str]] = []
        for k in sorted(chosen):
            self.items.extend(ep_groups[k])
        if len(self.items) == 0:
            raise RuntimeError(f"No frames assigned to split '{split}' (episodes={N})")

        task_counts: Dict[str, int] = {}
        for f in self.items:
            task_counts[f[1]] = task_counts.get(f[1], 0) + 1
        print(f"[RLBenchMultiTaskVoxelDataset] Split '{split}': {len(self.items)} frames from {len(chosen)} episodes")
        for t, c in sorted(task_counts.items()):
            print(f"  - {t}: {c} frames")

    # ------------- API -------------

    def __len__(self) -> int:
        return len(self.items)

    def get_kmeans_path(self, idx: int) -> Optional[str]:
        split_dir, task, ep_idx, frame_idx, _, _ = self.items[idx]
        if self.precompute_kmeans:
            ep_dir = os.path.join(
                self.root, split_dir, task, "all_variations", "episodes",
                f"episode{ep_idx}"
            )
            return os.path.join(ep_dir, "kmeans_cache", f"{frame_idx:06d}_kmeans.pt")
        if self.kmeans_cache_dir is not None:
            return os.path.join(self.kmeans_cache_dir, f"{idx:06d}_kmeans.pt")
        return None

    def _load_kmeans_into_meta(self, idx: int, meta: dict):
        km_path = self.get_kmeans_path(idx)
        if km_path is not None and os.path.exists(km_path):
            km = torch.load(km_path, map_location="cpu")
            meta["kmeans_kp"] = km["kp"]
            meta["kmeans_cov"] = km["cov"]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        split_dir, task, ep_idx, frame_idx, vox_path, meta_path = self.items[idx]

        # [4,D,H,W] RGBO (compressed sparse handled by load_voxel)
        vox = load_voxel(vox_path).float()
        if vox.dim() != 4 or vox.shape[0] < 4:
            raise RuntimeError(
                f"Expected [4,D,H,W] RGBO voxel at {vox_path}, got {tuple(vox.shape)}"
            )
        rgb = vox[:3].contiguous()                        # [3,D,H,W]
        fg_mask = (vox[3] > 0).to(torch.bool).contiguous()  # [D,H,W] GT occupancy

        try:
            meta = torch.load(meta_path)
        except FileNotFoundError:
            meta = {}

        if self.device is not None:
            rgb = rgb.to(self.device, non_blocking=True)
            fg_mask = fg_mask.to(self.device, non_blocking=True)
            meta = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in meta.items()
            }

        self._load_kmeans_into_meta(idx, meta)

        return {
            "voxels": rgb,
            "fg_mask": fg_mask,
            "meta": meta,
            "id": f"{task}_ep{ep_idx}_frame{frame_idx}",
            "task": task,
            "task_idx": self._task_to_idx[task],
            "split_dir": split_dir,
            "episode": ep_idx,
            "frame": frame_idx,
            "voxels_path": vox_path,
        }
