# datasets/point_cloud_datasets/mimicgen_ds.py

import os
import glob
import json
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.voxelize_ds_wrapper import load_voxel


class MimicGenVoxelDataset(Dataset):
    """
    MimicGen voxel dataset that reads from precomputed voxel cache.

    Cache structure (created by preprocess_mimicgen_voxels.py):
        voxel_cache/
          manifest.json
          demo_0/
            frame0_voxels.pt
            frame0_meta.pt
            frame0_extras.pt
            ...
          demo_1/
            ...

    Two ways to specify the cache location:
        1. root + task: cache at {root}/{task}_d0/core/voxel_cache/
        2. root only (task=None): root is the direct path to voxel_cache/

    Train/val/test are done *internally* via index splitting on demos.
    Use max_demos to limit the number of demos (complete trajectories) used.

    Each sample:
        {
          "voxels": [C, D, H, W] tensor,
          "meta": dict with pmin, pmax, voxel_size, etc.,
          "fg_mask": [D, H, W] bool tensor,
          "id": str,
          "task": str,
          "demo": int,
          "frame": int,
          "path": str (original PLY path),
        }
    """

    def __init__(
        self,
        root: str,
        task: Optional[str] = None,       # task name, or None to use root as direct cache path
        split: str = "train",             # "train" | "val" | "test"
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,          # optional extra downsampling
        max_demos: Optional[int] = None,  # limit number of demos (not frames)
        cache_suffix: str = "",           # e.g., "_debug" for voxel_cache_debug
        device: Optional[torch.device] = None,
    ):
        self.root = os.path.abspath(root)
        self.task = task
        self.split = split
        self.max_demos = max_demos
        self.device = device

        # Build cache directory path
        # If task is provided: {root}/{task}_d0/core/voxel_cache{suffix}/
        # If task is None: treat root as direct path to voxel cache
        if task is not None:
            cache_name = f"voxel_cache{cache_suffix}"
            self.cache_dir = os.path.join(self.root, f"{task}_d0", "core", cache_name)
        else:
            self.cache_dir = self.root

        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(f"Voxel cache not found: {self.cache_dir}")

        # Read manifest if available
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

        # Discover all cached frames: list of (demo_idx, frame_idx, vox_path, meta_path, extras_path)
        all_frames = self._discover_frames()
        if len(all_frames) == 0:
            raise RuntimeError(f"No voxel files found in {self.cache_dir}")

        # Optional proportion downsampling (applied before split)
        proportion = float(proportion)
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {proportion}")
        n_total = len(all_frames)
        n_keep = max(1, int(round(n_total * proportion)))
        all_frames = all_frames[:n_keep]

        # ------- internal train/val/test split via demo indices -------
        # Split by demo to avoid data leakage between train/val/test
        train_ratio = float(train_ratio)
        val_ratio = float(val_ratio)
        if train_ratio + val_ratio > 1.0 + 1e-6:
            raise ValueError(f"train_ratio + val_ratio must be <= 1, got {train_ratio + val_ratio}")

        # Group frames by demo
        demos = {}
        for demo_idx, frame_idx, vox_path, meta_path, extras_path in all_frames:
            if demo_idx not in demos:
                demos[demo_idx] = []
            demos[demo_idx].append((demo_idx, frame_idx, vox_path, meta_path, extras_path))

        demo_indices = sorted(demos.keys())

        # Shuffle demos deterministically
        rng = np.random.RandomState(seed)
        shuffled_demos = demo_indices.copy()
        rng.shuffle(shuffled_demos)

        # Apply max_demos BEFORE split (so max_demos=10 → 8 train, 1 val, 1 test)
        if self.max_demos is not None and self.max_demos < len(shuffled_demos):
            shuffled_demos = shuffled_demos[:self.max_demos]

        N_demos = len(shuffled_demos)
        n_train = int(round(train_ratio * N_demos))
        n_val = int(round(val_ratio * N_demos))
        n_train = min(n_train, N_demos)
        n_val = min(n_val, max(0, N_demos - n_train))

        train_demos = set(shuffled_demos[:n_train])
        val_demos = set(shuffled_demos[n_train:n_train + n_val])
        test_demos = set(shuffled_demos[n_train + n_val:])

        split_l = split.lower()
        if split_l == "train":
            chosen_demos = train_demos
        elif split_l == "val":
            chosen_demos = val_demos
        elif split_l == "test":
            chosen_demos = test_demos
        else:
            raise ValueError(f"Unknown split '{split}' (expected 'train'/'val'/'test')")

        # Collect all frames from chosen demos
        self.items: List[Tuple[int, int, str, str, str]] = []
        for demo_idx in sorted(chosen_demos):
            self.items.extend(demos[demo_idx])

        if len(self.items) == 0:
            raise RuntimeError(f"No files assigned to split '{split}' (N_demos={N_demos})")

        print(f"[MimicGenVoxelDataset] {split}: {len(self.items)} frames from {len(chosen_demos)} demos")

    def _discover_frames(self) -> List[Tuple[int, int, str, str, str]]:
        """
        Discover all cached voxel frames.
        Returns list of (demo_idx, frame_idx, vox_path, meta_path, extras_path).
        """
        frames = []

        demo_dirs = sorted(
            glob.glob(os.path.join(self.cache_dir, "demo_*")),
            key=lambda x: int(os.path.basename(x).split("_")[1])
        )

        print(f"[MimicGenVoxelDataset] Scanning {len(demo_dirs)} demos...")
        for demo_dir in tqdm(demo_dirs, desc="Discovering demos", leave=False):
            demo_idx = int(os.path.basename(demo_dir).split("_")[1])

            # Find all voxel files in this demo
            vox_files = sorted(glob.glob(os.path.join(demo_dir, "frame*_voxels.pt")))

            for vox_path in vox_files:
                fname = os.path.basename(vox_path)
                try:
                    # Extract frame index from "frameN_voxels.pt"
                    frame_idx = int(fname.replace("frame", "").replace("_voxels.pt", ""))
                except (ValueError, IndexError):
                    continue

                # Build paths for meta and extras (don't check existence - assume they exist)
                meta_path = os.path.join(demo_dir, f"frame{frame_idx}_meta.pt")
                extras_path = os.path.join(demo_dir, f"frame{frame_idx}_extras.pt")

                frames.append((demo_idx, frame_idx, vox_path, meta_path, extras_path))

        print(f"[MimicGenVoxelDataset] Found {len(frames)} frames across {len(demo_dirs)} demos")
        return frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        demo_idx, frame_idx, vox_path, meta_path, extras_path = self.items[idx]

        # Load voxels
        vox = load_voxel(vox_path)  # [C, D, H, W]

        # Load meta (skip existence check for speed - assume it exists)
        try:
            meta = torch.load(meta_path)
        except FileNotFoundError:
            meta = {}

        # Skip loading extras - contains raw points (~100KB), not needed for training

        # Move to device if specified
        if self.device is not None:
            if isinstance(vox, torch.Tensor):
                vox = vox.to(self.device, non_blocking=True)
            meta = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in meta.items()}

        # Create foreground mask
        if isinstance(vox, torch.Tensor) and vox.dim() == 4:
            if vox.shape[0] == 1:
                fg_mask = (vox[0] > 0).to(torch.bool)
            else:
                fg_mask = (vox.abs().sum(dim=0) > 0).to(torch.bool)
        else:
            # Fallback: create empty mask if vox format unexpected
            fg_mask = torch.zeros((64, 64, 64), dtype=torch.bool)

        task_str = self.task if self.task else "mimicgen"
        sample = {
            "voxels": vox,                          # [C, D, H, W]
            "meta": meta,                           # dict with pmin, pmax, voxel_size, etc.
            "fg_mask": fg_mask,                     # [D, H, W] bool
            "id": f"{task_str}_demo{demo_idx}_frame{frame_idx}",
            "task": task_str,                       # never None (use "mimicgen" as fallback)
            "demo": demo_idx,
            "frame": frame_idx,
            "voxels_path": vox_path,
        }
        return sample


# Keep the old class name as an alias for backwards compatibility
MimicGenPointCloudDataset = MimicGenVoxelDataset


class MimicGenMultiTaskVoxelDataset(Dataset):
    """
    Multi-task MimicGen dataset that reads from precomputed voxel caches.

    Supports different cache folder names per task (e.g., voxel_cache vs voxel_cache_new).

    Cache structure per task (created by preprocess_mimicgen_voxels.py):
        {root}/{task}_d0/core/{cache_folder}/
          manifest.json
          demo_0/
            frame0_voxels.pt
            frame0_meta.pt
            ...
          demo_1/
            ...

    Args:
        root: Root directory containing task folders (e.g., /path/to/mimicgen/)
        tasks: List of task names to include. If None, auto-discovers all *_d0 folders.
        task_cache_suffixes: Dict mapping task name -> cache suffix.
                             e.g., {"coffee": "", "stack": "_new"} means:
                               - coffee uses voxel_cache/
                               - stack uses voxel_cache_new/
                             If None, uses default_cache_suffix for all tasks.
        default_cache_suffix: Default suffix when task not in task_cache_suffixes (default: "")
        split: "train" | "val" | "test"
        train_ratio: Fraction for training split (default: 0.8)
        val_ratio: Fraction for validation split (default: 0.1)
        seed: Random seed for reproducible splits
        max_demos_per_task: Optional limit on demos per task (None = no limit)
        device: Optional device to load tensors to

    Each sample:
        {
          "voxels": [C, D, H, W] tensor,
          "meta": dict with pmin, pmax, voxel_size, etc.,
          "fg_mask": [D, H, W] bool tensor,
          "id": str,
          "task": str,
          "task_idx": int,
          "demo": int,
          "frame": int,
          "voxels_path": str,
        }
    """

    def __init__(
        self,
        root: str,
        tasks: Optional[List[str]] = None,
        task_cache_suffixes: Optional[Dict[str, str]] = None,
        default_cache_suffix: str = "",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_demos_per_task: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.device = device
        self.task_cache_suffixes = task_cache_suffixes or {}
        self.default_cache_suffix = default_cache_suffix

        # Discover tasks if not provided
        if tasks is None:
            task_dirs = sorted(glob.glob(os.path.join(self.root, "*_d0")))
            tasks = [os.path.basename(d).replace("_d0", "") for d in task_dirs]
            if len(tasks) == 0:
                raise RuntimeError(f"No *_d0 task folders found in {self.root}")
            print(f"[MimicGenMultiTaskVoxelDataset] Auto-discovered {len(tasks)} tasks: {tasks}")

        self.tasks = tasks
        self._task_to_idx = {t: i for i, t in enumerate(tasks)}

        # Build cache directories for each task
        self.task_cache_dirs = {}
        for task in tasks:
            suffix = self.task_cache_suffixes.get(task, self.default_cache_suffix)
            cache_name = f"voxel_cache{suffix}"
            cache_dir = os.path.join(self.root, f"{task}_d0", "core", cache_name)
            if not os.path.isdir(cache_dir):
                print(f"[MimicGenMultiTaskVoxelDataset] Warning: cache not found for {task}: {cache_dir}")
                continue
            self.task_cache_dirs[task] = cache_dir

        if len(self.task_cache_dirs) == 0:
            raise RuntimeError(f"No valid voxel caches found for any task in {self.root}")

        # Discover all frames from all tasks
        # Format: (task, demo_idx, frame_idx, vox_path, meta_path)
        all_frames = []
        for task, cache_dir in self.task_cache_dirs.items():
            task_frames = self._discover_task_frames(task, cache_dir, max_demos_per_task)
            all_frames.extend(task_frames)
            print(f"  - {task}: {len(task_frames)} frames")

        if len(all_frames) == 0:
            raise RuntimeError(f"No voxel files found in any task cache")

        print(f"[MimicGenMultiTaskVoxelDataset] Total: {len(all_frames)} frames across {len(self.task_cache_dirs)} tasks")

        # ------- Train/val/test split by (task, demo) to avoid leakage -------
        train_ratio = float(train_ratio)
        val_ratio = float(val_ratio)
        if train_ratio + val_ratio > 1.0 + 1e-6:
            raise ValueError(f"train_ratio + val_ratio must be <= 1, got {train_ratio + val_ratio}")

        # Group frames by (task, demo)
        demo_groups = {}  # (task, demo_idx) -> list of frames
        for frame_info in all_frames:
            task, demo_idx = frame_info[0], frame_info[1]
            key = (task, demo_idx)
            if key not in demo_groups:
                demo_groups[key] = []
            demo_groups[key].append(frame_info)

        demo_keys = sorted(demo_groups.keys())

        # Shuffle demos deterministically
        rng = np.random.RandomState(seed)
        shuffled_keys = demo_keys.copy()
        rng.shuffle(shuffled_keys)

        N_demos = len(shuffled_keys)
        n_train = int(round(train_ratio * N_demos))
        n_val = int(round(val_ratio * N_demos))
        n_train = min(n_train, N_demos)
        n_val = min(n_val, max(0, N_demos - n_train))

        train_keys = set(shuffled_keys[:n_train])
        val_keys = set(shuffled_keys[n_train:n_train + n_val])
        test_keys = set(shuffled_keys[n_train + n_val:])

        split_l = split.lower()
        if split_l == "train":
            chosen_keys = train_keys
        elif split_l == "val":
            chosen_keys = val_keys
        elif split_l == "test":
            chosen_keys = test_keys
        else:
            raise ValueError(f"Unknown split '{split}' (expected 'train'/'val'/'test')")

        # Collect all frames from chosen demos
        self.items = []
        for key in sorted(chosen_keys):
            self.items.extend(demo_groups[key])

        if len(self.items) == 0:
            raise RuntimeError(f"No files assigned to split '{split}' (N_demos={N_demos})")

        # Count per task
        task_counts = {}
        for frame_info in self.items:
            task = frame_info[0]
            task_counts[task] = task_counts.get(task, 0) + 1

        print(f"[MimicGenMultiTaskVoxelDataset] Split '{split}': {len(self.items)} frames from {len(chosen_keys)} demos")
        for t, c in sorted(task_counts.items()):
            print(f"  - {t}: {c} frames")

    def _discover_task_frames(
        self, task: str, cache_dir: str, max_demos: Optional[int]
    ) -> List[Tuple[str, int, int, str, str]]:
        """
        Discover all cached voxel frames for a single task.
        Returns list of (task, demo_idx, frame_idx, vox_path, meta_path).
        """
        frames = []

        demo_dirs = sorted(
            glob.glob(os.path.join(cache_dir, "demo_*")),
            key=lambda x: int(os.path.basename(x).split("_")[1])
        )

        # Apply max_demos limit
        if max_demos is not None and max_demos < len(demo_dirs):
            demo_dirs = demo_dirs[:max_demos]

        for demo_dir in demo_dirs:
            demo_idx = int(os.path.basename(demo_dir).split("_")[1])

            # Find all voxel files in this demo
            vox_files = sorted(glob.glob(os.path.join(demo_dir, "frame*_voxels.pt")))

            for vox_path in vox_files:
                fname = os.path.basename(vox_path)
                try:
                    frame_idx = int(fname.replace("frame", "").replace("_voxels.pt", ""))
                except (ValueError, IndexError):
                    continue

                meta_path = os.path.join(demo_dir, f"frame{frame_idx}_meta.pt")
                frames.append((task, demo_idx, frame_idx, vox_path, meta_path))

        return frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        task, demo_idx, frame_idx, vox_path, meta_path = self.items[idx]

        # Load voxels
        vox = load_voxel(vox_path)  # [C, D, H, W]

        # Load meta
        try:
            meta = torch.load(meta_path)
        except FileNotFoundError:
            meta = {}

        # Move to device if specified
        if self.device is not None:
            if isinstance(vox, torch.Tensor):
                vox = vox.to(self.device, non_blocking=True)
            meta = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in meta.items()}

        # Create foreground mask
        if isinstance(vox, torch.Tensor) and vox.dim() == 4:
            if vox.shape[0] == 1:
                fg_mask = (vox[0] > 0).to(torch.bool)
            else:
                fg_mask = (vox.abs().sum(dim=0) > 0).to(torch.bool)
        else:
            fg_mask = torch.zeros((64, 64, 64), dtype=torch.bool)

        sample = {
            "voxels": vox,
            "meta": meta,
            "fg_mask": fg_mask,
            "id": f"{task}_demo{demo_idx}_frame{frame_idx}",
            "task": task,
            "task_idx": self._task_to_idx[task],
            "demo": demo_idx,
            "frame": frame_idx,
            "voxels_path": vox_path,
        }
        return sample
