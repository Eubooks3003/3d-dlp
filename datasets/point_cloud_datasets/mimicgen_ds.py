# datasets/point_cloud_datasets/mimicgen_ds.py

import os
import glob
import json
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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
        vox = torch.load(vox_path)  # [C, D, H, W]

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
            fg_mask = None

        task_str = self.task if self.task else "mimicgen"
        sample = {
            "voxels": vox,                          # [C, D, H, W]
            "meta": meta,                           # dict with pmin, pmax, voxel_size, etc.
            "fg_mask": fg_mask,                     # [D, H, W] bool
            "id": f"{task_str}_demo{demo_idx}_frame{frame_idx}",
            "task": self.task,
            "demo": demo_idx,
            "frame": frame_idx,
            "voxels_path": vox_path,
        }
        return sample


# Keep the old class name as an alias for backwards compatibility
MimicGenPointCloudDataset = MimicGenVoxelDataset
