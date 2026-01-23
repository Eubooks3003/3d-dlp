# datasets/point_cloud_datasets/realworld_ds.py

import os
import glob
import json
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RealWorldVoxelDataset(Dataset):
    """
    Real-world tabletop voxel dataset that reads from precomputed voxel cache.

    Cache structure (created by preprocess_realworld_voxels.py):
        voxel_cache/
          manifest.json
          round_table/
            scene_09_voxels.pt
            scene_09_meta.pt
            scene_09_extras.pt
            ...
          long_table/
            ...
          augmented/  (optional)
            ...

    Train/val/test splits are done internally via index splitting.

    Each sample:
        {
          "voxels": [C, D, H, W] tensor,
          "meta": dict with pmin, pmax, voxel_size, etc.,
          "fg_mask": [D, H, W] bool tensor,
          "id": str,
          "table_type": str,
          "voxels_path": str,
        }
    """

    def __init__(
        self,
        root: str,
        table_types: Optional[List[str]] = None,  # None = auto-discover
        split: str = "train",                     # "train" | "val" | "test"
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        proportion: float = 1.0,
        include_augmented: bool = False,
        cache_suffix: str = "",                   # e.g., "_debug" for voxel_cache_debug
        device: Optional[torch.device] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.device = device
        self.include_augmented = include_augmented

        # Use root directly as cache directory
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

        # Discover or use specified table types
        if table_types is None:
            table_types = self._discover_table_types()
        self.table_types = table_types

        if include_augmented and "augmented" not in self.table_types:
            aug_dir = os.path.join(self.cache_dir, "augmented")
            if os.path.isdir(aug_dir):
                self.table_types = list(self.table_types) + ["augmented"]

        # Discover all cached files: list of (table_type, scene_id, vox_path, meta_path, extras_path)
        all_files = self._discover_files()
        if len(all_files) == 0:
            raise RuntimeError(f"No voxel files found in {self.cache_dir}")

        # Optional proportion downsampling
        proportion = float(proportion)
        if not (0.0 < proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {proportion}")
        n_total = len(all_files)
        n_keep = max(1, int(round(n_total * proportion)))
        all_files = all_files[:n_keep]

        # Internal train/val/test split
        train_ratio = float(train_ratio)
        val_ratio = float(val_ratio)
        if train_ratio + val_ratio > 1.0 + 1e-6:
            raise ValueError(f"train_ratio + val_ratio must be <= 1, got {train_ratio + val_ratio}")

        # Shuffle deterministically
        rng = np.random.RandomState(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)

        N = len(indices)
        n_train = int(round(train_ratio * N))
        n_val = int(round(val_ratio * N))
        n_train = min(n_train, N)
        n_val = min(n_val, max(0, N - n_train))

        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train:n_train + n_val])
        test_indices = set(indices[n_train + n_val:])

        split_l = split.lower()
        if split_l == "train":
            chosen_indices = train_indices
        elif split_l == "val":
            chosen_indices = val_indices
        elif split_l == "test":
            chosen_indices = test_indices
        else:
            raise ValueError(f"Unknown split '{split}' (expected 'train'/'val'/'test')")

        # Collect chosen files
        self.items: List[Tuple[str, str, str, str, str]] = [
            all_files[i] for i in sorted(chosen_indices)
        ]

        if len(self.items) == 0:
            raise RuntimeError(f"No files assigned to split '{split}' (N={N})")

        print(f"[RealWorldVoxelDataset] {split}: {len(self.items)} samples from {len(self.table_types)} table types")

    def _discover_table_types(self) -> List[str]:
        """Auto-discover table type folders in the cache directory."""
        known_tables = ["round_table", "long_table", "square_table", "office_table", "round_table_augmented"]
        found = []

        for table in known_tables:
            table_dir = os.path.join(self.cache_dir, table)
            if os.path.isdir(table_dir):
                vox_files = glob.glob(os.path.join(table_dir, "*_voxels.pt"))
                if vox_files:
                    found.append(table)

        # Also check for files directly in the root directory (no subdirectory structure)
        if not found:
            vox_files = glob.glob(os.path.join(self.cache_dir, "*_voxels.pt"))
            if vox_files:
                found.append(".")  # Use "." to indicate root directory

        return found

    def _discover_files(self) -> List[Tuple[str, str, str, str, str]]:
        """
        Discover all cached voxel files.
        Returns list of (table_type, scene_id, vox_path, meta_path, extras_path).
        """
        files = []

        for table_type in self.table_types:
            # Handle "." as special case for files directly in root
            if table_type == ".":
                table_dir = self.cache_dir
            else:
                table_dir = os.path.join(self.cache_dir, table_type)

            if not os.path.isdir(table_dir):
                continue

            vox_files = sorted(glob.glob(os.path.join(table_dir, "*_voxels.pt")))

            for vox_path in vox_files:
                fname = os.path.basename(vox_path)
                scene_id = fname.replace("_voxels.pt", "")

                meta_path = os.path.join(table_dir, f"{scene_id}_meta.pt")
                extras_path = os.path.join(table_dir, f"{scene_id}_extras.pt")

                files.append((table_type, scene_id, vox_path, meta_path, extras_path))

        print(f"[RealWorldVoxelDataset] Found {len(files)} files across {len(self.table_types)} table types")
        return files

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        table_type, scene_id, vox_path, meta_path, extras_path = self.items[idx]

        # Load voxels
        vox = torch.load(vox_path)  # [C, D, H, W]

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
            "voxels": vox,                          # [C, D, H, W]
            "fg_mask": fg_mask,                     # [D, H, W] bool
        }
        return sample
