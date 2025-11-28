# voxel_dataset.py
import os
import re
import math
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    """
    Reads precomputed voxel files from a flat folder:
      root/
        000000_voxels.pt
        000000_meta.pt
        000001_voxels.pt
        000001_meta.pt
        ...

    Usage (matches your factory):
        ds = VoxelDataset(root="/path/to/voxel_ds", split="train", sample_length=1)

    Notes:
      - `split` is made inside the dataset (no train/val/test subfolders in root).
      - Deterministic split on sorted IDs; default ratios (train=0.8, val=0.1, test=0.1).
      - `proportion` lets you use only the first p% of the split for quick experiments.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",                 # "train" | "val" | "test"
        sample_length: int = 1,               # kept for API symmetry; each item is 1 sample
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        proportion: float = 1.0,              # (0,1] take only this fraction of the chosen split
        seed: int = 42,                       # affects nothing unless you later shuffle externally
        device: Optional[torch.device] = None # leave None to keep tensors on CPU when loaded
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.sample_length = int(sample_length)
        self.split_ratios = tuple(float(x) for x in split_ratios)
        self.proportion = float(proportion)
        self.device = device  # optional move-on-load

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"VoxelDataset root not found: {self.root}")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of ['train','val','test'], got '{self.split}'")

        s = sum(self.split_ratios)
        if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            self.split_ratios = (self.split_ratios[0]/s, self.split_ratios[1]/s, self.split_ratios[2]/s)

        if not (0.0 < self.proportion <= 1.0):
            raise ValueError(f"proportion must be in (0,1], got {self.proportion}")

        random.seed(seed)

        # ---- scan folder for pairs ----
        vox_pat = re.compile(r"^(\d+)_voxels\.pt$")
        meta_pat = re.compile(r"^(\d+)_meta\.pt$")

        by_id: Dict[str, Dict[str, str]] = {}
        for fname in os.listdir(self.root):
            m = vox_pat.match(fname)
            if m:
                k = m.group(1)
                by_id.setdefault(k, {})["vox"] = os.path.join(self.root, fname)
                continue
            m = meta_pat.match(fname)
            if m:
                k = m.group(1)
                by_id.setdefault(k, {})["meta"] = os.path.join(self.root, fname)

        # keep only complete pairs
        ids_all = sorted([k for k, v in by_id.items() if "vox" in v and "meta" in v], key=lambda x: int(x))
        if len(ids_all) == 0:
            raise RuntimeError(f"No voxel/meta pairs found in {self.root}")

        # ---- split indices deterministically on sorted IDs ----
        n = len(ids_all)
        n_train = int(round(self.split_ratios[0] * n))
        n_val   = int(round(self.split_ratios[1] * n))
        # adjust to sum exactly
        n_test  = n - n_train - n_val

        ids_train = ids_all[:n_train]
        ids_val   = ids_all[n_train:n_train+n_val]
        ids_test  = ids_all[n_train+n_val:]

        if self.split == "train":
            ids_split = ids_train
        elif self.split == "val":
            ids_split = ids_val
        else:
            ids_split = ids_test

        # optional proportion cut
        k_keep = max(1, int(round(len(ids_split) * self.proportion)))
        ids_split = ids_split[:k_keep]

        # store file paths for this split
        self.items: List[Tuple[str, str, str]] = [
            (id_str, by_id[id_str]["vox"], by_id[id_str]["meta"]) for id_str in ids_split
        ]

    # ------------- Dataset API -------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        id_str, vox_path, meta_path = self.items[idx]

        vox = torch.load(vox_path)  # expected [C, D, H, W] float tensor
        meta = torch.load(meta_path)  # dict with pmin/pmax/voxel_size, etc.

        # move to requested device if provided
        if self.device is not None:
            if isinstance(vox, torch.Tensor):
                vox = vox.to(self.device, non_blocking=True)
            # move tensor values found in meta
            meta = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in meta.items()}

        # make a simple mask (nonzero occupancy/density) for convenience
        # if C==1 -> use that; else create a foreground proxy from any nonzero channel
        if isinstance(vox, torch.Tensor):
            if vox.dim() == 4:
                if vox.shape[0] == 1:
                    fg_mask = (vox[0] > 0).to(torch.bool)
                else:
                    fg_mask = (vox.abs().sum(dim=0) > 0).to(torch.bool)
            else:
                fg_mask = None
        else:
            fg_mask = None

        sample = {
            "voxels": vox,          # [C,D,H,W]
            "meta": meta,           # dict with pmin,pmax,voxel_size,W,H,D (depends on your writer)
            "id": id_str,
            "voxels_path": vox_path,
            "meta_path": meta_path,
            "fg_mask": fg_mask,     # [D,H,W] bool (optional convenience)
        }
        return sample