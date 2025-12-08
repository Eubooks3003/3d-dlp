import os
import glob
from typing import List, Dict, Optional
import numpy as np
import torch
from datasets.point_cloud_datasets.to_scene_ds import TODataset
from datasets.voxel_ds import VoxelDataset
from datasets.voxelize_ds_wrapper import VoxelizedDataset

def get_point_cloud_dataset(
    ds: str,
    root: str,
    mode: str = "train",                 # split: "train" | "val" | "test"
    sample_length: int = 1,
    max_points: int = 4096,
    normalize_to_unit_cube: bool = True,
    include_rgb: bool = False,
    *,
    # --- new: voxelization toggle + params ---
    voxelize: bool = False,              # when True, return BOTH points and voxels
    voxel_grid_whd: tuple = (64, 64, 64),
    voxel_mode: str = "density",         # {"occupancy","density","moments","avg_rgb"}
    bounds_mode: "str|tuple" = "global",  # "per_item" | "global" | ((pmin),(pmax))
    keep_points: bool = True,            # keep raw points alongside voxels
    cache_dir: str = None,        # optional on-disk cache for voxelization
    cache_extras: bool = False,
    force_rebuild: bool = False,
    device=None,
):
    """
    Generic getter for point-cloud datasets.
      - ds in {"TO", "to", "to-scene"} → loads flat .plys via TODataset
         * if voxelize=True → wraps with VoxelizedDataset, returning BOTH points and voxels
      - ds == "voxel" → precomputed voxel files (*.pt pairs) via VoxelDataset (as before)
    """
    ds_key = (ds or "").lower()

    # --- precomputed voxels path (unchanged behavior) ---
    if ds_key == "voxel":
        return VoxelDataset(
            root=root,           # folder containing *_voxels.pt / *_meta.pt
            split=mode,          # "train" | "val" | "test"
            sample_length=sample_length,
        )

    if ds_key in ("to","to-scene","to_scene"):
        base = TODataset(
            root=root, split=mode, max_points=max_points,
            normalize_to_unit_cube=normalize_to_unit_cube, include_rgb=include_rgb,
        )
        if voxelize:
            return VoxelizedDataset(
                base_ds=base,
                grid_whd=voxel_grid_whd,
                mode=voxel_mode,
                bounds_mode=bounds_mode,
                keep_points=True,                   # <-- ensures 'points' is present
                device=device or torch.device("cpu"),
                cache_dir=cache_dir,
                cache_extras=cache_extras,
                force_rebuild=force_rebuild,
            )
        return base

    raise NotImplementedError(f"Unknown point-cloud dataset: {ds}")

# -------- collate for variable-size point clouds (single-frame) --------
def pc_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Pads variable-length point clouds in a batch.

    Input items:
      {
        "points": FloatTensor [N_i, C],
        "mask":   BoolTensor   [N_i],
        "path":   str,
        "id":     str
      }

    Output:
      {
        "points": FloatTensor [B, N_max, C],
        "mask":   BoolTensor   [B, N_max],
        "paths":  List[str],
        "ids":    List[str]
      }
    """
    B = len(batch)
    C = batch[0]["points"].shape[1]
    N_max = max(item["points"].shape[0] for item in batch)

    pts = torch.zeros(B, N_max, C, dtype=torch.float32)
    msk = torch.zeros(B, N_max, dtype=torch.bool)
    paths, ids = [], []

    for b, item in enumerate(batch):
        p = item["points"]
        n = p.shape[0]
        pts[b, :n] = p
        msk[b, :n] = True
        paths.append(item["path"])
        ids.append(item["id"])

    return {"points": pts, "mask": msk, "paths": paths, "ids": ids}