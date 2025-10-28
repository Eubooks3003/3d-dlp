import os
import glob
from typing import List, Dict, Optional
import numpy as np
import torch
from datasets.point_cloud_datasets.to_scene_ds import TODataset
from datasets.voxel_ds import VoxelDataset

def get_point_cloud_dataset(
    ds: str,
    root: str,
    mode: str = "train",
    sample_length: int = 1,
    max_points: int = 4096,
    normalize_to_unit_cube: bool = True,
    include_rgb: bool = False,
):
    """
    Generic getter for point-cloud datasets. For your TO dataset, pass ds="to" or "to-scene".
    """
    if ds == "TO":
        return TODataset(
            root=root,
            split=mode,
            max_points=max_points,
            normalize_to_unit_cube=normalize_to_unit_cube,
            include_rgb=include_rgb,
        )
    if ds == "voxel":
      return VoxelDataset(
          root=root,         # folder containing *_voxels.pt / *_meta.pt pairs
          split=mode,        # "train" | "val" | "test"
          sample_length=sample_length,
      )
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