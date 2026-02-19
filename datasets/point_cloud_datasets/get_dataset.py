import os
import glob
from typing import List, Dict, Optional, Union
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
    bounds_mode: Union[str, tuple] = "global",  # "per_item" | "global" | ((pmin),(pmax))
    keep_points: bool = True,            # keep raw points alongside voxels
    cache_dir: str = None,        # optional on-disk cache for voxelization
    cache_extras: bool = False,
    force_rebuild: bool = False,
    max_items: Optional[int] = None,     # max items to read from cache (None = no limit)
    device=None,
    # --- kmeans caching ---
    kmeans_cache_dir: Optional[str] = None,  # path to precomputed kmeans cache
    # --- mimicgen specific ---
    task: Optional[str] = None,          # for mimicgen_voxel: single task name
    tasks: Optional[List[str]] = None,   # for mimicgen multi-task: list of task names or None for auto-discover
    max_demos: Optional[int] = None,     # for mimicgen_voxel: limit number of demos
    cache_suffix: str = "",              # for mimicgen_voxel: e.g., "_debug" for voxel_cache_debug
    task_cache_suffixes: Optional[Dict[str, str]] = None,  # for mimicgen_multitask_voxel: per-task cache suffixes
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    proportion: float = 1.0,
):
    """
    Generic getter for point-cloud datasets.
      - ds in {"TO", "to", "to-scene"} → loads flat .plys via TODataset
         * if voxelize=True → wraps with VoxelizedDataset, returning BOTH points and voxels
      - ds == "voxel" → precomputed voxel files (*.pt pairs) via VoxelDataset (as before)
      - ds == "mimicgen_voxel" → precomputed MimicGen voxel cache via MimicGenVoxelDataset
         * If task is provided: {root}/{task}_d0/core/voxel_cache/
         * If task is None: treat root as direct path to voxel_cache folder
         * Use max_demos= to limit number of demos
      - ds in {"mimicgen", "mimicgen_multitask"} → multi-task MimicGen dataset (point clouds)
         * Expects: {root}/{task}_d0/core/mimicgen_from_depth_pcd/demo_*/frame*.ply
         * if voxelize=True → wraps with VoxelizedDataset
    """
    ds_key = (ds or "").lower()

    # --- precomputed voxels path (unchanged behavior) ---
    if ds_key == "voxel":
        return VoxelDataset(
            root=root,           # folder containing *_voxels.pt / *_meta.pt
            split=mode,          # "train" | "val" | "test"
            sample_length=sample_length,
        )

    # --- MimicGen precomputed voxel cache (from preprocess_mimicgen_voxels.py) ---
    if ds_key == "mimicgen_voxel":
        from datasets.point_cloud_datasets.mimicgen_ds import MimicGenVoxelDataset

        # If task is provided: {root}/{task}_d0/core/voxel_cache/
        # If task is None: treat root as direct path to voxel cache
        return MimicGenVoxelDataset(
            root=root,
            task=task,
            split=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_demos=max_demos,
            cache_suffix=cache_suffix,
            device=device,
            kmeans_cache_dir=kmeans_cache_dir,
        )

    # --- MimicGen MULTI-TASK precomputed voxel cache ---
    if ds_key in ("mimicgen_multitask_voxel", "mimicgen_multi_voxel"):
        from datasets.point_cloud_datasets.mimicgen_ds import MimicGenMultiTaskVoxelDataset

        # Reads precomputed voxels from per-task cache directories
        # Supports different cache folder names per task via task_cache_suffixes
        return MimicGenMultiTaskVoxelDataset(
            root=root,
            tasks=tasks,
            task_cache_suffixes=task_cache_suffixes,
            default_cache_suffix=cache_suffix,
            split=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_demos_per_task=max_demos,
            proportion=proportion,
            device=device,
            kmeans_cache_dir=kmeans_cache_dir,
        )

    # --- Real-world tabletop voxel cache (from preprocess_realworld_voxels.py) ---
    if ds_key in ("realworld_voxel", "realworld", "tabletop_voxel"):
        from datasets.point_cloud_datasets.realworld_ds import RealWorldVoxelDataset

        # root should point to the directory containing voxel_cache/
        # table_types can be specified via 'tasks' parameter
        return RealWorldVoxelDataset(
            root=root,
            table_types=tasks,  # reuse 'tasks' param for table types
            split=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            include_augmented=False,  # can be added as param if needed
            cache_suffix=cache_suffix,
            device=device,
            kmeans_cache_dir=kmeans_cache_dir,
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
                max_items=max_items,
                kmeans_cache_dir=kmeans_cache_dir,
            )
        return base

    # --- MimicGen multi-task dataset ---
    if ds_key in ("mimicgen", "mimicgen_multitask", "mimicgen_multi"):
        from datasets.point_cloud_datasets.mimicgen_multitask_ds import MimicGenMultiTaskDataset

        base = MimicGenMultiTaskDataset(
            root=root,
            tasks=tasks,
            split=mode,
            max_points=max_points,
            normalize_to_unit_cube=normalize_to_unit_cube,
            include_rgb=include_rgb,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        if voxelize:
            # Auto-determine cache_dir if not provided:
            # Use a combined cache under {root}/voxel_cache_combined/
            # or use per-task caching if only one task
            if cache_dir is None:
                if len(base.tasks) == 1:
                    # Single task: use task-specific cache (with optional suffix)
                    cache_dir = base.get_cache_dir_for_task(base.tasks[0], cache_suffix)
                else:
                    # Multiple tasks: use combined cache at root level
                    suffix = cache_suffix if cache_suffix else ""
                    cache_dir = os.path.join(root, f"voxel_cache_combined{suffix}")
                print(f"[get_point_cloud_dataset] Auto-set cache_dir: {cache_dir}")

            return VoxelizedDataset(
                base_ds=base,
                grid_whd=voxel_grid_whd,
                mode=voxel_mode,
                bounds_mode=bounds_mode,
                keep_points=keep_points,
                device=device or torch.device("cpu"),
                cache_dir=cache_dir,
                cache_extras=cache_extras,
                force_rebuild=force_rebuild,
                max_items=max_items,
                kmeans_cache_dir=kmeans_cache_dir,
            )
        return base

    # --- Legacy single-task MimicGen (flat directory) ---
    if ds_key in ("mimicgen_flat", "mimicgen_single"):
        from datasets.point_cloud_datasets.mimicgen_ds import MimicGenPointCloudDataset

        base = MimicGenPointCloudDataset(
            root=root,
            split=mode,
            max_points=max_points,
            normalize_to_unit_cube=normalize_to_unit_cube,
            include_rgb=include_rgb,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        if voxelize:
            # For flat mimicgen, cache_dir defaults to {root}/voxel_cache if not provided
            if cache_dir is None:
                cache_dir = os.path.join(root, "voxel_cache")
                print(f"[get_point_cloud_dataset] Auto-set cache_dir: {cache_dir}")

            return VoxelizedDataset(
                base_ds=base,
                grid_whd=voxel_grid_whd,
                mode=voxel_mode,
                bounds_mode=bounds_mode,
                keep_points=keep_points,
                device=device or torch.device("cpu"),
                cache_dir=cache_dir,
                cache_extras=cache_extras,
                force_rebuild=force_rebuild,
                kmeans_cache_dir=kmeans_cache_dir,
                max_items=max_items,
            )
        return base

    # --- RLBench multi-task dataset ---
    if ds_key in ("rlbench", "rlbench_multitask"):
        from datasets.point_cloud_datasets.rlbench_ds import RLBenchPointCloudDataset

        # For RLBench, 'tasks' is the list of task names
        # 'splits' would be data splits like ['train', 'test_data']
        # We pass 'tasks' as tasks and use a separate kwarg for data splits if needed
        base = RLBenchPointCloudDataset(
            root=root,
            splits=None,  # Auto-discover data splits (train/test_data/etc)
            tasks=tasks,  # Task names (slide_block_to_color_target, etc)
            split=mode,   # Internal train/val/test split
            max_points=max_points,
            normalize_to_unit_cube=normalize_to_unit_cube,
            include_rgb=include_rgb,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        if voxelize:
            # For RLBench, voxel caches are stored per-episode
            # Use a combined cache at root level for training
            if cache_dir is None:
                cache_dir = os.path.join(root, "voxel_cache_combined")
                print(f"[get_point_cloud_dataset] Auto-set cache_dir: {cache_dir}")

            return VoxelizedDataset(
                base_ds=base,
                grid_whd=voxel_grid_whd,
                mode=voxel_mode,
                bounds_mode=bounds_mode,
                keep_points=keep_points,
                device=device or torch.device("cpu"),
                cache_dir=cache_dir,
                cache_extras=cache_extras,
                force_rebuild=force_rebuild,
                max_items=max_items,
                kmeans_cache_dir=kmeans_cache_dir,
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