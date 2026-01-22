#!/usr/bin/env python3
"""
Voxelization preprocessing for real-world tabletop point cloud datasets.

Expects input structure (from real_world_augment_dataset.py output):
    <root>/
    ├── round_table/
    │   ├── 09.ply, 10.ply, ...
    │   └── ...
    ├── long_table/
    │   └── ...
    ├── square_table/
    │   └── ...
    ├── office_table/
    │   └── ...
    └── augmented/  (optional: augmented scenes)
        └── scene_XX_aug_YYY.ply

Or flat structure:
    <root>/
    ├── scene_01_aug_000.ply
    ├── scene_01_aug_001.ply
    └── ...

Usage:
    # Process all table types
    python preprocess_realworld_voxels.py --root /path/to/pc

    # Process specific table types
    python preprocess_realworld_voxels.py --root /path/to/pc --tables round_table long_table

    # Debug mode: visualize one sample per table type on wandb
    python preprocess_realworld_voxels.py --root /path/to/pc --debug

    # Debug dataset mode: process limited files for testing
    python preprocess_realworld_voxels.py --root /path/to/pc --debug_dataset --debug_n 10

Output structure:
    <root>/voxel_cache/
    ├── manifest.json
    ├── round_table/
    │   ├── scene_09_voxels.pt
    │   ├── scene_09_meta.pt
    │   └── ...
    └── ...
"""

import os
import sys
import argparse
import glob
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
import numpy as np
import open3d as o3d

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.voxelize_ds_wrapper import VoxelGridXYZ


def safe_tensor(arr, dtype=torch.float32):
    """Convert numpy array to torch tensor without torch.from_numpy() for AARCH compatibility."""
    return torch.tensor(arr.tolist(), dtype=dtype)


def read_ply(path: str, include_rgb: bool = True) -> np.ndarray:
    """Read a .ply file with Open3D. Returns [N, 3] or [N, 6] for xyz(+rgb)."""
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points, dtype=np.float32)
    if include_rgb and len(pc.colors) > 0:
        rgb = np.asarray(pc.colors, dtype=np.float32)
        if rgb.shape[0] == xyz.shape[0]:
            return np.concatenate([xyz, rgb], axis=-1)
    return xyz


def center_scale_unit_cube(pts: np.ndarray) -> np.ndarray:
    """Center and isotropically scale to fit in [-1,1]^3."""
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


def discover_table_types(root: str):
    """Auto-discover table type folders (round_table, long_table, etc.)."""
    known_tables = ["round_table", "long_table", "square_table", "office_table"]
    found = []
    for table in known_tables:
        table_dir = os.path.join(root, table)
        if os.path.isdir(table_dir):
            # Check if it has any PLY files
            ply_files = glob.glob(os.path.join(table_dir, "*.ply"))
            if ply_files:
                found.append(table)
    return found


def get_table_files(root: str, table_type: str):
    """Get all PLY files for a table type."""
    table_dir = os.path.join(root, table_type)
    if not os.path.isdir(table_dir):
        return []

    ply_files = sorted(glob.glob(os.path.join(table_dir, "*.ply")))

    files = []
    for ply_path in ply_files:
        fname = os.path.basename(ply_path)
        # Extract scene ID from filename (e.g., "09.ply" -> "09", "scene_09_aug_000.ply" -> "scene_09_aug_000")
        scene_id = os.path.splitext(fname)[0]
        files.append((scene_id, ply_path))

    return files


def get_augmented_files(root: str):
    """Get all augmented PLY files from the augmented folder."""
    aug_dir = os.path.join(root, "augmented")
    if not os.path.isdir(aug_dir):
        return []

    ply_files = sorted(glob.glob(os.path.join(aug_dir, "*.ply")))

    files = []
    for ply_path in ply_files:
        fname = os.path.basename(ply_path)
        scene_id = os.path.splitext(fname)[0]
        files.append((scene_id, ply_path))

    return files


def get_cache_structure(cache_dir: str, table_type: str):
    """Get set of already cached scene IDs for a table type."""
    table_cache_dir = os.path.join(cache_dir, table_type)
    if not os.path.isdir(table_cache_dir):
        return set()

    vox_files = glob.glob(os.path.join(table_cache_dir, "*_voxels.pt"))
    cached = set()
    for vox_path in vox_files:
        fname = os.path.basename(vox_path)
        scene_id = fname.replace("_voxels.pt", "")
        cached.add(scene_id)

    return cached


def _voxelize_single_file(args_tuple):
    """Worker function for parallel voxelization."""
    (scene_id, ply_path, table_cache_dir, table_type,
     grid_whd, voxel_mode, include_rgb, normalize_to_unit_cube, max_points) = args_tuple

    try:
        vox_path = os.path.join(table_cache_dir, f"{scene_id}_voxels.pt")
        meta_path = os.path.join(table_cache_dir, f"{scene_id}_meta.pt")
        extras_path = os.path.join(table_cache_dir, f"{scene_id}_extras.pt")

        # Read and preprocess
        pts = read_ply(ply_path, include_rgb=include_rgb)
        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)

        if max_points is not None and pts.shape[0] > max_points:
            idx_sample = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx_sample]

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        # Voxelize with per-item bounds (no fixed bounds for real-world data)
        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=None, mode=voxel_mode)
        vox = vg.to_dense().cpu()
        md = vg.meta_dict()
        md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}

        # Save
        torch.save(vox, vox_path)
        torch.save(md_cpu, meta_path)

        # Save extras (path info for later reference)
        extras = {
            "path": ply_path,
            "id": scene_id,
            "table_type": table_type,
            "points": pts,  # Store raw points too
        }
        torch.save(extras, extras_path)

        return (scene_id, True, None)
    except Exception as e:
        return (scene_id, False, str(e))


def voxelize_table_type(
    root: str,
    table_type: str,
    grid_whd=(64, 64, 64),
    voxel_mode="avg_rgb",
    include_rgb=True,
    normalize_to_unit_cube=True,
    max_points=None,
    cache_name="voxel_cache",
    cache_suffix="",
    max_files=None,
    num_workers=None,
):
    """Voxelize all PLY files for a single table type and save to cache."""
    cache_name = f"{cache_name}{cache_suffix}"
    cache_dir = os.path.join(root, cache_name)
    table_cache_dir = os.path.join(cache_dir, table_type)

    # Get all files for this table type
    files = get_table_files(root, table_type)
    if len(files) == 0:
        print(f"[{table_type}] No PLY files found, skipping.")
        return

    # Get already cached files
    cached = get_cache_structure(cache_dir, table_type)

    # Filter to only uncached files
    to_process = [(sid, path) for sid, path in files if sid not in cached]

    # Limit files if max_files is set (for debug mode)
    if max_files is not None and max_files > 0:
        to_process = to_process[:max_files]

    total_files = len(files)
    total_cached = len(cached)
    total_to_process = len(to_process)

    if total_to_process == 0:
        print(f"[{table_type}] Already complete: {total_cached}/{total_files} files cached.")
        return

    print(f"\n[{table_type}] Processing {total_to_process} files ({total_cached}/{total_files} already cached)...")

    os.makedirs(table_cache_dir, exist_ok=True)

    # Prepare arguments for workers
    worker_args = [
        (scene_id, ply_path, table_cache_dir, table_type,
         grid_whd, voxel_mode, include_rgb, normalize_to_unit_cube, max_points)
        for scene_id, ply_path in to_process
    ]

    # Use multiprocessing if num_workers > 1
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Default to min(num_cpus, 8)

    if num_workers > 1 and total_to_process > 1:
        print(f"  Using {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_voxelize_single_file, worker_args),
                total=len(worker_args),
                desc=f"  [{table_type}] Voxelizing"
            ))
    else:
        # Sequential processing
        results = []
        for args in tqdm(worker_args, desc=f"  [{table_type}] Voxelizing"):
            results.append(_voxelize_single_file(args))

    # Report errors
    errors = [(sid, err) for sid, success, err in results if not success]
    if errors:
        print(f"  [{table_type}] {len(errors)} errors:")
        for sid, err in errors[:5]:
            print(f"    {sid}: {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    success_count = sum(1 for _, success, _ in results if success)
    print(f"  [{table_type}] Done! {success_count}/{total_to_process} voxels saved to {table_cache_dir}")


def voxelize_augmented(
    root: str,
    grid_whd=(64, 64, 64),
    voxel_mode="avg_rgb",
    include_rgb=True,
    normalize_to_unit_cube=True,
    max_points=None,
    cache_name="voxel_cache",
    cache_suffix="",
    max_files=None,
    num_workers=None,
):
    """Voxelize all augmented PLY files."""
    cache_name = f"{cache_name}{cache_suffix}"
    cache_dir = os.path.join(root, cache_name)
    aug_cache_dir = os.path.join(cache_dir, "augmented")

    # Get all augmented files
    files = get_augmented_files(root)
    if len(files) == 0:
        print("[augmented] No PLY files found, skipping.")
        return

    # Get already cached files
    cached = get_cache_structure(cache_dir, "augmented")

    # Filter to only uncached files
    to_process = [(sid, path) for sid, path in files if sid not in cached]

    # Limit files if max_files is set
    if max_files is not None and max_files > 0:
        to_process = to_process[:max_files]

    total_files = len(files)
    total_cached = len(cached)
    total_to_process = len(to_process)

    if total_to_process == 0:
        print(f"[augmented] Already complete: {total_cached}/{total_files} files cached.")
        return

    print(f"\n[augmented] Processing {total_to_process} files ({total_cached}/{total_files} already cached)...")

    os.makedirs(aug_cache_dir, exist_ok=True)

    # Prepare arguments for workers
    worker_args = [
        (scene_id, ply_path, aug_cache_dir, "augmented",
         grid_whd, voxel_mode, include_rgb, normalize_to_unit_cube, max_points)
        for scene_id, ply_path in to_process
    ]

    # Use multiprocessing if num_workers > 1
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    if num_workers > 1 and total_to_process > 1:
        print(f"  Using {num_workers} workers...")
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_voxelize_single_file, worker_args),
                total=len(worker_args),
                desc="  [augmented] Voxelizing"
            ))
    else:
        results = []
        for args in tqdm(worker_args, desc="  [augmented] Voxelizing"):
            results.append(_voxelize_single_file(args))

    # Report errors
    errors = [(sid, err) for sid, success, err in results if not success]
    if errors:
        print(f"  [augmented] {len(errors)} errors:")
        for sid, err in errors[:5]:
            print(f"    {sid}: {err}")

    success_count = sum(1 for _, success, _ in results if success)
    print(f"  [augmented] Done! {success_count}/{total_to_process} voxels saved to {aug_cache_dir}")


def debug_visualize(
    root: str,
    table_types: list,
    grid_whd=(64, 64, 64),
    voxel_mode="avg_rgb",
    include_rgb=True,
    normalize_to_unit_cube=True,
    wandb_project="realworld-voxel-debug",
):
    """Debug mode: voxelize one sample per table type and log to wandb."""
    import wandb
    from eval.eval_vox import log_rgb_voxels

    wandb.init(project=wandb_project, name=f"debug-voxels-{len(table_types)}-tables")
    print(f"[debug] Logging to wandb project: {wandb_project}")

    step = 0
    for table_type in table_types:
        files = get_table_files(root, table_type)
        if len(files) == 0:
            print(f"[{table_type}] No files found, skipping.")
            continue

        # Just take the first file
        scene_id, ply_path = files[0]
        print(f"\n[{table_type}] Visualizing {scene_id}: {ply_path}")

        # Read and preprocess
        pts = read_ply(ply_path, include_rgb=include_rgb)
        print(f"  Raw points: {pts.shape}")

        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)
            print(f"  After normalization: min={pts[:,:3].min(axis=0)}, max={pts[:,:3].max(axis=0)}")

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        # Compute bounds (per-item)
        pmin = pts_xyz.amin(dim=0)
        pmax = pts_xyz.amax(dim=0)
        bounds = (pmin, pmax)
        print(f"  Bounds: min={pmin.tolist()}, max={pmax.tolist()}")

        # Voxelize
        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=bounds, mode=voxel_mode)
        vox = vg.to_dense()
        print(f"  Voxel shape: {vox.shape}, non-zero: {(vox.abs().sum(dim=0) > 0).sum().item()}")

        # Log to wandb
        log_rgb_voxels(
            name=f"debug/{table_type}",
            rgb_vol=vox,
            alpha_vol=None,
            KPx=None,
            step=step,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )
        print(f"  Logged to wandb: debug/{table_type}")
        step += 1

    # Also visualize augmented if available
    aug_files = get_augmented_files(root)
    if aug_files:
        scene_id, ply_path = aug_files[0]
        print(f"\n[augmented] Visualizing {scene_id}: {ply_path}")

        pts = read_ply(ply_path, include_rgb=include_rgb)
        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=None, mode=voxel_mode)
        vox = vg.to_dense()

        log_rgb_voxels(
            name="debug/augmented",
            rgb_vol=vox,
            alpha_vol=None,
            KPx=None,
            step=step,
            mode="splat",
            topk=60000,
            alpha_thresh=0.05,
            pad=2.0,
            show_axes=True,
        )
        print(f"  Logged to wandb: debug/augmented")

    wandb.finish()
    print("\n[debug] Done! Check wandb for visualizations.")


def write_manifest(root: str, table_types: list, cache_name: str, grid_whd: tuple, voxel_mode: str):
    """Write manifest.json to the cache directory."""
    cache_dir = os.path.join(root, cache_name)
    manifest_path = os.path.join(cache_dir, "manifest.json")

    # Count total files per table type
    counts = {}
    for table_type in table_types + ["augmented"]:
        table_cache_dir = os.path.join(cache_dir, table_type)
        if os.path.isdir(table_cache_dir):
            vox_files = glob.glob(os.path.join(table_cache_dir, "*_voxels.pt"))
            counts[table_type] = len(vox_files)

    manifest = {
        "dataset": "realworld_tabletop",
        "table_types": table_types,
        "counts": counts,
        "total": sum(counts.values()),
        "grid_whd": list(grid_whd),
        "mode": voxel_mode,
        "bounds_mode": "per_item",
        "complete": True,
    }

    os.makedirs(cache_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"  Total voxels: {manifest['total']}")
    for table_type, count in counts.items():
        print(f"    {table_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess real-world point clouds to voxels")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing table type folders (round_table, etc.)")
    parser.add_argument("--tables", type=str, nargs="*", default=None,
                        help="Specific table types to process (default: auto-discover)")
    parser.add_argument("--grid_whd", type=int, nargs=3, default=[64, 64, 64],
                        help="Voxel grid dimensions (default: 64 64 64)")
    parser.add_argument("--voxel_mode", type=str, default="avg_rgb",
                        choices=["occupancy", "density", "avg_rgb"],
                        help="Voxelization mode (default: avg_rgb)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Max points per cloud (default: no limit)")
    parser.add_argument("--no_rgb", action="store_true",
                        help="Don't include RGB (xyz only)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Don't normalize to unit cube")
    parser.add_argument("--include_augmented", action="store_true",
                        help="Also process the augmented folder")
    parser.add_argument("--cache_name", type=str, default="voxel_cache",
                        help="Name for the voxel cache folder (default: voxel_cache)")

    # Debug modes
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: voxelize one sample per table type and log to wandb")
    parser.add_argument("--debug_dataset", action="store_true",
                        help="Debug dataset mode: process limited files and save to voxel_cache_debug")
    parser.add_argument("--debug_n", type=int, default=10,
                        help="Number of files to process per table type in debug_dataset mode (default: 10)")
    parser.add_argument("--wandb_project", type=str, default="realworld-voxel-debug",
                        help="Wandb project for debug visualizations")
    parser.add_argument("--num_workers", "-j", type=int, default=None,
                        help="Number of parallel workers (default: min(num_cpus, 8))")

    args = parser.parse_args()

    # Discover or use specified table types
    if args.tables is None:
        table_types = discover_table_types(args.root)
        print(f"Auto-discovered {len(table_types)} table types: {table_types}")
    else:
        table_types = args.tables
        print(f"Processing specified table types: {table_types}")

    if len(table_types) == 0:
        print("No table types found! Make sure the root directory contains folders like round_table/, long_table/, etc.")
        return

    # Debug mode: just visualize one sample per table type
    if args.debug:
        debug_visualize(
            root=args.root,
            table_types=table_types,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            wandb_project=args.wandb_project,
        )
        return

    # Debug dataset mode: process limited files and save to voxel_cache_debug
    if args.debug_dataset:
        print(f"\n=== DEBUG DATASET MODE: Processing up to {args.debug_n} files per table type ===")
        for table_type in table_types:
            voxelize_table_type(
                root=args.root,
                table_type=table_type,
                grid_whd=tuple(args.grid_whd),
                voxel_mode=args.voxel_mode,
                include_rgb=not args.no_rgb,
                normalize_to_unit_cube=not args.no_normalize,
                max_points=args.max_points,
                cache_name=args.cache_name,
                cache_suffix="_debug",
                max_files=args.debug_n,
                num_workers=args.num_workers,
            )

        if args.include_augmented:
            voxelize_augmented(
                root=args.root,
                grid_whd=tuple(args.grid_whd),
                voxel_mode=args.voxel_mode,
                include_rgb=not args.no_rgb,
                normalize_to_unit_cube=not args.no_normalize,
                max_points=args.max_points,
                cache_name=args.cache_name,
                cache_suffix="_debug",
                max_files=args.debug_n,
                num_workers=args.num_workers,
            )

        write_manifest(args.root, table_types, f"{args.cache_name}_debug", tuple(args.grid_whd), args.voxel_mode)
        print(f"\nDebug dataset saved to {args.cache_name}_debug/")
        return

    # Full processing mode
    for table_type in table_types:
        voxelize_table_type(
            root=args.root,
            table_type=table_type,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            max_points=args.max_points,
            cache_name=args.cache_name,
            num_workers=args.num_workers,
        )

    if args.include_augmented:
        voxelize_augmented(
            root=args.root,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            max_points=args.max_points,
            cache_name=args.cache_name,
            num_workers=args.num_workers,
        )

    write_manifest(args.root, table_types, args.cache_name, tuple(args.grid_whd), args.voxel_mode)
    print("\nAll done!")


if __name__ == "__main__":
    main()
