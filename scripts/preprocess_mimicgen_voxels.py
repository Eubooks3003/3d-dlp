#!/usr/bin/env python3
"""
Parallelizable voxelization preprocessing for MimicGen multi-task dataset.

Run on multiple machines with different --tasks to parallelize:
    Machine 1: python preprocess_mimicgen_voxels.py --root /path/to/mimicgen --tasks coffee stack
    Machine 2: python preprocess_mimicgen_voxels.py --root /path/to/mimicgen --tasks kitchen square
    ...

Or process all tasks on one machine:
    python preprocess_mimicgen_voxels.py --root /path/to/mimicgen

Each task's voxels are saved to: {root}/{task}_d0/core/voxel_cache/
"""

import os
import sys
import argparse
import glob
import json
from tqdm import tqdm

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


def discover_tasks(root: str):
    """Auto-discover all *_d0 task folders."""
    task_dirs = sorted(glob.glob(os.path.join(root, "*_d0")))
    return [os.path.basename(d).replace("_d0", "") for d in task_dirs]


def get_task_files(root: str, task: str):
    """Get all .ply files for a task, sorted by demo and frame."""
    task_pcd_root = os.path.join(root, f"{task}_d0", "core", "mimicgen_from_depth_pcd")
    if not os.path.isdir(task_pcd_root):
        return []

    files = []
    demo_dirs = sorted(
        glob.glob(os.path.join(task_pcd_root, "demo_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    for demo_dir in demo_dirs:
        demo_idx = int(os.path.basename(demo_dir).split("_")[1])
        ply_files = sorted(glob.glob(os.path.join(demo_dir, "*.ply")))

        for ply_path in ply_files:
            fname = os.path.basename(ply_path)
            try:
                frame_idx = int(fname.split("_")[0].replace("frame", ""))
            except (ValueError, IndexError):
                frame_idx = 0
            files.append((demo_idx, frame_idx, ply_path))

    return files


def compute_global_bounds_for_task(files, include_rgb=True, normalize_to_unit_cube=True):
    """Compute global bounds across all files for a task."""
    print(f"  Computing global bounds across {len(files)} files...")
    pmins, pmaxs = [], []

    for _, _, path in tqdm(files, desc="  Scanning bounds", leave=False):
        pts = read_ply(path, include_rgb=include_rgb)
        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)
        xyz = pts[:, :3]
        pmins.append(xyz.min(axis=0))
        pmaxs.append(xyz.max(axis=0))

    pmin = np.stack(pmins, 0).min(axis=0)
    pmax = np.stack(pmaxs, 0).max(axis=0)
    return pmin, pmax


def voxelize_task(
    root: str,
    task: str,
    grid_whd=(64, 64, 64),
    voxel_mode="avg_rgb",
    include_rgb=True,
    normalize_to_unit_cube=True,
    max_points=None,
    force_rebuild=False,
):
    """Voxelize all frames for a single task and save to cache."""
    cache_dir = os.path.join(root, f"{task}_d0", "core", "voxel_cache")
    manifest_path = os.path.join(cache_dir, "manifest.json")

    # Check if already done
    if not force_rebuild and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            mani = json.load(f)
        if mani.get("complete", False):
            print(f"[{task}] Already complete ({mani.get('length', '?')} files), skipping. Use --force to rebuild.")
            return

    files = get_task_files(root, task)
    if len(files) == 0:
        print(f"[{task}] No files found, skipping.")
        return

    print(f"\n[{task}] Processing {len(files)} frames...")
    os.makedirs(cache_dir, exist_ok=True)

    # Compute global bounds for this task
    pmin, pmax = compute_global_bounds_for_task(files, include_rgb, normalize_to_unit_cube)
    bounds = (safe_tensor(pmin), safe_tensor(pmax))

    print(f"  Bounds: min={pmin}, max={pmax}")

    # Voxelize each file
    for idx, (demo_idx, frame_idx, path) in enumerate(tqdm(files, desc=f"  [{task}] Voxelizing")):
        vox_path = os.path.join(cache_dir, f"{idx:06d}_voxels.pt")
        meta_path = os.path.join(cache_dir, f"{idx:06d}_meta.pt")
        extras_path = os.path.join(cache_dir, f"{idx:06d}_extras.pt")

        # Skip if already exists
        if not force_rebuild and os.path.exists(vox_path) and os.path.exists(meta_path):
            continue

        # Read and preprocess
        pts = read_ply(path, include_rgb=include_rgb)
        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)

        if max_points is not None and pts.shape[0] > max_points:
            idx_sample = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx_sample]

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        # Voxelize
        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=bounds, mode=voxel_mode)
        vox = vg.to_dense().cpu()
        md = vg.meta_dict()
        md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}

        # Save
        torch.save(vox, vox_path)
        torch.save(md_cpu, meta_path)

        # Save extras (path info for later reference)
        extras = {
            "path": path,
            "id": os.path.splitext(os.path.basename(path))[0],
            "task": task,
            "demo": demo_idx,
            "frame": frame_idx,
            "points": pts,  # Store raw points too
        }
        torch.save(extras, extras_path)

    # Write manifest
    manifest = {
        "task": task,
        "length": len(files),
        "grid_whd": list(grid_whd),
        "mode": voxel_mode,
        "bounds_mode": "global",
        "pmin": pmin.tolist(),
        "pmax": pmax.tolist(),
        "complete": True,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [{task}] Done! Saved {len(files)} voxels to {cache_dir}")


def debug_visualize_tasks(
    root: str,
    tasks: list,
    grid_whd=(64, 64, 64),
    voxel_mode="avg_rgb",
    include_rgb=True,
    normalize_to_unit_cube=True,
    wandb_project="mimicgen-voxel-debug",
):
    """Debug mode: voxelize one frame per task and log to wandb."""
    import wandb
    from eval.eval_vox import log_rgb_voxels

    # Initialize wandb
    wandb.init(project=wandb_project, name=f"debug-voxels-{len(tasks)}-tasks")
    print(f"[debug] Logging to wandb project: {wandb_project}")

    for step, task in enumerate(tasks):
        files = get_task_files(root, task)
        if len(files) == 0:
            print(f"[{task}] No files found, skipping.")
            continue

        # Just take the first frame
        demo_idx, frame_idx, path = files[0]
        print(f"\n[{task}] Visualizing demo_{demo_idx}/frame_{frame_idx}: {path}")

        # Read and preprocess
        pts = read_ply(path, include_rgb=include_rgb)
        print(f"  Raw points: {pts.shape}")

        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)
            print(f"  After normalization: min={pts[:,:3].min(axis=0)}, max={pts[:,:3].max(axis=0)}")

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        # Compute bounds from this single frame
        pmin = pts_xyz.amin(dim=0)
        pmax = pts_xyz.amax(dim=0)
        bounds = (pmin, pmax)
        print(f"  Bounds: min={pmin.tolist()}, max={pmax.tolist()}")

        # Voxelize
        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=bounds, mode=voxel_mode)
        vox = vg.to_dense()  # [C, D, H, W]
        print(f"  Voxel shape: {vox.shape}, non-zero: {(vox.abs().sum(dim=0) > 0).sum().item()}")

        # Log to wandb using log_rgb_voxels
        log_rgb_voxels(
            name=f"debug/{task}",
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
        print(f"  Logged to wandb: debug/{task}")

    wandb.finish()
    print("\n[debug] Done! Check wandb for visualizations.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MimicGen point clouds to voxels")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing *_d0 task folders")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks to process (default: all)")
    parser.add_argument("--grid_whd", type=int, nargs=3, default=[64, 64, 64],
                        help="Voxel grid dimensions (default: 64 64 64)")
    parser.add_argument("--voxel_mode", type=str, default="avg_rgb",
                        choices=["occupancy", "density", "avg_rgb"],
                        help="Voxelization mode (default: avg_rgb)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Max points per cloud (default: no limit)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--no_rgb", action="store_true",
                        help="Don't include RGB (xyz only)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Don't normalize to unit cube")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: voxelize one frame per task and log to wandb")
    parser.add_argument("--wandb_project", type=str, default="mimicgen-voxel-debug",
                        help="Wandb project for debug visualizations (default: mimicgen-voxel-debug)")

    args = parser.parse_args()

    # Discover or use specified tasks
    if args.tasks is None:
        tasks = discover_tasks(args.root)
        print(f"Auto-discovered {len(tasks)} tasks: {tasks}")
    else:
        tasks = args.tasks
        print(f"Processing specified tasks: {tasks}")

    if len(tasks) == 0:
        print("No tasks found!")
        return

    # Debug mode: just visualize one frame per task
    if args.debug:
        debug_visualize_tasks(
            root=args.root,
            tasks=tasks,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            wandb_project=args.wandb_project,
        )
        return

    # Process each task
    for task in tasks:
        voxelize_task(
            root=args.root,
            task=task,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            max_points=args.max_points,
            force_rebuild=args.force,
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()
