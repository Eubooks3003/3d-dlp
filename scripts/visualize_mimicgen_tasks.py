#!/usr/bin/env python3
"""
Visualize one frame from each MimicGen task on wandb.
Logs both voxels and PLY point clouds for comparison.

Usage:
    python scripts/visualize_mimicgen_tasks.py --root /path/to/mimicgen
    python scripts/visualize_mimicgen_tasks.py --root /path/to/mimicgen --tasks coffee hammer
"""

import os
import sys
import argparse
import glob

import torch
import numpy as np
import open3d as o3d

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_vox import log_rgb_voxels
import wandb


def discover_tasks(root: str):
    """Auto-discover all *_d0 task folders."""
    task_dirs = sorted(glob.glob(os.path.join(root, "*_d0")))
    return [os.path.basename(d).replace("_d0", "") for d in task_dirs]


def get_first_voxel(root: str, task: str, cache_suffix: str = ""):
    """Get the first voxel from the first demo of a task."""
    cache_name = f"voxel_cache{cache_suffix}"
    cache_dir = os.path.join(root, f"{task}_d0", "core", cache_name)

    if not os.path.isdir(cache_dir):
        return None, None

    # Find first demo
    demo_dirs = sorted(
        glob.glob(os.path.join(cache_dir, "demo_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    if not demo_dirs:
        return None, None

    demo_dir = demo_dirs[0]

    # Find first frame
    vox_files = sorted(glob.glob(os.path.join(demo_dir, "frame*_voxels.pt")))
    if not vox_files:
        return None, None

    vox_path = vox_files[0]
    vox = torch.load(vox_path)

    return vox, vox_path


def get_first_ply(root: str, task: str):
    """Get the first PLY from the first demo of a task."""
    pcd_dir = os.path.join(root, f"{task}_d0", "core", "mimicgen_from_depth_pcd")

    if not os.path.isdir(pcd_dir):
        return None, None, None

    # Find first demo
    demo_dirs = sorted(
        glob.glob(os.path.join(pcd_dir, "demo_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    if not demo_dirs:
        return None, None, None

    demo_dir = demo_dirs[0]

    # Find first PLY
    ply_files = sorted(glob.glob(os.path.join(demo_dir, "*.ply")))
    if not ply_files:
        return None, None, None

    ply_path = ply_files[0]

    # Read PLY
    pc = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pc.points, dtype=np.float32)

    if len(pc.colors) > 0:
        rgb = np.asarray(pc.colors, dtype=np.float32)  # 0-1 range
        rgb = (rgb * 255).astype(np.uint8)  # Convert to 0-255 for wandb
    else:
        rgb = np.ones_like(xyz, dtype=np.uint8) * 128  # Gray if no color

    return xyz, rgb, ply_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MimicGen tasks on wandb")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing *_d0 task folders")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks to visualize (default: all)")
    parser.add_argument("--cache_suffix", type=str, default="",
                        help="Voxel cache suffix (e.g., '_debug')")
    parser.add_argument("--project", type=str, default="mimicgen-task-preview",
                        help="Wandb project name")
    parser.add_argument("--max_points", type=int, default=50000,
                        help="Max points to log for PLY (default: 50000)")

    args = parser.parse_args()

    # Discover tasks
    if args.tasks is None:
        tasks = discover_tasks(args.root)
        print(f"Auto-discovered {len(tasks)} tasks: {tasks}")
    else:
        tasks = args.tasks
        print(f"Visualizing specified tasks: {tasks}")

    if len(tasks) == 0:
        print("No tasks found!")
        return

    # Initialize wandb
    wandb.init(project=args.project, name=f"task-preview-{len(tasks)}-tasks")

    for step, task in enumerate(tasks):
        print(f"\n[{step+1}/{len(tasks)}] {task}")

        # --- Log Voxel ---
        vox, vox_path = get_first_voxel(args.root, task, args.cache_suffix)
        if vox is not None:
            print(f"  Voxel: {vox.shape} from {vox_path}")
            log_rgb_voxels(
                name=f"voxel/{task}",
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
        else:
            print(f"  Voxel: NOT FOUND")

        # --- Log PLY ---
        xyz, rgb, ply_path = get_first_ply(args.root, task)
        if xyz is not None:
            print(f"  PLY: {xyz.shape[0]} points from {ply_path}")

            # Downsample if needed
            if xyz.shape[0] > args.max_points:
                idx = np.random.choice(xyz.shape[0], args.max_points, replace=False)
                xyz = xyz[idx]
                rgb = rgb[idx]

            # Create point cloud array for wandb: [N, 6] with xyz + rgb
            pc_array = np.concatenate([xyz, rgb], axis=1)

            # Log to wandb
            wandb.log({
                f"ply/{task}": wandb.Object3D(pc_array)
            }, step=step)
        else:
            print(f"  PLY: NOT FOUND")

    wandb.finish()
    print(f"\nDone! Check wandb project: {args.project}")


if __name__ == "__main__":
    main()
