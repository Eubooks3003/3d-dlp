#!/usr/bin/env python3
"""
Visualize one frame from each MimicGen task on wandb.

Usage:
    python scripts/visualize_mimicgen_tasks.py --root /path/to/3D-DLP-mimicgen-data
    python scripts/visualize_mimicgen_tasks.py --root /path/to/3D-DLP-mimicgen-data --tasks coffee_d0 coffee_d2 hammer_cleanup_d1
"""

import os
import sys
import argparse
import glob

import torch

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_vox import log_rgb_voxels
import wandb


def discover_tasks(root: str):
    """Auto-discover all task folders that have a voxel cache."""
    tasks = []
    for d in sorted(os.listdir(root)):
        vox_dir = os.path.join(root, d, "voxel_cache", "voxel")
        if os.path.isdir(vox_dir):
            tasks.append(d)
    return tasks


def load_voxel_dense(path):
    """Load compressed or dense voxel, return dense [C,D,H,W] float32."""
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and data.get("compressed"):
        shape = data["shape"]
        coords = data["coords"].long()
        values = data["values"].float()
        vox = torch.zeros(shape, dtype=torch.float32)
        vox[:, coords[:, 0], coords[:, 1], coords[:, 2]] = values.T
        return vox
    if isinstance(data, torch.Tensor):
        return data.float()
    raise RuntimeError(f"Unexpected format in {path}: {type(data)}")


def get_first_voxel(root: str, task: str):
    """Get the first voxel from the first demo of a task."""
    cache_dir = os.path.join(root, task, "voxel_cache", "voxel")

    if not os.path.isdir(cache_dir):
        return None, None

    demo_dirs = sorted(
        glob.glob(os.path.join(cache_dir, "demo_*")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    if not demo_dirs:
        return None, None

    demo_dir = demo_dirs[0]

    vox_files = sorted(glob.glob(os.path.join(demo_dir, "frame*_voxels.pt")))
    if not vox_files:
        return None, None

    vox_path = vox_files[0]
    vox = load_voxel_dense(vox_path)

    return vox, vox_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MimicGen tasks on wandb")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing task folders")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks to visualize (default: all with voxel caches)")
    parser.add_argument("--project", type=str, default="mimicgen-task-preview",
                        help="Wandb project name")

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

        vox, vox_path = get_first_voxel(args.root, task)
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

    wandb.finish()
    print(f"\nDone! Check wandb project: {args.project}")


if __name__ == "__main__":
    main()
