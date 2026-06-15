#!/usr/bin/env python3
"""
One-shot RLBench RGB-D -> 64^3 voxel cache, with no intermediate .ply on disk.

This collapses the two documented stages -- rlbench_ply.py (fuse RGB-D into a
point cloud) and preprocess_rlbench_voxels.py (voxelize the point cloud) -- into
a single pass. The fused point cloud is built in memory and voxelized directly,
so no `fused_pcd/*.ply` files are written. The `voxel_cache/` output is identical
to running the two stages back to back (same frame ordering, file names, meta,
and extras), so existing dataset loaders need no changes.

Input (standard RLBench layout, as produced by tools/dataset_generator.py):
    root/split/task/all_variations/episodes/episodeN/
        front_rgb/0.png, 1.png, ...
        front_depth/0.png, 1.png, ...
        overhead_rgb/...  overhead_depth/...
        low_dim_obs.pkl              # camera intrinsics/extrinsics (optional)

Output (same level as the camera folders):
    root/split/task/all_variations/episodes/episodeN/
        voxel_cache/
            manifest.json
            000000_voxels.pt
            000000_meta.pt
            000000_extras.pt
            ...

Usage:
    python scripts/rlbench_rgbd_to_voxels.py \
        --root /path/to/rlbench \
        --splits train_data test_data \
        --tasks close_jar open_drawer turn_tap \
        --cameras front overhead left_shoulder right_shoulder \
        --depth-scale 1000.0 \
        --grid_whd 64 64 64 \
        --voxel_mode avg_rgb

    python scripts/rlbench_rgbd_to_voxels.py --root /path/to/rlbench   # all splits/tasks
"""

import os
import sys
import argparse
import glob
import json

import numpy as np
import torch
from tqdm import tqdm

# scripts/ (this file's dir) is on sys.path[0] when run directly, so the two
# stage modules import cleanly. Also add the repo root for datasets.* imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fusion helpers -- reused verbatim from the PLY stage so geometry is identical.
from rlbench_ply import (
    get_default_intrinsics,
    DEFAULT_CAMERA_EXTRINSICS,
    load_camera_params_from_pkl,
    read_rgb,
    read_depth,
    backproject_depth,
    transform_points,
    crop_points,
    discover_episodes,
)

# Voxelization helpers -- reused verbatim from the voxel stage so the cache
# (voxels, meta, extras, manifest) is byte-for-byte the two-stage output.
from preprocess_rlbench_voxels import (
    safe_tensor,
    center_scale_unit_cube,
    get_episode_info,
)
from datasets.voxelize_ds_wrapper import VoxelGridXYZ

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Install with: pip install pillow")
    sys.exit(1)


def fuse_frame(
    episode_dir: str,
    fname: str,
    available_cams: list,
    cam_params: dict,
    depth_scale: float,
    pixel_stride: int,
    include_rgb: bool,
    crop_bounds: dict = None,
    max_points: int = 200000,
) -> np.ndarray:
    """Fuse one frame's multi-view RGB-D into a point cloud in memory.

    Mirrors the per-frame body of rlbench_ply.process_episode, but returns the
    fused [N, 3] (xyz) or [N, 6] (xyz+rgb) array instead of writing a .ply.
    Returns a (0, C) array if the frame yields no valid points.
    """
    all_xyz = []
    all_rgb = []

    for cam in available_cams:
        rgb_path_cam = os.path.join(episode_dir, f"{cam}_rgb", fname)
        depth_path_cam = os.path.join(episode_dir, f"{cam}_depth", fname)
        if not os.path.exists(rgb_path_cam) or not os.path.exists(depth_path_cam):
            continue

        rgb = read_rgb(rgb_path_cam)
        depth = read_depth(depth_path_cam, scale=depth_scale)

        K_cam, T_cam2world = cam_params[cam]

        pts_cam, pixel_idx = backproject_depth(depth, K_cam, pixel_stride)
        if pts_cam.shape[0] == 0:
            continue

        pts_world = transform_points(pts_cam, T_cam2world)
        all_xyz.append(pts_world)

        if include_rgb:
            v_idx, u_idx = pixel_idx[:, 0], pixel_idx[:, 1]
            all_rgb.append(rgb[v_idx, u_idx])

    if len(all_xyz) == 0:
        return np.zeros((0, 6 if include_rgb else 3), np.float32)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb_colors = (
        np.concatenate(all_rgb, axis=0)
        if include_rgb
        else np.zeros((xyz.shape[0], 3), np.float32)
    )

    xyz, rgb_colors = crop_points(xyz, rgb_colors, crop_bounds)
    if xyz.shape[0] == 0:
        return np.zeros((0, 6 if include_rgb else 3), np.float32)

    # Same fuse-stage downsample as rlbench_ply.process_episode.
    if max_points and max_points > 0 and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        rgb_colors = rgb_colors[idx]

    if include_rgb:
        return np.concatenate([xyz, rgb_colors], axis=-1).astype(np.float32)
    return xyz.astype(np.float32)


def process_episode(
    episode_dir: str,
    cameras: list,
    depth_scale: float = 1000.0,
    pixel_stride: int = 1,
    fov_deg: float = 60.0,
    crop_bounds: dict = None,
    max_fuse_points: int = 200000,
    grid_whd: tuple = (64, 64, 64),
    voxel_mode: str = "avg_rgb",
    include_rgb: bool = True,
    normalize_to_unit_cube: bool = True,
    max_points: int = None,
    force_rebuild: bool = False,
) -> int:
    """Fuse + voxelize every frame of one episode straight into voxel_cache/."""
    cache_dir = os.path.join(episode_dir, "voxel_cache")
    manifest_path = os.path.join(cache_dir, "manifest.json")

    # Which camera folders actually exist for this episode.
    available_cams = []
    for cam in cameras:
        rgb_dir = os.path.join(episode_dir, f"{cam}_rgb")
        depth_dir = os.path.join(episode_dir, f"{cam}_depth")
        if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
            available_cams.append(cam)
    if len(available_cams) == 0:
        return 0

    # Frame list from the first available camera, ordered numerically by frame
    # index. This matches the two-stage output order: rlbench_ply writes
    # {frame_idx:06d}.ply, and preprocess_rlbench_voxels re-sorts those names
    # (zero-padded -> lexicographic == numeric) before enumerating cache idx.
    first_cam = available_cams[0]
    rgb_files = glob.glob(os.path.join(episode_dir, f"{first_cam}_rgb", "*.png"))
    if len(rgb_files) == 0:
        return 0
    frames = sorted(
        (int(os.path.splitext(os.path.basename(p))[0]), os.path.basename(p))
        for p in rgb_files
    )
    n_frames = len(frames)

    # Skip episodes already fully cached.
    if not force_rebuild and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            mani = json.load(f)
        if mani.get("complete", False) and mani.get("length", 0) >= n_frames:
            return 0

    os.makedirs(cache_dir, exist_ok=True)
    ep_info = get_episode_info(episode_dir)

    # Camera intrinsics: identical fallback chain to rlbench_ply (pkl -> default).
    sample = Image.open(os.path.join(episode_dir, f"{first_cam}_rgb", frames[0][1]))
    width, height = sample.size
    K = get_default_intrinsics(width, height, fov_deg)
    cam_params = {}
    for cam in available_cams:
        loaded = load_camera_params_from_pkl(episode_dir, cam)
        if loaded is not None:
            cam_params[cam] = loaded
        else:
            cam_params[cam] = (K.copy(), DEFAULT_CAMERA_EXTRINSICS.get(cam, np.eye(4, dtype=np.float32)))

    frames_done = 0

    for idx, (frame_idx, fname) in enumerate(frames):
        vox_path = os.path.join(cache_dir, f"{idx:06d}_voxels.pt")
        meta_path = os.path.join(cache_dir, f"{idx:06d}_meta.pt")
        extras_path = os.path.join(cache_dir, f"{idx:06d}_extras.pt")

        if not force_rebuild and os.path.exists(vox_path) and os.path.exists(meta_path):
            frames_done += 1
            continue

        pts = fuse_frame(
            episode_dir, fname, available_cams, cam_params,
            depth_scale=depth_scale, pixel_stride=pixel_stride,
            include_rgb=include_rgb, crop_bounds=crop_bounds,
            max_points=max_fuse_points,
        )
        if pts.shape[0] == 0:
            continue

        # ----- voxel stage: identical to preprocess_rlbench_voxels.voxelize_episode -----
        if normalize_to_unit_cube:
            pts = center_scale_unit_cube(pts)

        if max_points is not None and pts.shape[0] > max_points:
            idx_sample = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx_sample]

        pts_xyz = safe_tensor(pts[:, :3])
        colors = safe_tensor(pts[:, 3:6]) if (pts.shape[-1] >= 6 and voxel_mode == "avg_rgb") else None

        vg = VoxelGridXYZ(pts_xyz, colors, grid_whd=grid_whd, bounds=None, mode=voxel_mode)
        vox = vg.to_dense().cpu()
        md = vg.meta_dict()
        md_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in md.items()}

        torch.save(vox, vox_path)
        torch.save(md_cpu, meta_path)

        # Virtual ply path keeps the extras schema identical to the two-stage flow.
        virtual_ply = os.path.join(episode_dir, "fused_pcd", f"{frame_idx:06d}.ply")
        extras = {
            "path": virtual_ply,
            "id": f"{frame_idx:06d}",
            "split": ep_info["split"],
            "task": ep_info["task"],
            "episode": ep_info["episode"],
            "frame": frame_idx,
            "points": pts,
        }
        torch.save(extras, extras_path)
        frames_done += 1

    manifest = {
        "split": ep_info["split"],
        "task": ep_info["task"],
        "episode": ep_info["episode"],
        "length": n_frames,
        "grid_whd": list(grid_whd),
        "mode": voxel_mode,
        "bounds_mode": "per_item",
        "complete": True,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return frames_done


def main():
    parser = argparse.ArgumentParser(
        description="One-shot RLBench RGB-D -> voxel cache (no intermediate .ply)")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of RLBench data")
    parser.add_argument("--splits", type=str, nargs="*", default=None,
                        help="Splits to process (default: all)")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Tasks to process (default: all)")
    # --- fuse stage ---
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["front", "overhead", "left_shoulder", "right_shoulder"],
                        help="Camera views to fuse")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Depth scale factor (1000 for mm, 1 for meters)")
    parser.add_argument("--pixel-stride", type=int, default=1,
                        help="Pixel subsampling stride (default: 1)")
    parser.add_argument("--fov", type=float, default=60.0,
                        help="Camera field of view in degrees (default: 60)")
    parser.add_argument("--max-fuse-points", type=int, default=200000,
                        help="Max fused points per frame before voxelization (default: 200000)")
    parser.add_argument("--crop-xmin", type=float, default=None)
    parser.add_argument("--crop-xmax", type=float, default=None)
    parser.add_argument("--crop-ymin", type=float, default=None)
    parser.add_argument("--crop-ymax", type=float, default=None)
    parser.add_argument("--crop-zmin", type=float, default=None)
    parser.add_argument("--crop-zmax", type=float, default=None)
    # --- voxel stage ---
    parser.add_argument("--grid_whd", type=int, nargs=3, default=[64, 64, 64],
                        help="Voxel grid dimensions (default: 64 64 64)")
    parser.add_argument("--voxel_mode", type=str, default="avg_rgb",
                        choices=["occupancy", "density", "avg_rgb"],
                        help="Voxelization mode (default: avg_rgb)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Max points fed to the voxelizer (default: no limit)")
    parser.add_argument("--no_rgb", action="store_true",
                        help="Don't include RGB (xyz only)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Don't normalize to unit cube")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Max episodes to process (-1 for all)")

    args = parser.parse_args()

    crop_bounds = None
    crop_vals = [args.crop_xmin, args.crop_xmax, args.crop_ymin,
                 args.crop_ymax, args.crop_zmin, args.crop_zmax]
    if any(v is not None for v in crop_vals):
        crop_bounds = {}
        for key, val in zip(
            ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"], crop_vals):
            if val is not None:
                crop_bounds[key] = val

    print(f"Root: {args.root}")
    print(f"Cameras: {args.cameras}")
    print(f"Grid: {args.grid_whd}  Mode: {args.voxel_mode}")
    print(f"Crop bounds: {crop_bounds}")

    episodes = discover_episodes(args.root, args.splits, args.tasks)
    print(f"Found {len(episodes)} episodes")

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
        print(f"Processing first {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No episodes found!")
        return

    total_frames = 0
    for episode_dir in tqdm(episodes, desc="Fuse+voxelize episodes"):
        total_frames += process_episode(
            episode_dir=episode_dir,
            cameras=args.cameras,
            depth_scale=args.depth_scale,
            pixel_stride=args.pixel_stride,
            fov_deg=args.fov,
            crop_bounds=crop_bounds,
            max_fuse_points=args.max_fuse_points,
            grid_whd=tuple(args.grid_whd),
            voxel_mode=args.voxel_mode,
            include_rgb=not args.no_rgb,
            normalize_to_unit_cube=not args.no_normalize,
            max_points=args.max_points,
            force_rebuild=args.force,
        )

    print(f"\nDone! Voxelized {total_frames} frames across {len(episodes)} episodes")


if __name__ == "__main__":
    main()
