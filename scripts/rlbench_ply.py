#!/usr/bin/env python3
"""
Convert RLBench RGB + depth images to fused point cloud PLY files.

RLBench data structure:
    root/split/task/all_variations/episodes/episodeN/
        front_rgb/0.png, 1.png, ...
        front_depth/0.png, 1.png, ...
        overhead_rgb/...
        overhead_depth/...
        low_dim_obs.pkl
        ...

Output structure (same level as camera folders):
    root/split/task/all_variations/episodes/episodeN/
        fused_pcd/
            000000.ply
            000001.ply
            ...

Usage:
    python rlbench_ply.py --root /path/to/rlbench --splits train test --tasks slide_block_to_color_target
    python rlbench_ply.py --root /path/to/rlbench  # Process all splits and tasks
"""

import os
import sys
import argparse
import glob
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print("open3d not installed. Install with: pip install open3d")
    sys.exit(1)


# ============== Default RLBench Camera Parameters ==============
# These are approximate defaults for standard RLBench setup.
# Intrinsics assume 128x128 images with ~60 degree FOV.
# Extrinsics are approximate and may need tuning for your specific setup.

def get_default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """
    Compute camera intrinsic matrix from image size and field of view.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Horizontal field of view in degrees

    Returns:
        K: 3x3 intrinsic matrix
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # Square pixels
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    return K


# Default camera extrinsics (camera-to-world transforms)
# These are approximate for standard RLBench - adjust if needed
DEFAULT_CAMERA_EXTRINSICS = {
    "front": np.array([
        [1.0,  0.0,  0.0,  1.75],   # Looking from front, ~1.75m away
        [0.0,  0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32),
    "overhead": np.array([
        [1.0,  0.0,  0.0,  0.0],   # Looking from above
        [0.0,  1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  1.5],   # ~1.5m above
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32),
    "left_shoulder": np.array([
        [0.0, -1.0,  0.0, -0.3],
        [0.0,  0.0,  1.0,  0.5],
        [-1.0, 0.0,  0.0,  1.2],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32),
    "right_shoulder": np.array([
        [0.0,  1.0,  0.0,  0.3],
        [0.0,  0.0,  1.0, -0.5],
        [1.0,  0.0,  0.0,  1.2],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float32),
    "wrist": np.eye(4, dtype=np.float32),  # Wrist camera moves with end-effector
}


def load_camera_params_from_pkl(episode_dir: str, cam: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Try to load camera parameters from low_dim_obs.pkl if available.

    Returns:
        (K, T_cam2world) if found, else None
    """
    pkl_path = os.path.join(episode_dir, "low_dim_obs.pkl")
    if not os.path.exists(pkl_path):
        return None

    try:
        with open(pkl_path, "rb") as f:
            obs = pickle.load(f)

        # RLBench stores camera info in different places depending on version
        # Try common locations
        if hasattr(obs, "misc") and obs.misc is not None:
            misc = obs.misc
            intrinsic_key = f"{cam}_camera_intrinsics"
            extrinsic_key = f"{cam}_camera_extrinsics"

            if intrinsic_key in misc and extrinsic_key in misc:
                K = np.array(misc[intrinsic_key], dtype=np.float32)
                T = np.array(misc[extrinsic_key], dtype=np.float32)
                return K, T
    except Exception as e:
        print(f"  Warning: Could not parse camera params from pkl: {e}")

    return None


def read_rgb(path: str) -> np.ndarray:
    """Read RGB image as float32 array [H, W, 3] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def read_depth(path: str, scale: float = 1000.0) -> np.ndarray:
    """
    Read depth image as float32 array [H, W] in meters.

    Args:
        path: Path to depth PNG
        scale: Divisor to convert to meters (1000 for mm, 1 for already in meters)

    Returns:
        Depth map in meters
    """
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)

    # Handle different depth formats
    if arr.ndim == 3:
        # Sometimes depth is stored as RGB - take first channel
        arr = arr[:, :, 0]

    # Convert to meters
    depth_m = arr / scale

    # Mark invalid depth (0 or very large values) as NaN
    depth_m[depth_m <= 0] = np.nan
    depth_m[depth_m > 10.0] = np.nan  # Clip unreasonable depths

    return depth_m


def backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    pixel_stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject depth map to 3D points in camera coordinates.

    Args:
        depth: [H, W] depth map in meters
        K: [3, 3] camera intrinsic matrix
        pixel_stride: Subsample pixels for speed (1 = all pixels)

    Returns:
        pts_cam: [N, 3] points in camera coordinates
        pixel_idx: [N, 2] corresponding pixel indices (v, u)
    """
    H, W = depth.shape

    # Create pixel grid
    vv = np.arange(0, H, pixel_stride, dtype=np.int32)
    uu = np.arange(0, W, pixel_stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)

    Z = depth[V, U]
    valid = np.isfinite(Z) & (Z > 0)

    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

    Uv = U[valid].astype(np.float32)
    Vv = V[valid].astype(np.float32)
    Zv = Z[valid].astype(np.float32)

    # Unproject
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (Uv - cx) / fx * Zv
    Y = (Vv - cy) / fy * Zv

    pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)
    pixel_idx = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)

    return pts_cam, pixel_idx


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform points by 4x4 matrix."""
    if pts.shape[0] == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_out = (T.astype(np.float32) @ pts_h.T).T
    return pts_out[:, :3]


def crop_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    bounds: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop points to bounding box."""
    if bounds is None:
        return xyz, rgb

    mask = (
        (xyz[:, 0] >= bounds.get("xmin", -np.inf)) &
        (xyz[:, 0] <= bounds.get("xmax", np.inf)) &
        (xyz[:, 1] >= bounds.get("ymin", -np.inf)) &
        (xyz[:, 1] <= bounds.get("ymax", np.inf)) &
        (xyz[:, 2] >= bounds.get("zmin", -np.inf)) &
        (xyz[:, 2] <= bounds.get("zmax", np.inf))
    )

    if not np.any(mask):
        return xyz, rgb

    return xyz[mask], rgb[mask]


def process_episode(
    episode_dir: str,
    cameras: List[str],
    depth_scale: float = 1000.0,
    pixel_stride: int = 1,
    max_points: int = 200000,
    crop_bounds: Optional[Dict[str, float]] = None,
    fov_deg: float = 60.0,
    force: bool = False,
) -> int:
    """
    Process a single episode: fuse RGB+depth from all cameras into PLY files.

    Returns:
        Number of frames processed
    """
    out_dir = os.path.join(episode_dir, "fused_pcd")

    # Check what camera folders exist
    available_cams = []
    for cam in cameras:
        rgb_dir = os.path.join(episode_dir, f"{cam}_rgb")
        depth_dir = os.path.join(episode_dir, f"{cam}_depth")
        if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
            available_cams.append(cam)

    if len(available_cams) == 0:
        return 0

    # Get frame count from first available camera
    first_cam = available_cams[0]
    rgb_files = sorted(glob.glob(os.path.join(episode_dir, f"{first_cam}_rgb", "*.png")))
    if len(rgb_files) == 0:
        return 0

    # Check if already done
    if not force and os.path.isdir(out_dir):
        existing = glob.glob(os.path.join(out_dir, "*.ply"))
        if len(existing) >= len(rgb_files):
            return 0  # Already processed

    os.makedirs(out_dir, exist_ok=True)

    # Get image size from first image
    sample_img = Image.open(rgb_files[0])
    width, height = sample_img.size

    # Get camera intrinsics (same for all cameras in standard RLBench)
    K = get_default_intrinsics(width, height, fov_deg)

    # Try to load from pkl, fall back to defaults
    cam_params = {}
    for cam in available_cams:
        loaded = load_camera_params_from_pkl(episode_dir, cam)
        if loaded is not None:
            cam_params[cam] = loaded
        else:
            T_default = DEFAULT_CAMERA_EXTRINSICS.get(cam, np.eye(4, dtype=np.float32))
            cam_params[cam] = (K.copy(), T_default)

    frames_done = 0

    for rgb_path in rgb_files:
        fname = os.path.basename(rgb_path)
        frame_idx = int(os.path.splitext(fname)[0])
        out_path = os.path.join(out_dir, f"{frame_idx:06d}.ply")

        if not force and os.path.exists(out_path):
            frames_done += 1
            continue

        all_xyz = []
        all_rgb = []

        for cam in available_cams:
            rgb_path_cam = os.path.join(episode_dir, f"{cam}_rgb", fname)
            depth_path_cam = os.path.join(episode_dir, f"{cam}_depth", fname)

            if not os.path.exists(rgb_path_cam) or not os.path.exists(depth_path_cam):
                continue

            # Read data
            rgb = read_rgb(rgb_path_cam)
            depth = read_depth(depth_path_cam, scale=depth_scale)

            # Get camera params
            K_cam, T_cam2world = cam_params[cam]

            # Backproject to camera coords
            pts_cam, pixel_idx = backproject_depth(depth, K_cam, pixel_stride)
            if pts_cam.shape[0] == 0:
                continue

            # Get colors
            v_idx, u_idx = pixel_idx[:, 0], pixel_idx[:, 1]
            colors = rgb[v_idx, u_idx]

            # Transform to world coords
            pts_world = transform_points(pts_cam, T_cam2world)

            all_xyz.append(pts_world)
            all_rgb.append(colors)

        if len(all_xyz) == 0:
            continue

        # Fuse all cameras
        xyz = np.concatenate(all_xyz, axis=0)
        rgb_colors = np.concatenate(all_rgb, axis=0)

        # Crop
        xyz, rgb_colors = crop_points(xyz, rgb_colors, crop_bounds)

        if xyz.shape[0] == 0:
            continue

        # Random downsample if too many points
        if max_points > 0 and xyz.shape[0] > max_points:
            idx = np.random.choice(xyz.shape[0], max_points, replace=False)
            xyz = xyz[idx]
            rgb_colors = rgb_colors[idx]

        # Save PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors.astype(np.float64))
        o3d.io.write_point_cloud(out_path, pcd)

        frames_done += 1

    return frames_done


def discover_episodes(
    root: str,
    splits: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
) -> List[str]:
    """
    Discover all episode directories.

    Returns:
        List of episode directory paths
    """
    # Find all splits
    if splits is None:
        split_dirs = glob.glob(os.path.join(root, "*"))
        splits = [os.path.basename(d) for d in split_dirs if os.path.isdir(d)]

    episodes = []

    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue

        # Find tasks
        if tasks is None:
            task_dirs = glob.glob(os.path.join(split_path, "*"))
            task_names = [os.path.basename(d) for d in task_dirs if os.path.isdir(d)]
        else:
            task_names = tasks

        for task in task_names:
            # Standard RLBench structure
            episodes_root = os.path.join(split_path, task, "all_variations", "episodes")
            if not os.path.isdir(episodes_root):
                # Try without all_variations
                episodes_root = os.path.join(split_path, task, "episodes")

            if not os.path.isdir(episodes_root):
                continue

            episode_dirs = sorted(
                glob.glob(os.path.join(episodes_root, "episode*")),
                key=lambda x: int(os.path.basename(x).replace("episode", ""))
            )
            episodes.extend(episode_dirs)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Convert RLBench RGB+depth to PLY point clouds")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of RLBench data")
    parser.add_argument("--splits", type=str, nargs="*", default=None,
                        help="Splits to process (default: all)")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Tasks to process (default: all)")
    parser.add_argument("--cameras", type=str, nargs="+", default=["front", "overhead"],
                        help="Camera views to fuse (default: front overhead)")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Depth scale factor (1000 for mm, 1 for meters)")
    parser.add_argument("--pixel-stride", type=int, default=1,
                        help="Pixel subsampling stride (default: 1)")
    parser.add_argument("--max-points", type=int, default=200000,
                        help="Max points per frame (default: 200000)")
    parser.add_argument("--fov", type=float, default=60.0,
                        help="Camera field of view in degrees (default: 60)")
    parser.add_argument("--crop-xmin", type=float, default=None)
    parser.add_argument("--crop-xmax", type=float, default=None)
    parser.add_argument("--crop-ymin", type=float, default=None)
    parser.add_argument("--crop-ymax", type=float, default=None)
    parser.add_argument("--crop-zmin", type=float, default=None)
    parser.add_argument("--crop-zmax", type=float, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if PLY exists")
    parser.add_argument("--max-episodes", type=int, default=-1,
                        help="Max episodes to process (-1 for all)")

    args = parser.parse_args()

    # Build crop bounds
    crop_bounds = None
    if any([args.crop_xmin, args.crop_xmax, args.crop_ymin,
            args.crop_ymax, args.crop_zmin, args.crop_zmax]):
        crop_bounds = {}
        if args.crop_xmin is not None: crop_bounds["xmin"] = args.crop_xmin
        if args.crop_xmax is not None: crop_bounds["xmax"] = args.crop_xmax
        if args.crop_ymin is not None: crop_bounds["ymin"] = args.crop_ymin
        if args.crop_ymax is not None: crop_bounds["ymax"] = args.crop_ymax
        if args.crop_zmin is not None: crop_bounds["zmin"] = args.crop_zmin
        if args.crop_zmax is not None: crop_bounds["zmax"] = args.crop_zmax

    print(f"Root: {args.root}")
    print(f"Cameras: {args.cameras}")
    print(f"Depth scale: {args.depth_scale}")
    print(f"Crop bounds: {crop_bounds}")

    # Discover episodes
    episodes = discover_episodes(args.root, args.splits, args.tasks)
    print(f"Found {len(episodes)} episodes")

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
        print(f"Processing first {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No episodes found!")
        return

    # Process each episode
    total_frames = 0
    for episode_dir in tqdm(episodes, desc="Processing episodes"):
        n_frames = process_episode(
            episode_dir=episode_dir,
            cameras=args.cameras,
            depth_scale=args.depth_scale,
            pixel_stride=args.pixel_stride,
            max_points=args.max_points,
            crop_bounds=crop_bounds,
            fov_deg=args.fov,
            force=args.force,
        )
        total_frames += n_frames

    print(f"\nDone! Processed {total_frames} total frames across {len(episodes)} episodes")


if __name__ == "__main__":
    main()
